import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.checkpoint import checkpoint

from transformers import LlamaForCausalLM, LlamaModel, BitsAndBytesConfig
from einops import rearrange


class Llama2(nn.Module):
    
    def __init__(self, configs, device):
        super(Llama2, self).__init__()
        self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        # self.quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        if configs.is_gpt:
            if configs.pretrain:
                try:
                    model_dir = '/data_disk/lichx/Model_from_HF/LLAMA2'
                    self.llama2 = LlamaForCausalLM.from_pretrained(model_dir,
                                                                    output_attentions=True,
                                                                    output_hidden_states=True,
                                                                    torch_dtype=torch.float16,
                                                                    # quantization_config=self.quantization_config
                                                                    )
                    print('------------------Llma2 loading completed------------------')
                except Exception as e:
                    print(f'Error: {e}')
            else:
                print("------------------no pretrain------------------")
                # self.llama2 = LlamaForCausalLM(GPT2Config())

            self.llama2.model.layers = self.llama2.model.layers[:configs.llama_layers]

        self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        self.fc = nn.Linear(configs.d_model * self.patch_num, configs.fc_layer)
        self.out_layer = nn.Linear(configs.fc_layer, configs.pred_len)
        self.leaky_relu = nn.LeakyReLU()

        if configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.llama2.model.named_parameters()): 
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = False ##
                else:
                    param.requires_grad = False

        for layer in (self.llama2.model, self.fc, self.in_layer, self.out_layer):
            layer.to(device=device)
            layer.train()
        
        self.cnt = 0
    
    def forward(self, x, itr, mark=None):
        B, L, M = x.shape

        if mark is not None:
            mark = mark.to(next(self.time_embed.parameters()).dtype)  
            time_embedding = self.time_embed(mark)

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        x /= stdev

        if mark is not None:
            x = x + time_embedding

        x = rearrange(x, 'b l m -> b m l')
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride) 
        x = rearrange(x, 'b m n p -> (b m) n p')
        outputs = self.in_layer(x)

        if self.is_gpt:
            outputs = checkpoint(lambda inp: self.llama2.model(inputs_embeds=inp).last_hidden_state, outputs)

        outputs = checkpoint(lambda inp: self.fc(inp), outputs.reshape(B * M, -1))
        outputs = self.leaky_relu(outputs)
        outputs = self.out_layer(outputs.reshape(B*M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        outputs = outputs * stdev
        outputs = outputs + means

        return outputs
