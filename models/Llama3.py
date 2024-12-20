import numpy as np
import torch
import torch.nn as nn
from torch import optim

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from einops import rearrange

class Llama3(nn.Module):
    
    def __init__(self, configs, device):
        super(Llama3, self).__init__()
        self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        self.quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        if configs.is_gpt:
            if configs.pretrain:
                self.llama3 = AutoModelForCausalLM.from_pretrained('/data_disk/lichx/Model_from_HF/LLAMA3',
                                                        output_hidden_states=True,
                                                        attn_implementation="eager",
                                                        torch_dtype=torch.float16,
                                                        low_cpu_mem_usage=True,
                                                        quantization_config=self.quantization_config
                                                        )
            else:
                print("------------------no pretrain------------------")


        self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        self.fc = nn.Linear(configs.d_model * self.patch_num, configs.fc_layer)
        self.out_layer = nn.Linear(configs.fc_layer, configs.pred_len)
        self.leaky_relu = nn.LeakyReLU()

        if configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.llama3.named_parameters()):
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = False

        for layer in (self.in_layer, self.out_layer, self.fc):
            layer.to(device=device)
            layer.train()
        
        self.cnt = 0


    def forward(self, x, itr):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride) 
        x = rearrange(x, 'b m n p -> (b m) n p')
        outputs = self.in_layer(x)
        if self.is_gpt:
            outputs = self.llama3(inputs_embeds=outputs).hidden_states[-1]
  
        outputs = self.fc(outputs.reshape(B*M, -1))
        outputs = self.leaky_relu(outputs)
        outputs = self.out_layer(outputs.reshape(B*M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        outputs = outputs * stdev
        outputs = outputs + means

        return outputs
