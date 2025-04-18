import numpy as np
import torch
import torch.nn as nn
from torch import optim

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel
from einops import rearrange
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

class GPT4TS(nn.Module):
    
    def __init__(self, configs, device):
        super(GPT4TS, self).__init__()
        self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        if configs.is_gpt:
            if configs.pretrain:
                self.gpt2 = GPT2Model.from_pretrained('/data_disk/lichx/Model_from_HF/GPT2', output_attentions=True, output_hidden_states=True)
            else:
                print("------------------no pretrain------------------")
                self.gpt2 = GPT2Model(GPT2Config())
            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
                
        self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        self.fc = nn.Linear(configs.d_model * self.patch_num, configs.fc_layer)
        self.out_layer = nn.Linear(configs.fc_layer, configs.pred_len)
        self.leaky_relu = nn.LeakyReLU()

        if configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = False

        for layer in (self.gpt2, self.in_layer, self.out_layer, self.fc):
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
            outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state
        
        outputs = self.fc(outputs.reshape(B*M, -1))
        outputs = self.leaky_relu(outputs)
        outputs = self.out_layer(outputs.reshape(B*M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        outputs = outputs * stdev
        outputs = outputs + means

        return outputs
