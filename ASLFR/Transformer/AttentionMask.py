import torch
import torch.nn as nn
import torch.nn.functional as F

N_TARGET_FRAMES = 128

class AttentionMask(nn.Module):
    def __init__(self):
        super().__init__()

    def get_attention_mask(self, x_inp):
        attention_mask = torch.count_nonzero( x_inp, dim = 2, keepdim =True, dtype = torch.int32)
        attention_mask = torch.count_nonzero(attention_mask, dim =2, keepdim = False)
        attention_mask = attention_mask.unsqueeze(1)
        attention_mask = attention_mask.unsqueeze(1)

        return attention_mask
    
    def get_casual_attention_mask(self):
        i = torch.arange(N_TARGET_FRAMES).unsqueeze(1)
        j = torch.arange(N_TARGET_FRAMES)
        mask = (i > j).to(torch.int32)
        mask = mask.view(1, self.N_TARGET_FRAMES, N_TARGET_FRAMES)
        mult = torch.cat(
            [torch.tensor([1]).unsqueeze(-1), torch.tensor([1, 1], dtype = torch.int32)],
            dim = 0
        )
        mask = mask.repeat(*mult)
        mask = mask.to(torch.float32)
        
        return mask