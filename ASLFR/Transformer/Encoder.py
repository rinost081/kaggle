import torch
import torch.nn as nn
import torch.nn.functional as F
import MultiHeadAttention
import AttentionMask

N_UNIQUE_CHARACTERS = 58 + 3
UNITS_ENCODER = 384
UNITS_DECODER = 256
N_TARGET_FRAMES = 128
MAX_PHRASE_LENGTH = 31 + 1
SOS_TOKEN = 58 + 1
PAD_TOKEN = 58 + 2
EMBEDDING_DROPOUT = 0.00
MLP_DROPOUT_RATIO = 0.30
MHA_DROPOUT_RATIO = 0.20
CLASSIFIER_DROPOUT_RATIO = 0.10
LAYER_NORM_EPS = 1e-6
MLP_RATIO = 2
NUM_HEADS = 4

class Encoder(nn.Module):
    def __init__(self, num_blocks):
        super().__init__()
        
        self.num_blocks = num_blocks
        self.supports_masking = True
        
        self.ln = nn.LayerNorm(UNITS_ENCODER, eps = LAYER_NORM_EPS)
        self.mha = MultiHeadAttention(UNITS_ENCODER, NUM_HEADS, MHA_DROPOUT_RATIO)
        
        self.fc1 = nn.Linear(UNITS_ENCODER, UNITS_ENCODER*MLP_RATIO, bias = False)
        self.fc2 = nn.Linear(UNITS_ENCODER*MLP_RATIO, UNITS_ENCODER, bias = False)
        self.fc3 = nn.Linear(UNITS_ENCODER, UNITS_DECODER, bias = False)
    
        
        nn.init.xavier_uniform_(self.fc3.weight)
        self.dropout = nn.Dropout(MLP_DROPOUT_RATIO)
        
        self.mlp = nn.Sequential(
            self.fc1,
            nn.init.xavier_uniform_(self.fc1.weight),
            nn.GELU(),
            nn.Dropout(MLP_DROPOUT_RATIO),
            self.fc2,
            nn.init.xavier_uniform_(self.fc2.weight)
            )
        
    
        self.attention_mask = AttentionMask()
        
        
    def forward(self, x, x_inp):
        attention_mask = self.attention_mask.get_attention_mask(x_inp)
        for i in range(self.num_blocks):
            x = self.ln(x + self.mha(x, x, x, attention_mask = attention_mask))
            x = self.ln(x + self.mlp(x))
            
        x = self.fc3(x)
        
        return x
    


