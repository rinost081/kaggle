import torch
import torch.nn as nn
import torch.nn.functional as F

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

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout_ratio, d_out = None):
        super().__init__()
        
        self.depth = d_model // 2
        self.scale = 1.0 / torch.sqrt(torch.tensor(self.depth, dtype = torch.float32))
        
        self.softmax = nn.Softmax()

        self.fc1 = nn.Linear(self.depth, self.depth, bias = False)
        self.fc2 = nn.Linear(self.depth, d_model, bias = False)
        
        self.reshape = nn.Sequential(
            nn.Unflatten(1, (N_TARGET_FRAMES, n_heads, self.depth // n_heads)),
            nn.Permute(2, 1, 3)
        )
        self.dropout = nn.Dropout(dropout_ratio)
        self.supports_masking = True
        
    def forward(self, q, k, v, attention_mask = None):
        Q = self.reshape(self.fc1(q))
        K = self.reshape(self.fc1(k))
        V = self.reshape(self.fc1(v))
        
        x = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        x = self.softmax(x, mask = attention_mask) @ V
        x = self.reshape(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        return x