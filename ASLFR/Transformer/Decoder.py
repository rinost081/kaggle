import torch
import torch.nn as nn
import torch.nn.functional as F
import MultiHeadAttention
import AttentionMask

N_UNIQUE_CHARACTERS = 58 + 3
UNITS_ENCODER = 384
UNITS_DECODER = 256
NUM_HEADS = 4
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

class Decoder(nn.Module):
    def __init__(self, num_blocks, encoder_outputs, phrase):
        super().__init__()
        self.num_blocks = num_blocks
        self.supports_masking = True
        
        self.batch_size = encoder_outputs[0].shape
        self.phrase = phrase.to(torch.int32)
        
        self.char_emb = nn.Embedding(N_UNIQUE_CHARACTERS, UNITS_DECODER)
        nn.init.constant_(self.char_emb, 0.0)
        self.positional_embedding = nn.Parameter(torch.zeros(N_TARGET_FRAMES, UNITS_DECODER), requires_grad = True)
        
        self.fc1 = nn.Linear(UNITS_DECODER, UNITS_DECODER*MLP_RATIO, bias = False)
        self.fc2 = nn.Linear(UNITS_DECODER*MLP_RATIO, UNITS_DECODER, bias = False)
        
        self.ln = nn.LayerNorm(UNITS_DECODER, eps = LAYER_NORM_EPS)
        self.mha = MultiHeadAttention(UNITS_DECODER, NUM_HEADS, MHA_DROPOUT_RATIO)
        self.mlp = nn.Sequential(
                self.fc1,
                nn.init.xavier_uniform_(self.fc1),
                nn.GELU(),
                nn.Dropout(MLP_DROPOUT_RATIO),
                self.fc2,
                nn.init.xavier_uniform_(self.fc2.weight)
        )

        self.attention_mask, = AttentionMask()



    def forward(self, encoder_outputs, phrase, x_inp):
        phrase = self.phrase.to(torch.int32)
        phrase = F.pad(phrase, (1,0), value = SOS_TOKEN)
        phrase = F.pad(phrase, (0, N_TARGET_FRAMES-MAX_PHRASE_LENGTH-1), value = PAD_TOKEN)
        
        x = self.positional_embedding + self.char_emb(phrase)
        x = self.ln(x + self.mha(x, x, x, attention_mask = self.get_casual_attention_mask()))

        attention_mask = self.attention_mask.get_attention_mask(x_inp)

        for i in range(self.num_blocks):
            x = self.ln(x + self.mha(x, encoder_outputs, encoder_outputs, attention_mask = attention_mask))
            x = self.ln(x + self.mlp(x))

        x = x[:, :MAX_PHRASE_LENGTH, :]

        return x