import torch 
from torch import nn
from torch.nn import functional as F
from sd.attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, vocab_size : int, n_embedding : int, n_token : int) :
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, n_embedding) 
        self.position_embedding = nn.Parameter(torch.zeros(n_token, n_embedding))

    def forward(self, tokens ):
        # (Batch_Size, Sequence_Length) -> (Batch_Size, Sequence_Length, dimen)
        x = self.token_embedding(tokens) + self.position_embedding
        return x

class CLIPLayer(nn.Module):
    def __init__(self, n_head : int, n_embedding : int):
        super().__init__()

        self.layernorm1 = nn.LayerNorm(n_embedding)
        self.attention = SelfAttention(n_head, n_embedding)
        self.layernorm2 = nn.LayerNorm(n_embedding)
        self.linear1 = nn.Linear(n_embedding, 4 * n_embedding)
        self.linear2 = nn.Linear(4 * n_embedding, n_embedding)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # (Batch_Size, Sequence_Length, dimen) 
        residue = x

        # SELF ATTENTION
        x = self.layernorm1(x)
        x = self.attention(x, casual_mask = True)

        x = x + residue

        # FEED FORWARD
        residue = x
        x = self.layernorm2(x)
        x = self.linear1(x)
        
        # QuickGELU, Practically works better
        x = x * torch.sigmoid(1.702 * x)
        x = self.linear2(x)
        x = x + residue

        return x

class CLIP(nn.Module):

    def __init__(self): 
        self.embedding = CLIPEmbedding(49408, 768, 77) 
        self.layers = nn.Module([
            CLIPLayer(12, 768) for i in range(12)
        ])
        # 768 features 
        self.layernorm = nn.LayerNorm(768) 
    def forward(self, tokens : torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        # (Batch_Size, Sequence_Length) -> (Batch_Size, Sequence_Length, dimen)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        # (Batch_Size, Sequence_Length, dimen) 
        output = self.layernorm(state)

        return output
    