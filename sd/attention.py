import torch
from torch import nn
from torch.nn import functional as F
import math 

class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embedding, in_proj_bias = True, out_proj_bias = True):
        super().__init__()
        
        self.in_proj = nn.Linear(d_embedding, 3 * d_embedding, bias = in_proj_bias)
        self.out_proj = nn.Linear(d_embedding, d_embedding, bias = out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embedding // n_heads


    def forward(self, x: torch.Tensor, casual_mask = False) :
        # x : (Batch_Size, seq_len, dimen)

        inp_shape = x.shape
        batch_size, sequence_len, d_embedding = inp_shape

        intermim_shape = (batch_size, sequence_len, self.n_heads, self.d_head)

        # (Batch_Size, seq_len, dimen) -> (Batch_Size, seq_len, dimen * 3) -> 3 tensors of shape (Batch_Size, seq_len, dimen)
        q, k, v = self.in_proj(x).chunk(3, dim = -1)

        # (Batch_Size, seq_len, dimen) -> (Batch_Size, seq_len, H, dimen / H) -> (Batch_Size, H, seq_len, dimen / H)
        q = q.view(intermim_shape).transpose(1, 2) 
        k = k.view(intermim_shape).transpose(1, 2) 
        v = v.view(intermim_shape).transpose(1, 2) 

        # (Batch_Size, H, seq_len, seq_len)
        weight = q @ k.transpose(-2, -1)

        if casual_mask:
            # mask where the upper triangl(above the principal diagonal) is made up of 1
            mask = torch.ones_like(weight, dtype = torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight = weight / math.sqrt(self.d_head)
        weight = F.softmax(weight, dim = -1)

        # (Batch_Size, H, seq_len, seq_len) @ (Batch_Size, H, seq_len, dimen / H) -> (Batch_Size, H, seq_len, dimen / H) 
        out = weight @ v

        # (Batch_Size, H, seq_len, dimen / H) -> (Batch_Size, seq_len, H, dimen / H)
        out = out.transpose(1, 2) 

        out = out.reshape(inp_shape) 

        out = self.out_proj(out)

        # (Batch_Size, seq_len, dimen)
        return out

class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj   = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    
    def forward(self, x, y):
        # x (latent): # (Batch_Size, Seq_Len_Q, Dim_Q)
        # y (context): # (Batch_Size, Seq_Len_KV, Dim_KV) = (Batch_Size, 77, 768)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        # Divide each embedding of Q into multiple heads such that d_heads * n_heads = Dim_Q
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)
        
        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        q = self.q_proj(x)
        # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        k = self.k_proj(y)
        # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        v = self.v_proj(y)

        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        q = q.view(interim_shape).transpose(1, 2) 
        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        k = k.view(interim_shape).transpose(1, 2) 
        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        v = v.view(interim_shape).transpose(1, 2) 
        
        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) @ (Batch_Size, H, Dim_Q / H, Seq_Len_KV) -> (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight = q @ k.transpose(-1, -2)
        
        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight /= math.sqrt(self.d_head)
        
        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight = F.softmax(weight, dim=-1)
        
        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV) @ (Batch_Size, H, Seq_Len_KV, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        output = weight @ v
        
        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H)
        output = output.transpose(1, 2).contiguous()
        
        # (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = output.view(input_shape)
        
        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = self.out_proj(output)

        # (Batch_Size, Seq_Len_Q, Dim_Q)
        return output