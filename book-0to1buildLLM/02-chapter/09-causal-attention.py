import torch
import torch.nn as nn

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],  # Your     (x^1)
     [0.55, 0.87, 0.66],  # journey  (x^2)
        [0.57, 0.85, 0.64],  # starts   (x^3)
        [0.22, 0.58, 0.33],  # with     (x^4)
        [0.77, 0.25, 0.10],  # one      (x^5)
        [0.05, 0.80, 0.55]]  # step     (x^6)
)


class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec


sa_v2 = SelfAttention_v2(3, 2)
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
print("attn_weights:", attn_weights)

context_length = attn_scores.shape[0]
mask_siple = torch.tril(torch.ones(context_length, context_length))
print("mask_siple:", mask_siple)

masked_simple = attn_weights * mask_siple
print("masked_simple:", masked_simple)

row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print("masked_simple_norm:", masked_simple_norm)

# 更高效的掩码
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print("masked:", masked)

attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
print("attn_weights:", attn_weights)

torch.manual_seed(123)
dropout = nn.Dropout(0.5)
example = torch.ones(6, 6)
dropout_example = dropout(example)
print("dropout_example:", dropout_example)

torch.manual_seed(123)
attn_weights_dropped = dropout(attn_weights)
print("attn_weights_dropped:", attn_weights_dropped)
