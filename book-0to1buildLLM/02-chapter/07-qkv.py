import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],  # Your     (x^1)
     [0.55, 0.87, 0.66],  # journey  (x^2)
        [0.57, 0.85, 0.64],  # starts   (x^3)
        [0.22, 0.58, 0.33],  # with     (x^4)
        [0.77, 0.25, 0.10],  # one      (x^5)
        [0.05, 0.80, 0.55]]  # step     (x^6)
)


x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2

torch.manual_seed(123)
# 权重矩阵，是在训练过程中优化的神经网络参数
W_query = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print("query_2:", query_2)
print("key_2:", key_2)
print("value_2:", value_2)


# 所有的键向量和值向量
keys = inputs @ W_key
values = inputs @ W_value
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)
# w22注意力分数
keys_2 = keys[1]
attn_scores_22 = query_2 @ keys_2.T
print("attn_scores_2:", attn_scores_22)

# 所有的注意力分数
attention_scores_2 = query_2 @ keys.T
print("attention_scores_2:", attention_scores_2)

# 注意力分数转换为注意力权重
d_k = keys.shape[-1]
# 缩放点积
attention_weights_2 = torch.softmax(attention_scores_2 / d_k**0.5, dim=-1)
print("attention_weights_2:", attention_weights_2)


# 上下文向量
# 在这里，注意力权重作为加权因子，用于权衡每个值向量的重要性
context_vec_2 = attention_weights_2 @ values
print("context_vec_2:", context_vec_2)
