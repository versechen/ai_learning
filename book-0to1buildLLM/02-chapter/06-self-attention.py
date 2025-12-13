import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],  # Your     (x^1)
     [0.55, 0.87, 0.66],  # journey  (x^2)
        [0.57, 0.85, 0.64],  # starts   (x^3)
        [0.22, 0.58, 0.33],  # with     (x^4)
        [0.77, 0.25, 0.10],  # one      (x^5)
        [0.05, 0.80, 0.55]]  # step     (x^6)
)

query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
print(attn_scores_2.shape)
# 计算每个输入与查询的点积
for i in range(inputs.shape[0]):
    attn_scores_2[i] = torch.dot(query, inputs[i])
print("attn_scores_2:", attn_scores_2)
# 归一化
attn_scores_2_temp = attn_scores_2 / attn_scores_2.sum()
print("attn_scores_2_temp:", attn_scores_2_temp)
print("Sum:", attn_scores_2_temp.sum())


def softmax_native(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)


# 使用自定义的 softmax 函数, 可能会存在
attn_scores_2_native = softmax_native(attn_scores_2)
print("attn_scores_2_native:", attn_scores_2_native)
print("Sum:", attn_scores_2_native.sum())

# 计算 softmax
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("attn_weights_2_native:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

# 计算单个上下文向量
query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
# 上下文向量，可以理解为一种包含了序列中所有元素信息的嵌入向量
print("context_vec_2:", context_vec_2)

# 方法1： 计算全部上下文向量
attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
# 自注意力分数
print("attn_scores:", attn_scores)

# 方法2： 计算全部上下文向量
atten_scores = inputs @ inputs.T
print("atten_scores:", atten_scores)

# 方法3： 计算全部上下文向量
atten_scores = inputs @ inputs.T
print("atten_scores:", atten_scores)

# 归一化
atten_weights = torch.softmax(atten_scores, dim=-1)
print("atten_weights:", atten_weights)

# 计算全部上下文向量
context_vecs = atten_weights @ inputs
print("context_vecs:", context_vecs)
