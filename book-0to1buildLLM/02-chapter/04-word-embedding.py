import torch
input_ids = torch.tensor([2,3,5,1])
# 词汇表大小，仅为6个词
vocab_size = 6
output_dim =3
torch.manual_seed(123)
embedding = torch.nn.Embedding(vocab_size, output_dim)
print("Embedding weight:")
print(embedding.weight)

# 根据次元素id获取对应的词向量
# 嵌入层（Embedding Layer）是一种把离散数据（如整数ID）映射到连续向量空间的神经网络层。
# 嵌入层本质就是一个 查找表（Lookup Table）：
# 输入：词的ID（整数）
#       ↓
#     查找表
#       ↓
# 输出：词的向量表示（浮点数列表）
word_vector = embedding(torch.tensor([3]))
print("Word vectors:")
print(word_vector)