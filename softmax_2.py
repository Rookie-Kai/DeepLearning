import torch
import torchvision
import numpy as np
import d2lzh_pytorch as d2l

# 获取和读取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 初始化模型参数
num_imputs = 784
num_outputs = 10

W = torch.tensor(np.random.normal(0, 0.01, (num_imputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs,dtype=torch.float)

W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


# 实现softmax运算,同一列（dim=0）或同一行（dim=1）
# X = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print(X.sum(dim=0, keepdim=True))
# print(X.sum(dim=1, keepdim=True))
#
#
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制
#
#
# X = torch.rand((2, 5))
# X_prob = softmax(X)
# print(X_prob, X_prob.sum(dim=1))


# 定义模型
def net(X):
    return softmax(torch.mm(X.view((-1, num_imputs)), W) + b)


# 定义损失函数
def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))


# 计算分类准确率
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()


# 训练模型



