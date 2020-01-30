import torch
import numpy as np
import random
import torch.utils.data as Data
from torch import nn


# 生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)


# 根据data模块读取数据
batch_size = 10
dataset = Data.TensorDataset(features, labels)
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)


# 打印数据
# for X, y in data_iter:
#     print(X, '\n', y)
#     break

# 定义模型
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    # forward定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y


net = LinearNet(num_inputs)
# print(net)
#
# 通过nn.Sequential搭建网络
# 写法一
# net = nn.Sequential(
#     nn.Linear(num_inputs, 1)
# )
#
# 写法二
# net = nn.Sequential()
# net.add_module('linear',nn.Linear(num_inputs, 1))
#
# 写法三
# from collections import OrderedDict
# net = nn.Sequential(OrderedDict([
#     ('linear', nn.Linear(num_inputs, 1))
# ]))
#
# print(net)
# print(net[0])
#
# 通过net.parameters() 查看模型的全部可学习参数
# for param in net.parameters():
#     print(param)
#
# 初始化模型参数
from torch.nn import init

init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

# init.normal_(net[0].weight, mean=0, std=0.01)
# init.constant_(net[0].bias, val=0)

# 定义损失函数
loss = nn.MSELoss()

# 定义优化算法
import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.03)

# 调整学习率
# for param_group in optimizer.param_groups:
#     param_group['lr'] *= 0.1
# print(optimizer)

# 训练模型
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))

dense = net.linear
# dense = net[0]
print(true_w, dense.weight)
print(true_b, dense.bias)
