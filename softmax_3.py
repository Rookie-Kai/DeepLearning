import torch
from torch import nn
from torch.nn import init
import numpy as np
import d2lzh_pytorch as d2l
from collections import OrderedDict

# 获取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 使用线性模块定义和初始化模型
num_inputs = 784
num_outputs = 10


class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        y = self.linear(x.view(x.shape[0], -1))
        return y


net = LinearNet(num_inputs, num_outputs)


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


# 定义模型
net = nn.Sequential(
    OrderedDict([
        ('flatten', FlattenLayer()),
        ('linear', nn.Linear(num_inputs, num_outputs))
    ])
)

# 使用均值为0、标准差为0.01的正态分布随机初始化模型的权重参数
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

# softmax运算与交叉熵损失函数
loss = nn.CrossEntropyLoss()

# 优化算法
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# 训练模型
num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

# 预测
X, y = iter(test_iter).next()

true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(X[0:9], titles[0:9])
