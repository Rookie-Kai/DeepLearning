import torch
import torch.nn as nn
import torch.optim as optim
import time
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# LeNet
# 展平
class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


# 重定型图像大小
class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)


net = torch.nn.Sequential(
    Reshape(),
    # (n_h-k_h+p_h+s_h)/s_h ,此处s_h为默认值1
    # 1*28*28 => (28-5+2*2+1)/1 * (28-5+2*2+1)/1 => 6*28*28
    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
    nn.Sigmoid(),
    # 6*28*28 => 6*14*14
    nn.AvgPool2d(kernel_size=2, stride=2),
    # 6*14*14 => (14-5+1)/1 * (14-5+1)/1 => 16*10*10 (padding = 0)
    nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
    nn.Sigmoid(),
    # 16*10*10 => 16*5*5
    nn.AvgPool2d(kernel_size=2, stride=2),
    Flatten(),
    nn.Linear(in_features=16*5*5, out_features=120),
    nn.Sigmoid(),
    nn.Linear(in_features=120, out_features=84),
    nn.Sigmoid(),
    nn.Linear(in_features=84, out_features=10)
)


# 查看形状
# X = torch.randn(size=(1, 1, 28, 28), dtype=torch.float32)
# for layer in net:
#     X = layer(X)
#     print(layer.__class__.__name__, 'output_shape: \t', X.shape)

# 获取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# print(len(train_iter)) = 235,共235个批次，每个批次有batch_size张图片


# 使用GPU加速计算
# def try_gpu():
#     if torch.cuda.is_available():
#         device=torch.device('cuda:1')
#     else:
#         device=torch.device('cpu')
#     return device


# 计算准确率
'''
(1). net.train()
  启用 BatchNormalization 和 Dropout，将BatchNormalization和Dropout置为True
(2). net.eval()
不启用 BatchNormalization 和 Dropout，将BatchNormalization和Dropout置为False
'''


def evaluate_accuracy(data_iter, net, device=torch.device('cpu')):
    acc_sum, n = torch.tensor([0], dtype=torch.float32, device=device),0
    for X,y in data_iter:
        # 将X，y代表的tensor变量，copy到device代表的设备上
        X, y = X.to(device), y.to(device)
        net.eval()
        with torch.no_grad():
            y = y.long()
            # 返回最大元素所对应的索引数，从0开始
            acc_sum += torch.sum((torch.argmax(net(X), dim=1) == y))
            n += y.shape[0]
    return acc_sum.item()/n


# 定义训练函数
def train_ch5(net, train_iter, test_iter, criterion, num_epochs, batch_size, device, lr=None):
    print('Train on', device)
    net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        train_l_sum = torch.tensor([0.0], dtype=torch.float32, device=device)
        train_acc_sum = torch.tensor([0.0], dtype=torch.float32, device=device)
        n, start = 0, time.time()
        for X, y in train_iter:
            net.train()

            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                y = y.long()
                train_l_sum += loss.float()

                # 计算训练集中预测正确的总数
                train_acc_sum += (torch.sum((torch.argmax(y_hat, dim=1) == y))).float()
                n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net, device)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
              'time %.1f sec'
              % (epoch + 1, train_l_sum/n, train_acc_sum/n, test_acc,
                 time.time() - start))


# 训练
lr, num_epochs = 0.9, 10


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)


net.apply(init_weights)
net = net.to(device)
criterion = nn.CrossEntropyLoss()
train_ch5(net, train_iter, test_iter, criterion, num_epochs, batch_size, device, lr)

# 测试
for testdata, testlabe in test_iter:
    testdata, testlabe = testdata.to(device), testlabe.to(device)
    break

print(testdata.shape, testlabe.shape)
net.eval()
y_pre = net(testdata)
print(torch.argmax(y_pre, dim=1)[:20])
print(testlabe[:20])
