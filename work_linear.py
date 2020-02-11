import torch

# y_hat = torch.tensor([[2.33], [1.07], [1.23]])
y_hat = torch.tensor([2.33, 1.07, 1.23])
y = torch.tensor([3.14, 0.98, 1.32])


def Loss(y_hat, y):
    return (y_hat - y) ** 2 / 2


mean = Loss(y_hat, y).sum() / 3
t1 = y_hat
t4 = y_hat.view(-1)
t3 = y.view(-1)
t2 = y.view(1, -1)

print('%.3f' % mean)
# print(t3)
# print(y)
# print(t2)

# view(-1)的作用
print(y_hat)
print(t4)
print(y)
print(t3)

