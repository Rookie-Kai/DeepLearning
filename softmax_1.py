import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
import d2lzh_pytorch as d2l


mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIst', train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIst', train=False, download=True, transform=transforms.ToTensor())
# print(type(mnist_train))
# print(len(mnist_train), len(mnist_test))

# 通过下标访问任一样本， feature尺寸=通道数*宽*高，因为是灰度图像，通道数=1
# feature, label = mnist_train[1]
# print(feature.shape, label)


# 数值标签化为文本标签
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankel boot']
    return [text_labels[int(i)] for i in labels]


# 在一行画出多张图像和对应标签
def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    # _ 表示忽略不使用的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
show_fashion_mnist(X, get_fashion_mnist_labels(y))
