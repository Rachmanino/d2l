import torch
from torch import nn
from d2l import torch as d2l

from loadFashionMNIST import train_iter, test_iter

'''定义模型'''
net = nn.Sequential(nn.Flatten(),   # 将28*28图像展平为784
                    nn.Linear(784, 10))

'''初始化权重'''
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights)

'''定义损失函数'''
loss = nn.CrossEntropyLoss()    # reduction是none会保留数组形态，才能计算loss

'''定义训练过程和学习率'''
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

'''训练'''
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)