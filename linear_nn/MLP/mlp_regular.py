'''
    Implementation for a Multilayer Perceptron with dropout and L2 regularzation on FashionMNIST.
'''

import torch
from torch import nn
from d2l import torch as d2l

from loadFashionMNIST import train_iter, test_iter

'''定义模型'''
net = nn.Sequential(nn.Flatten(),   # 将28*28图像展平为784
                    nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.2),    #!靠近输入层dropout较低
                    nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(256, 10))

'''初始化权重'''
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std = 0.01)
net.apply(init_weights)

'''定义损失函数'''
loss_fn = nn.CrossEntropyLoss()

'''定义训练过程和学习率'''
weight_p, bias_p = [],[]
for name, p in net.named_parameters():
    if 'bias' in name:
        bias_p.append(p)
    else:
        weight_p.append(p)
trainer = torch.optim.SGD([{'params': weight_p, 'weight_decay': 1e-3},   #!注意仅有weight需要正则化
                           {'params': bias_p}],
                           lr = 0.1)

'''训练'''
num_epochs = 100
for epoch in range(num_epochs):
    net.train() #!有dropout需要在训练时打开dropout
    train_loss = 0
    for image, label in train_iter:
        loss = loss_fn(net(image), label)
        train_loss += loss

        trainer.zero_grad()
        loss.backward()
        trainer.step()

    with torch.no_grad():
        net.eval() #!有dropout需要在测试时关闭dropout
        test_correct = test_num = 0
        for image, label in test_iter:
            predict = net(image).argmax(axis=1)
            test_correct += (predict == label).sum()
            test_num += image.shape[0]
    print(
        f'Epoch {epoch+1}, train_loss = {train_loss:.2f}, test_acc = {test_correct / test_num:.2f}')

# d2l.train_ch3(net, train_iter, test_iter, loss_fn, num_epochs, trainer)
