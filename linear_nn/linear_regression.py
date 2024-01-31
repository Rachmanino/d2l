import numpy
import torch
from torch import nn
from torch.utils import data


######################################################## 1.生成数据#############################################################
'''
为了简单起见，我们将根据带有噪声的线性模型构造一个人造数据集。 
我们的任务是使用这个有限样本的数据集来恢复这个模型的参数。
我们将使用低维数据，这样可以很容易地将其可视化。 
在下面的代码中，我们生成一个包含1000个样本的数据集，每个样本包含从标准正态分布中采样的2个特征。 
'''

true_w = torch.tensor([2, -3.4])
true_b = 4.2


def generate_data(w, b, num_examples):
    """生成y=Xw+b+噪声（为了简化问题将标准差设为0.01）"""
    X = torch.normal(0, 1, (num_examples, len(w)))  # m*n
    y = torch.matmul(X, w) + b
    # X:m*n, w:n, note that y here is m-dim vector rather than m*1-matrix
    y += torch.normal(0, 0.01, y.shape)
    # print(X.shape, w.shape, y.shape, y.reshape([-1, 1]).shape)
    return X, y.reshape([-1, 1])    # 把y做成m*1矩阵


features, labels = generate_data(true_w, true_b, 1000)


######################################################## 2.读取数据集#############################################################
def load_array(data_arrays, batch_size, _shuffle=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)  # *data_arrays表示遍历列表中每一个元素
    return data.DataLoader(dataset, batch_size, shuffle=_shuffle)


batch_size = 10
data_iter = load_array((features, labels), batch_size)


######################################################## 3.定义模型#############################################################
net = nn.Sequential(nn.Linear(2, 1))    # 定义模型结构：2->1

net[0].weight.data.fill_(0)    # 初始化weight和bias, net[0]表示网络第0层
net[0].bias.data.fill_(0)

loss = nn.MSELoss()  # 采用均方误差

# 定义训练方法：随机梯度下降，训练全体模型参数，定义学习率
trainer = torch.optim.SGD(net.parameters(), lr=0.1)


######################################################## 4.训练模型#############################################################
num_epochs = 10
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(
        f'epoch {epoch + 1}, loss {l:f}, w = {net[0].weight.tolist()}, b = {net[0].bias.item()}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
