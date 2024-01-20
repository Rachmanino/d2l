import torch
from torch import nn
from torch.utils import data


########################################################1.生成数据#############################################################
'''
使用与MNIST数据集类似但更复杂的Fashion-MNIST数据集
'''

true_w = torch.tensor([2, -3.4])
true_b = 4.2

def generate_data(w, b, num_examples):
    """生成y=Xw+b+噪声（为了简化问题将标准差设为0.01）"""
    X = torch.normal(0, 1, (num_examples, len(w))) # m*n
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape) # X:m*n, w:n, note that y here is m-dim vector rather than m*1-matrix
    # print(X.shape, w.shape, y.shape, y.reshape([-1, 1]).shape)
    return X, y.reshape([-1, 1])    # 把y做成m*1矩阵

features, labels = generate_data(true_w, true_b, 1000)


########################################################2.读取数据集############################################################# 
def load_array(data_arrays, batch_size, _shuffle=True): 
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays) # *data_arrays表示遍历列表中每一个元素
    return data.DataLoader(dataset, batch_size, shuffle=_shuffle)

batch_size = 10
data_iter = load_array((features, labels), batch_size)


########################################################3.定义模型############################################################# 
net = nn.Sequential(nn.Linear(2, 1))    # 定义模型结构：2->1

net[0].weight.data.fill_(0)    # 初始化weight和bias, net[0]表示网络第0层
net[0].bias.data.fill_(0)

loss = nn.MSELoss() # 采用均方误差

trainer = torch.optim.SGD(net.parameters(), lr = 0.1)  # 定义训练方法：随机梯度下降，训练全体模型参数，定义学习率


########################################################4.训练模型############################################################# 
num_epochs = 10
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)