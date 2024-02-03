import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms

from d2l import torch as d2l    # 采用torch版本d2l库
d2l.use_svg_display()
########################################################1.下载FashionMNIST#############################################################
'''
通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式
并除以255使得所有像素的数值均在0~1之间
'''
mnist_train = torchvision.datasets.FashionMNIST(    # 训练集包括60000张图像
    root = "../../data", train = True, transform = transforms.ToTensor(), download = True
)
mnist_test = torchvision.datasets.FashionMNIST(   # 测试集包括10000张图像
    root = "../../data", train = False, transform = transforms.ToTensor(), download = True
)
'''每个输入图像的高度和宽度均为28像素 
数据集由灰度图像组成,其通道数为1
E.g. mnist_train[0~59999][0]是一个[1, 28, 28]Tensor, mnist_train[0~59999][0]为label
'''

def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  # @save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

'''
# 在Jupyter中运行到该单元格(结果见samples.svg)
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
'''

#########################################################2.读取数据集############################################################# 
batch_size = 256
num_workers = 4   # 用4个进程读取数据
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=num_workers)
test_iter = data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=num_workers)
