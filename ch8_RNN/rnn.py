'''
    Implementation for RNN on the text in "Time Machine".
'''

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from d2l import torch as d2l

# device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# hyper-parameters
batch_size = 32
num_epochs = 30

seq_len = 35

num_hiddens = 256

