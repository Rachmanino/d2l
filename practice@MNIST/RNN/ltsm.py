'''
    Implementation for LTSM(Long-term short memory) on MNIST dataset, acc = 98.7%
'''
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm import tqdm

# device configuration 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# hyper-parameters
batch_size = 256
num_epochs = 20

input_size = 28     # size per calculation
seq_len = 28        # cycles of calculations 
hidden_size = 128   # size of hidden layer
num_classes = 10    # size of output
num_layers = 1      # layers of LTSM

lr = 1e-2
wd = 0

# datasets and dataloaders
train_dataset = MNIST(root='../data',
                      train=True,
                      transform=transforms.ToTensor(),
                      download=True)

test_dataset = MNIST(root='../data',
                     train=False,
                     transform=transforms.ToTensor())

train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

# LTSM
class LTSM(nn.Module):
    '''LTSM (Long-term short memory), refined RNN'''
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 num_layers,
                 num_classes
                 ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers    # layers of LTSM

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True) # keep batch on axis=0
        self.fc = nn.Linear(hidden_size, num_classes)   # fully-connected

    def forward(self, x):
        ltsm_out, _ = self.lstm(x)
        out = self.fc(ltsm_out[:, -1, :])  # [batch, seq, hidden], get last item of seq
        return out

model = LTSM(input_size, hidden_size, num_layers, num_classes).to(device)

# criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr = lr,
                             weight_decay = wd)

# train
print(f'Traing on {device}:')
model.train()
for epoch in tqdm(range(num_epochs)):
    train_loss = 0
    for inputs, labels in train_dataloader:
        inputs = inputs.reshape(-1, seq_len, input_size).to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, loss = {train_loss:.3f}')
    

# test
model.eval()
with torch.no_grad():
    correct = total = 0
    for inputs, labels in test_dataloader:
        inputs = inputs.reshape(-1, seq_len, input_size).to(device)
        labels = labels.to(device)

        output = model(inputs)
        predict = output.argmax(axis=1)   # axis=0是各inputs

        correct += (predict == labels).sum().item()
        total += predict.size(0)

    print(f'Test acc on MNIST: {correct}/{total} = {100*correct/total}%')






