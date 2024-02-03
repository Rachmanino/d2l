'''
    Implementation for traditional RNN on the text in "Time Machine".
'''

raise NotImplementedError

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from d2l import torch as d2l
from tqdm import tqdm

# device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# hyper-parameters
batch_size = 32
num_epochs = 30

seq_len = 35

hidden_size = 256

lr = 1e-2
wd = 0

# dataloader, dataset 
train_dataloader, vocab = d2l.load_data_time_machine(batch_size, num_steps=seq_len)

# RNN model
class RNN(nn.Module):
    '''RNN(traditional recurrent neural network)'''
    def __init__(self, vocab_size, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.vocab_size = vocab_size

        self.rnn = nn.RNN(vocab_size, hidden_size)
        self.fc = nn.RNN(hidden_size,)

model = RNN(len(vocab)).to(device)

# criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 
                             lr = lr,
                             wd = wd)

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