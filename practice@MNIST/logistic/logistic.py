'''
    Implementation for Logistic Regression on MNIST, acc = 92.35%
'''

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm.rich import tqdm

# hyperparams
batch_size = 256
num_epochs = 10

input_size = 28 * 28
num_classes = 10

lr = 1e-2
wd = 0

# datasets and dataloaders
train_dataset = MNIST(root = '../data', 
                      train = True,
                      transform = transforms.ToTensor(),
                      download = True)

test_dataset = MNIST(root = '../data', 
                      train = False,
                      transform = transforms.ToTensor())

train_dataloader = DataLoader(dataset = train_dataset,
                              batch_size = batch_size,
                              shuffle = True)

test_dataloader = DataLoader(dataset = test_dataset,
                              batch_size = batch_size,
                              shuffle = False)

# net, loss fn and optimizer
net = nn.Sequential(nn.Flatten(),  # [28, 28] to [784]
                    nn.Linear(input_size, num_classes))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

# train
net.train()
for epoch in tqdm(range(num_epochs)):
    train_loss = 0
    for inputs, labels in train_dataloader:
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_classes}, loss = {train_loss:.3f}')

# test
net.eval()
with torch.no_grad():
    correct = total = 0
    for inputs, labels in test_dataloader:
        output  = net(inputs)
        predict = output.argmax(axis = 1)   # axis=0是各inputs
        
        correct += (predict == labels).sum().item()
        total += predict.size(0)
    
    print(f'Test acc on MNIST: {correct}/{total} = {100*correct/total}%')

# save model
torch.save(net, 'net.pt')
print('Model has been saved to ./net.pt')
