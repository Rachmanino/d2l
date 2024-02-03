'''
    Implementation for CNN on FashionMNIST, acc = 90%
'''

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from tqdm.rich import tqdm

# hyperparams
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch_size = 256
num_epochs = 30

input_size = 28 * 28
num_classes = 10

lr = 1e-2
wd = 0

# datasets and dataloaders
train_dataset = FashionMNIST(root = '../data', 
                      train = True,
                      transform = transforms.ToTensor(),
                      download = True)

test_dataset = FashionMNIST(root = '../data', 
                      train = False,
                      transform = transforms.ToTensor())

train_dataloader = DataLoader(dataset = train_dataset,
                              batch_size = batch_size,
                              shuffle = True)

test_dataloader = DataLoader(dataset = test_dataset,
                              batch_size = batch_size,
                              shuffle = False)

# net, criterion and optimizer
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.BatchNorm2d(6), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.BatchNorm1d(120), nn.ReLU(), 
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.ReLU(), 
    nn.Linear(84, num_classes))
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

# train
print(f'Traing on {device}')
net.train()
for epoch in tqdm(range(num_epochs)):
    train_loss = 0
    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device) # Using GPU

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, loss = {train_loss:.3f}')

# test
net.eval()
with torch.no_grad():
    correct = total = 0
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device) # Using GPU

        output  = net(inputs)
        predict = output.argmax(axis = 1)   # axis=0是各inputs
        
        correct += (predict == labels).sum().item()
        total += predict.size(0)
    
    print(f'Test acc on FashionMNIST: {correct}/{total} = {100*correct/total}%')

# save model
torch.save(net, 'cnn.pt')
print('Model has been saved to cnn.pt')

