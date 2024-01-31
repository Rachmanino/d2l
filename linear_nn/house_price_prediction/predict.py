import pandas as pd
import torch
from torch import nn
from torch.utils import data
from sklearn.model_selection import KFold

def log_rmse(net, features, labels):    
    '''对数均方根误差'''
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(criterion(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()

train_data = pd.read_csv('./train.csv')  # (1460, 80 features + 1 label)
test_data = pd.read_csv('./test.csv')  # (1460, 80 features)

# print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

'''对数据进行标准化和归一化'''
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
# 若无法获得测试数据，则可根据训练数据计算均值和标准差
numberic_features_index = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numberic_features_index] = all_features[numberic_features_index].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numberic_features_index] = all_features[numberic_features_index].fillna(0)

# “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
all_features = pd.get_dummies(all_features, dummy_na=True)

train_size = train_data.shape[0]
train_features = torch.tensor(
    all_features[:train_size].values, dtype=torch.float32)
test_features = torch.tensor(
    all_features[train_size:].values, dtype=torch.float32)
train_labels = torch.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)
# 这里不要忘记reshape(-1, 1)来将train_labels转化为矩阵

net = nn.Sequential(nn.Linear(train_features.shape[1], 1000), nn.ReLU(), nn.Dropout(0.4),
                    nn.Linear(1000, 1)) #TODO: model 

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(net.parameters(), lr=0.2, weight_decay=1e-2)  #TODO: regularzation

kf = KFold()

# loop over the dataset multiple times
net.train()
for epoch in range(500):
    # train
    running_loss = 0
    for train_index, test_index in kf.split(train_features, train_labels):
        train_iter = data.DataLoader(data.TensorDataset(train_features[train_index], train_labels[train_index]),
                                     batch_size=256, shuffle=True)
        for inputs, labels in train_iter:
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
    print(f'Loss: {running_loss}', end=', ')

    # test
    net.eval()
    error = log_rmse(net, train_features, train_labels)
    print(f'error = {error}')

print('Finished Training')


