import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

# %%
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True)
trainloader = DataLoader(trainset, batch_size=20,
                         shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True)
testloader = DataLoader(testset, batch_size=20,
                        shuffle=False, num_workers=0)

# %%
class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (5, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, (5, 5))
        self.fc1 = nn.Linear(7 * 7 * 64, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.pool(F.relu(self.conv2(x)))

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# %%
cnn = Cnn()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.0001)


# %%

def train_step(model: nn.Module, features, labels):
    model.train()
    optimizer.zero_grad()
    y_pred = model(features)
    loss = criterion(y_pred, labels)

    loss.backward()
    optimizer.step()
    return loss.item()


# %%

for i, data in enumerate(trainloader):
    pass
