# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


# %%
# transform_train = transforms.Compose([transforms.ToTensor()])
transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.137,), (0.3081,))])
transform_valid = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform_train)#, target_transform=lambda x: torch.Tensor([x]).float())
trainloader = DataLoader(trainset, batch_size=4,
                         shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform_valid) #target_transform=lambda x: torch.Tensor([x]).float())
testloader = DataLoader(testset, batch_size=20,
                        shuffle=False, num_workers=0)


# %%
class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0)
        self.flatten = nn.Flatten(1)
        self.fc1 = nn.Linear(12 * 12 * 64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        # x = self.pool(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = F.dropout(x, 0.25)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

metric_func = lambda y_pred, y_true: roc_auc_score(y_true=y_true.data.numpy(), y_score=y_pred.data.numpy())
matric_name = "auc"
def train_step(model: nn.Module, features:torch.Tensor, labels: torch.Tensor):
    features = features.to(device=device)
    labels = labels.to(device)
    
    # model.train()
    optimizer.zero_grad()
    y_pred = model(features)
    loss = criterion(y_pred, labels)
    # metric = metric_func(y_pred, labels)
    metric = torch.Tensor(0)
    return loss, 0


# %%
from sklearn.metrics import roc_auc_score

cnn = Cnn()
cnn.to(device)
#criterion = nn.CrossEntropyLoss()
criterion = nn.NLLLoss()
optimizer = torch.optim.Adadelta(cnn.parameters(), lr=1.0)

# %%
import torchkeras
input_shape = (1, 28, 28)
torchkeras.summary(Cnn(), input_shape=(1, 28, 28))

# %%
(features, labels) = next(iter(trainloader))
# print((features[0], labels[0]))
print(trainloader.batch_sampler)
print(len(trainloader))

# %%
for epoch in range(20):
    loss_sum = []
    metric_sum = 0.0
    total_loss = 0.
    for i, data in enumerate(trainloader):
        features, labels = data
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        pred_y = cnn(features)
        loss = F.nll_loss(pred_y, labels)
        loss.backward()
        optimizer.step()
        loss_sum.append(loss)
        total_loss += loss.item()
        if i % 10 == 0:
            print('[%d, %5d] loss: %.5f' %
                    (epoch + 1, i + 1, loss.item()))


# %%



