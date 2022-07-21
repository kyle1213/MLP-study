import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
from torchsummary import summary as summary_
from tqdm import tqdm

random_seed = 1234
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

train_transform = transforms.Compose([transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])

train = torchvision.datasets.CIFAR10(root='D:\datasets\CIFAR-10',
                                      train=True, transform=train_transform,
                                      download=True)
test = torchvision.datasets.CIFAR10(root='D:\datasets\CIFAR-10',
                                     train=False, transform=test_transform,
                                     download=True)

train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=128, shuffle=False)

cuda = torch.device('cuda')


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer = nn.Sequential(nn.Linear(32*32*3, 597),
                                     nn.ReLU(True),
                                     nn.Linear(597, 10))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer(x)
        return x


model = MLP()
model = model.cuda()

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4, betas=(0.9, 0.999))
cost = 0

iterations = []
train_losses = []
test_losses = []
train_acc = []
test_acc = []

summary_(model,(3,32,32),batch_size=7)

for epoch in range(50):
    model.train()
    correct = 0
    for X, Y in tqdm(train_loader):
        X = X.to(cuda)
        Y = Y.to(cuda)
        optimizer.zero_grad()
        hypo = model(X)
        cost = loss(hypo, Y)
        cost.backward()
        optimizer.step()
        prediction = hypo.data.max(1)[1]
        correct += prediction.eq(Y.data).sum()

    model.eval()
    correct2 = 0
    for data, target in test_loader:
        data = data.to(cuda)
        target = target.to(cuda)
        output = model(data)
        cost2 = loss(output, target)
        prediction = output.data.max(1)[1]
        correct2 += prediction.eq(target.data).sum()

    print("Epoch : {:>4} / cost : {:>.9}".format(epoch + 1, cost))
    iterations.append(epoch)
    train_losses.append(cost.tolist())
    test_losses.append(cost2.tolist())
    train_acc.append((100*correct/len(train_loader.dataset)).tolist())
    test_acc.append((100*correct2/len(test_loader.dataset)).tolist())
print('Train set: Accuracy: {:.2f}%'.format(100. * correct / len(train_loader.dataset)))

# del train_loader
# torch.cuda.empty_cache()

model.eval()
correct = 0
for data, target in test_loader:
    data = data.to(cuda)
    target = target.to(cuda)
    output = model(data)
    prediction = output.data.max(1)[1]
    correct += prediction.eq(target.data).sum()

print('Test set: Accuracy: {:.2f}%'.format(100. * correct / len(test_loader.dataset)))

plt.subplot(121)
plt.plot(range(1, len(iterations)+1), train_losses, 'b--')
plt.plot(range(1, len(iterations)+1), test_losses, 'r--')
plt.subplot(122)
plt.plot(range(1, len(iterations)+1), train_acc, 'b-')
plt.plot(range(1, len(iterations)+1), test_acc, 'r-')
plt.title('loss and accuracy')
plt.show()