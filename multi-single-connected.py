import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from torchsummary import summary as summary_
from tqdm import tqdm
from torch import Tensor


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


class My_2D_Parameter(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, size_in):
        super(My_2D_Parameter, self).__init__()
        self.size_in = size_in
        bias = torch.Tensor(size_in)
        self.weights = nn.Parameter(bias)

        # initialize weights and biases
        torch.nn.init.zeros_(self.weights)

    def forward(self) -> Tensor:
        return self.weights


class My_3D_Parameter(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, size_in, size_out):
        super(My_3D_Parameter, self).__init__()
        self.size_in, self.size_out = size_in, size_out
        weights = torch.Tensor(size_in, size_out)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

    def forward(self) -> Tensor:
        return self.weights


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer = nn.Linear(32*32*3, 512)
        self.ws0 = My_2D_Parameter(512)
        self.bs0 = My_2D_Parameter(512)
        self.ws1 = My_2D_Parameter(512)
        self.bs1 = My_2D_Parameter(512)
        self.ws2 = My_2D_Parameter(512)
        self.bs2 = My_2D_Parameter(512)
        self.ws3 = My_2D_Parameter(512)
        self.bs3 = My_2D_Parameter(512)
        self.ws4 = My_2D_Parameter(512)
        self.bs4 = My_2D_Parameter(512)
        self.ws5 = My_2D_Parameter(512)
        self.bs5 = My_2D_Parameter(512)
        self.ws6 = My_2D_Parameter(512)
        self.bs6 = My_2D_Parameter(512)
        self.ws7 = My_2D_Parameter(512)
        self.bs7 = My_2D_Parameter(512)
        self.ws8 = My_2D_Parameter(512)
        self.bs8 = My_2D_Parameter(512)
        self.ws9 = My_2D_Parameter(512)
        self.bs9 = My_2D_Parameter(512)
        self.layer_last = nn.Linear(5120, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer(x)
        x0 = torch.mul(x, self.ws0()) + self.bs0()
        x1 = torch.mul(x, self.ws1()) + self.bs1()
        x2 = torch.mul(x, self.ws2()) + self.bs2()
        x3 = torch.mul(x, self.ws3()) + self.bs3()
        x4 = torch.mul(x, self.ws4()) + self.bs4()
        x5 = torch.mul(x, self.ws5()) + self.bs5()
        x6 = torch.mul(x, self.ws6()) + self.bs6()
        x7 = torch.mul(x, self.ws7()) + self.bs7()
        x8 = torch.mul(x, self.ws8()) + self.bs8()
        x9 = torch.mul(x, self.ws9()) + self.bs9()
        x = torch.cat((x0, x1), dim=1)
        x = torch.cat((x, x2), dim=1)
        x = torch.cat((x, x3), dim=1)
        x = torch.cat((x, x4), dim=1)
        x = torch.cat((x, x5), dim=1)
        x = torch.cat((x, x6), dim=1)
        x = torch.cat((x, x7), dim=1)
        x = torch.cat((x, x8), dim=1)
        x = torch.cat((x, x9), dim=1)

        x = nn.ReLU()(x)
        x = self.layer_last(x)
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

#summary_(model,(3,32,32),batch_size=7)

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