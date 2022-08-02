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


def make_sequential(num):
    layers = []
    for i in range(num):
        layers.append(nn.Linear(1, 1))
        layers.append(nn.ReLU(True))

    return nn.Sequential(*layers)

""" # double loop method
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer = nn.Linear(32*32*3, 512)
        for i in range(512): #512 = output size of before layer
            globals()['self.ws{}'.format(i)] = My_2D_Parameter(32).cuda() #32 = optional num
            globals()['self.bs{}'.format(i)] = My_2D_Parameter(32).cuda()
        self.layer_last = nn.Linear(32*512, 10)

    def forward(self, x, batch_len):
        x = x.view(x.size(0), -1)
        x = self.layer(x)
        #print(x.shape, globals()['self.ws0']().shape, globals()['self.bs0']().shape)
        for i in range(batch_len): #128 = batch size
            for j in range(512): #512 = output size of before layer
                if j == 0:
                    x0 = torch.mul(x[i][j], globals()['self.ws{}'.format(j)]()) + globals()['self.bs{}'.format(j)]()
                elif j == 1:
                    x1 = torch.mul(x[i][j], globals()['self.ws{}'.format(j)]()) + globals()['self.bs{}'.format(j)]()
                    x_tmp_tmp = torch.cat((x0, x1), dim=0)
                elif j > 1:
                    x1 = torch.mul(x[i][j], globals()['self.ws{}'.format(j)]()) + globals()['self.bs{}'.format(j)]()
                    x_tmp_tmp = torch.cat((x_tmp_tmp, x1), dim=0)
            if i == 0:
                x_tmp = x_tmp_tmp
            else:
                x_tmp = torch.cat((x_tmp, x_tmp_tmp), dim=0)
        x = torch.reshape(x_tmp, (128, 32*512))
        x = nn.ReLU()(x)
        x = self.layer_last(x)
        return x
"""

class MLP(nn.Module):
    def __init__(self, branch_num):
        super(MLP, self).__init__()
        self.branch_num = branch_num
        self.layer = nn.Linear(32*32*3, 512)
        self.ws = My_3D_Parameter(self.branch_num, 512)().cuda()
        self.bs = My_3D_Parameter(self.branch_num, 512)().cuda()
        self.layer_last = nn.Linear(self.branch_num*512, 10)

    def forward(self, x, batch_len):
        x = x.view(x.size(0), -1)
        x = self.layer(x)
        for i in range(self.branch_num):
            tmp = torch.mul(x, self.ws[i]) + self.bs[i]
            if i == 0:
                x0 = tmp.unsqueeze(-1)
            elif i > 0:
                x1 = tmp
                x0 = torch.cat((x0, x1.unsqueeze(-1)), dim=2)
        x = x0.permute(0, 2, 1)
        x = torch.reshape(x, (batch_len, self.branch_num*512))
        x = nn.ReLU()(x)
        x = self.layer_last(x)
        return x

model = MLP(branch_num=1)
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
        hypo = model(X, len(X))
        #hypo = model(X)
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
        output = model(data, len(data))
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
    output = model(data, len(data))
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