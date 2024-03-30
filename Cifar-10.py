import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch import Tensor
import numpy as np
from torch.optim import lr_scheduler

import Optimizer as Opt
import preresnet_cifar as Pre


def calculate_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


BATCH_SIZE = 512
EPOCH = 80
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多

transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

train_dataset = torchvision.datasets.CIFAR10(root='/input/cifar10-python',
                                             train=True,
                                             transform=transform,
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root='/input/cifar10-python',
                                            train=False,
                                            transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False)


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# ResNet
class Net(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(Net, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out





net_Momentum = Pre.preresnet56_cifar10(num_classes=10).to(device)
net_RMSprop = Pre.preresnet56_cifar10(num_classes=10).to(device)
net_Adam = Pre.preresnet56_cifar10(num_classes=10).to(device)
net_SGDAMD = Pre.preresnet56_cifar10(num_classes=10).to(device)
net_FRGD = Pre.preresnet56_cifar10(num_classes=10).to(device)
net_SGDMD = Pre.preresnet56_cifar10(num_classes=10).to(device)
# net_Momentum = Net(ResidualBlock, [2, 2, 2]).to(device)
# net_RMSprop = Net(ResidualBlock, [2, 2, 2]).to(device)
# net_Adam =Net(ResidualBlock, [2, 2, 2]).to(device)
# net_SGDAMD =Net(ResidualBlock, [2, 2, 2]).to(device)
# net_FRGD = Net(ResidualBlock, [2, 2, 2]).to(device)
# net_SGDMD =Net(ResidualBlock, [2, 2, 2]).to(device)
# net_PID = Net(ResidualBlock, [2, 2, 2]).to(device)

# net_SGDAMD2 =Net(ResidualBlock, [2, 2, 2]).to(device)
# net_SGDAMD3 =Net(ResidualBlock, [2, 2, 2]).to(device)
# net_SGDAMD4 =Net(ResidualBlock, [2, 2, 2]).to(device)
# net_SGDAMD5 =Net(ResidualBlock, [2, 2, 2]).to(device)
# net_SGDAMD6 =Net(ResidualBlock, [2, 2, 2]).to(device)
nets = [net_FRGD, net_Adam, net_RMSprop, net_SGDAMD, net_SGDMD]

# optimizer_SGD = torch.optim.SGD(net_SGD.parameters(), lr=0.01)

# optimizer_Momentum = Opt.SGDM(net_Momentum.parameters(), lr=0.1, momentum=0.9,weight_decay=5e-4)
optimizer_RAdam = Opt.RAdam(net_RMSprop.parameters(), lr=0.001, betas=(0.9, 0.99))
optimizer_Adam = Opt.Adam(net_Adam.parameters(), lr=0.001, betas=(0.9, 0.99))
optimizer_SGDAMD = Opt.SGDAMD(net_SGDAMD.parameters(), lr=0.5, weight_decay=5e-4)
optimizer_FRGD = Opt.FRGD(net_FRGD.parameters(), lr=0.1, weight_decay=5e-4)
optimizer_SGDMD = Opt.SGDMD(net_SGDMD.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
# optimizer_SGDAMD = SGDAMD(net_SGDAMD.parameters(), lr=0.1, weight_decay=5e-4)
# optimizer_SGDAMD2 = SGDAMD(net_SGDAMD2.parameters(), lr=0.3, weight_decay=5e-4)
# optimizer_SGDAMD3 = SGDAMD(net_SGDAMD3.parameters(), lr=0.5, weight_decay=5e-4)
# optimizer_SGDAMD4 = SGDAMD(net_SGDAMD4.parameters(), lr=1, weight_decay=5e-4)
# optimizer_SGDAMD5 = SGDAMD(net_SGDAMD5.parameters(), lr=0.01, weight_decay=5e-4)
# optimizer_SGDAMD6 = SGDAMD(net_SGDAMD6.parameters(), lr=0.01, weight_decay=5e-4)
# scheduler = lr_scheduler.MultiStepLR(optimizer_Momentum , milestones=[10,30,60], gamma=0.5)
scheduler2 = lr_scheduler.MultiStepLR(optimizer_RAdam, milestones=[70, 80], gamma=0.5)
scheduler3 = lr_scheduler.MultiStepLR(optimizer_Adam, milestones=[70, 80], gamma=0.5)
scheduler4 = lr_scheduler.MultiStepLR(optimizer_SGDAMD, milestones=[10, 60, 80], gamma=0.5)
scheduler5 = lr_scheduler.MultiStepLR(optimizer_FRGD, milestones=[10, 60, 80], gamma=0.5)
scheduler6 = lr_scheduler.MultiStepLR(optimizer_SGDMD, milestones=[60, 80], gamma=0.5)
# scheduler7 = lr_scheduler.MultiStepLR(optimizer_Momentum3, milestones=[40,80], gamma=0.1)
# scheduler6 = lr_scheduler.MultiStepLR(optimizer_CustomOptimizer3, milestones=[20,40], gamma=0.5)
# optimizers = [optimizer_Momentum,  optimizer_Adam, optimizer_CustomOptimizer,optimizer_CustomOptimizer2]

# scheduler = lr_scheduler.MultiStepLR(optimizer_SGDAMD , milestones=[10,60,80], gamma=0.5)
# scheduler2 = lr_scheduler.MultiStepLR(optimizer_SGDAMD2, milestones=[10,60,80], gamma=0.5)
# scheduler3 = lr_scheduler.MultiStepLR(optimizer_SGDAMD3, milestones=[10,60,80], gamma=0.5)
# scheduler4 = lr_scheduler.MultiStepLR(optimizer_SGDAMD4, milestones=[10,60,80], gamma=0.5)
# scheduler5 = lr_scheduler.MultiStepLR(optimizer_SGDAMD5, milestones=[10,60,80], gamma=0.5)
# scheduler6 = lr_scheduler.MultiStepLR(optimizer_SGDAMD5, milestones=[10,60,80], gamma=0.5)


# optimizers =[optimizer_SGDAMD,optimizer_SGDAMD2,optimizer_SGDAMD3,optimizer_SGDAMD4,optimizer_SGDAMD5,optimizer_SGDAMD6]
optimizers = [optimizer_FRGD, optimizer_Adam, optimizer_RAdam, optimizer_SGDAMD, optimizer_SGDMD]
loss_func = nn.CrossEntropyLoss()  # 使用多分类边际损失函数
# losses_his = [[], [], [], [], [],[],[],[]]
losses_his = [[], [], [], [], []]
epochs = []
steps_per_epoch = len(train_loader)  # 每个 epoch 中的步数
train_accuracies = [[] for _ in range(len(nets))]
test_accuracies = [[] for _ in range(len(nets))]
# labels=['SGDAMD(lr=0.1)','SGDAMD(lr=0.3)','SGDAMD(lr=0.5)','SGDAMD(lr=0.01)','SGDAMD(lr=1)','SGDAMD(lr=0.001)']
labels = ['FRSGD', 'Adam', 'RADam', 'SGDAMD', 'SGDMD']
# labels = ['Adam']
for epoch in range(EPOCH):
    print(f'Epoch: {epoch + 1}')
    epochs.append(epoch + 1)
    for step, (batch_x, batch_y) in enumerate(train_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        for net, optimizer, loss_his in zip(nets, optimizers, losses_his):
            net.train()
            net_output = net(batch_x)
            loss = loss_func(net_output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #             scheduler.step()
            scheduler2.step()
            scheduler3.step()
            scheduler4.step()
            scheduler5.step()
            scheduler6.step()
            loss_his.append(loss.item())

        # for i, l_his in enumerate(losses_his):
        #     print(f"优化器 {labels[i]} 的损失值: {l_his[-1]:.6f}")

    for i, net in enumerate(nets):
        train_accuracy = calculate_accuracy(net, train_loader)
        test_accuracy = calculate_accuracy(net, test_loader)
        train_accuracies[i].append(train_accuracy)
        test_accuracies[i].append(test_accuracy)
    #         print('优化器{}的训练准确率为:{}'.format(labels[i], train_accuracy))
    #         print('优化器{}的测试准确率为:{}'.format(labels[i], test_accuracy))

    for i, (loss, train_acc, test_acc) in enumerate(zip(losses_his, train_accuracies, test_accuracies)):
        print(
            f'Epoch: {epoch + 1} Algorithm {labels[i]} - Loss: {loss[-1]:.4f}, Train Accuracy: {train_acc[-1]:.2f}%, Test Accuracy: {test_acc[-1]:.2f}%')

# 计算每个 epoch 的平均 loss
avg_losses_his = [
    [sum(loss_his[i:i + steps_per_epoch]) / steps_per_epoch for i in range(0, len(loss_his), steps_per_epoch)] for
    loss_his in losses_his]

# 绘制 epoch 的图
for i, l_his in enumerate(avg_losses_his):
    plt.plot(epochs[:len(l_his)], l_his, label=labels[i], lw=1, linestyle='-')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.savefig("/kaggle/working/loss.eps", format="eps")
plt.show()

# for i, l_his in enumerate(losses_his):
#     plt.plot(l_his, label=labels[i],  lw=0.5,linestyle='--')
# plt.grid(True)
# plt.legend(loc='best')
# plt.xlabel('Steps')
# plt.ylabel('Loss')
# plt.savefig("test2.eps", dpi=300,format="eps")
# plt.show()


# 绘制 epoch 的图
for i, (train_acc, test_acc) in enumerate(zip(train_accuracies, test_accuracies)):
    plt.plot(epochs, train_acc, label=f'{labels[i]} Train', lw=2, linestyle='--', )
plt.grid(True)
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('Train Accuracy')
plt.xticks(range(0, 100, 20))
plt.yticks(range(40, 120, 20))
plt.savefig("/kaggle/working/train.eps", format="eps")
plt.show()

for i, (train_acc, test_acc) in enumerate(zip(train_accuracies, test_accuracies)):
    plt.plot(epochs, test_acc, label=f'{labels[i]} Test', lw=2, linestyle='--')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.xticks(range(0, 100, 20))
plt.yticks(range(40, 120, 20))
plt.savefig("/kaggle/working/test.eps", format="eps")
plt.show()

# for i, (train_acc, test_acc) in enumerate(zip(train_accuracies, test_accuracies)):
#     plt.plot(epochs, train_acc, label=f'{labels[i]} Train', lw=0.5,linestyle='--',)
#     plt.plot(epochs, test_acc, label=f'{labels[i]} Test', lw=0.5, linestyle='--')
# for i, (train_acc, test_acc) in enumerate(zip(train_accuracies, test_accuracies)):
#     plt.plot(epochs, train_acc, label='Train', lw=0.5, linestyle='--', )
#     plt.plot(epochs, test_acc, label='Test', lw=0.5, linestyle='--')
# plt.grid(True)
# plt.legend(loc='best')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.show()