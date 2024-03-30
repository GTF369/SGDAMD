import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from torch import Tensor
import numpy as np
import Optimizer as Opt
from torch.optim import lr_scheduler


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



EPOCH = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_data = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_data = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 1,28x28
        self.conv1 = nn.Conv2d(1, 10, 5)  # 10, 24x24
        self.conv2 = nn.Conv2d(10, 20, 3)  # 128, 10x10
        self.fc1 = nn.Linear(20 * 10 * 10, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        in_size = x.size(0)
        out = self.conv1(x)  # 24
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2)  # 12
        out = self.conv2(out)  # 10
        out = F.relu(out)
        out = out.view(in_size, -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out


# net_SGD = Net().to(device)
# net_Momentum = Net().to(device)
net_RMSprop = Net().to(device)
net_Adam = Net().to(device)
net_SGDAMD = Net().to(device)
net_FRGD = Net().to(device)
net_SGDMD = Net().to(device)
# nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam, net_SGDAMD,net_FRGD,net_SGDMD,net_PID ]
# nets = [net_RMSprop, net_Adam]
# nets = [net_Momentum]
nets = [net_RMSprop, net_Adam, net_SGDAMD,  net_FRGD,net_SGDMD]
# optimizer_SGD = torch.optim.SGD(net_SGD.parameters(), lr=0.01)
# optimizer_Momentum = Opt.SGDM(net_Momentum.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
optimizer_RMSprop = Opt.RAdam(net_RMSprop.parameters(), lr=0.001, betas=(0.9, 0.99), eps=1e-06, weight_decay=0.0005)
optimizer_Adam = Opt.Adam(net_Adam.parameters(), lr=0.001, betas=(0.9, 0.99), eps=1e-06, weight_decay=0.0005)
optimizer_SGDAMD = Opt.SGDAMD(net_SGDAMD.parameters(), lr=0.1, weight_decay=5e-4)
optimizer_FRGD = Opt.FRGD(net_FRGD.parameters(), lr=0.1, weight_decay=5e-4)
optimizer_SGDMD = Opt.SGDMD(net_SGDMD.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# scheduler = lr_scheduler.MultiStepLR(optimizer_Momentum , milestones=[10,30], gamma=0.1)
scheduler2 = lr_scheduler.MultiStepLR(optimizer_RMSprop, milestones=[30,40], gamma=0.5)
scheduler3 = lr_scheduler.MultiStepLR(optimizer_Adam, milestones=[30,40], gamma=0.5)
scheduler4 = lr_scheduler.MultiStepLR(optimizer_SGDAMD, milestones=[30,40], gamma=0.5)
scheduler5 = lr_scheduler.MultiStepLR(optimizer_FRGD, milestones=[30,40], gamma=0.5)
scheduler6 = lr_scheduler.MultiStepLR(optimizer_SGDMD, milestones=[30,40], gamma=0.5)
optimizers = [ optimizer_RMSprop, optimizer_Adam, optimizer_SGDAMD, optimizer_FRGD, optimizer_SGDMD]
# optimizers = [optimizer_Momentum]
# optimizers = [ optimizer_RMSprop, optimizer_Adam]
loss_func = nn.CrossEntropyLoss()  # 使用多分类边际损失函数

# losses_his = [[], [], [], [], [], []]
losses_his = [ [], [], [], [], []]
# losses_his = [[]]
epochs = []
steps_per_epoch = len(train_loader)  # 每个 epoch 中的步数
train_accuracies = [[] for _ in range(len(nets))]
test_accuracies = [[] for _ in range(len(nets))]

labels = ['RAdam', 'Adam', 'SGDAMD', 'FRSGD', 'SGDMD']
# labels = ['SGDM']

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
            # scheduler.step()
            # scheduler2.step()
            # scheduler3.step()
            # scheduler4.step()
            # scheduler5.step()
            # scheduler6.step()
            loss_his.append(loss.item())


    # 计算训练和测试准确率
    for i, net in enumerate(nets):
        train_accuracy = calculate_accuracy(net, train_loader)
        test_accuracy = calculate_accuracy(net, test_loader)
        train_accuracies[i].append(train_accuracy)
        test_accuracies[i].append(test_accuracy)

    for i, (loss, train_acc, test_acc) in enumerate(zip(losses_his, train_accuracies, test_accuracies)):
        print(
            f'Epoch: {epoch + 1} Algorithm {labels[i]} - Loss: {loss[-1]:.4f}, Train Accuracy: {train_acc[-1]:.2f}%, Test Accuracy: {test_acc[-1]:.2f}%')

# labels = ['SGD', 'SGDM', 'RMSprop', 'Adam', 'SGDAMD', 'FRGD', 'SGDMD', 'PID']

# 计算每个 epoch 的平均 loss
avg_losses_his = [
    [sum(loss_his[i:i + steps_per_epoch]) / steps_per_epoch for i in range(0, len(loss_his), steps_per_epoch)] for
    loss_his in losses_his]

# 绘制 epoch 的图
for i, l_his in enumerate(avg_losses_his):
    plt.plot(epochs[:len(l_his)], l_his, label=labels[i], lw=2)
plt.legend(loc='best',framealpha=1)
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
# plt.xticks(range(0,60,10))
plt.savefig("loss.eps", format="eps")
plt.show()

# for i, l_his in enumerate(losses_his):
#     plt.plot(l_his, label=labels[i],  lw=0.5)
# plt.grid(True)
# plt.legend(loc='best')
# plt.xlabel('Steps')
# plt.ylabel('Loss')
# plt.ylim(-0.25, 5)
# plt.show()

# 绘制训练准确率曲线
for i, train_acc in enumerate(train_accuracies):
    plt.plot(epochs, train_acc, label=f'{labels[i]} Train', lw=2, linestyle='--')

plt.grid(True)
plt.legend(loc='best',framealpha=1)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
# plt.xticks(range(0,60,10))
# plt.yticks(range(95,102,1))
plt.savefig("train.eps", format="eps")
plt.show()

# 绘制测试准确率曲线
for i, test_acc in enumerate(test_accuracies):
    plt.plot(epochs, test_acc, label=f'{labels[i]} Test', lw=2, linestyle='--')

plt.grid(True)
plt.legend(loc='best',framealpha=1)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Testing Accuracy')
# plt.xticks(range(0,60,10))
# plt.yticks(range(95,102,1))
plt.savefig("test.eps", format="eps")
plt.show()
