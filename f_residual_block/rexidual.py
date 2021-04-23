import torch
from torchvision import transforms  # 针对图像做处理
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

# 残差网络 解决梯度消失
batch_size = 64
# 使用GPU

# toTensor()：单通道28*28的图像（像素值0-255） -》 1*28*28（值0-1）的张量
# Normalize 归一化 均值 0.1307 标准差0.3081
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
# 训练
train_dataset = datasets.MNIST(root='../data/mnist/',
                               train=True,
                               download=True,
                               transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
# 测试
test_dataset = datasets.MNIST(root='../data/mnist/',
                              train=False,
                              download=True,
                              transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = F.relu((self.conv1(x)))
        y = self.conv2(y)
        return F.relu(x + y)  # 将x和y相加再激活


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)

        self.rblock1 = ResidualBlock(channels=16)
        self.rblock2 = ResidualBlock(channels=32)

        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.rblock1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.rblock2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x


model = Net()
# 使用gpu 步骤1 cuda:0 第一块显卡
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 使用gpu 步骤2 模型放入显卡
model.to(device)

# 损失函数
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # momentum 冲量


# 单轮训练过程
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        # 训练集的输入输出 input 64x1x28x28 taget 64
        inputs, target = data
        # 使用gpu 步骤3 输入和已知结果放入显卡
        inputs, target = inputs.to(device), target.to(device)
        # 优化器清零
        optimizer.zero_grad()
        # 前馈 output 64x10
        outputs = model(inputs)
        # 单个值 损失总和或平均值
        loss = criterion(outputs, target)
        # 反馈

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d,%5d] loss:%.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


# 单轮测试过程
def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 不计算梯度
        for data in test_loader:
            images, lables = data
            # 使用gpu 步骤4 测试的输入输出放入显卡
            images, lables = images.to(device), lables.to(device)
            outputs = model(images)
            # outputs.data为64 x 10 dim=1 表示比较第1个（从0开始）即10的这个维度的最大值
            # torch.max 返回 最大值和下标，这里不用最大值，使用_接受，只用最大值得下标
            # predicted 64
            _, predicted = torch.max(outputs.data, dim=1)
            total += lables.size(0)  # lables.size(0) 有多少条数据
            # predictec 64个值表示预测的下标  lables 实际64个下标 (predicted == lables).sum()计算对应相等的数量
            correct += (predicted == lables).sum().item()
        print('accuracy on test set: %.3f %% [%d/%d]' % (100 * correct / total, correct, total))


if __name__ == '__main__':
    for epoch in range(50):
        train(epoch)
        test()
