import torch
import torch.nn as nn
import math


class AlexNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(AlexNet, self).__init__()

        self.features = nn.ModuleList([
            nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        ])

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# 定义AlexNet模型类（与上述示例相同）
# ...

# 数据预处理和加载
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_dataset = datasets.CIFAR10(root='./data/', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)

# 创建AlexNet实例
in_channels = 3  # 输入通道数
num_classes = 10  # 输出类别数
model = AlexNet(in_channels, num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 训练循环
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(2):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1} - Loss: {running_loss / len(train_loader)}")
import torch
import torch.nn as nn


def get_conv1d_output_dim(sequence_length, kernel_size, stride, padding, dilation=1):
    return math.floor((sequence_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)


def get_maxpool1d_output_dim(input_dim, kernel_size, stride):
    return math.floor((input_dim - kernel_size) / stride) + 1


if __name__ == '__main__':
    print(get_conv1d_output_dim(5000, 11, 4, 2))
    print(get_maxpool1d_output_dim(1249, 3,2))
    print(get_conv1d_output_dim(624, 5, 1, 2))
    print(get_maxpool1d_output_dim(624, 3,2))


    print(get_conv1d_output_dim(1249, 11, 4, 2))
