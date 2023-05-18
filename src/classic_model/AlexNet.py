import torch.nn as nn
import torch
import torchvision.models.alexnet


class AlexNet(nn.Module):
    def __init__(self, in_channels, dropout: float = 0.5):
        super(AlexNet, self).__init__()
        # output_length = floor((sequence_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
        self.features = nn.ModuleList([
            nn.Conv1d(in_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2)
        ])

        self.avgpool = nn.AdaptiveAvgPool1d(6)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 , 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 2048),  # 添加额外的全连接层
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(2048, 1)
        )
        self.regression = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 , 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 2048),  # 添加额外的全连接层
            nn.Dropout(p=dropout),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1),
        )
        self.m = nn.Sigmoid()
    def forward(self, x):
        for layer in self.features:
            # print(layer,x.shape)
            x = layer(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.m(self.classifier(x)), self.regression(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), dilation_rate=(1, 1), padding=1):
        super(ConvBlock, self).__init__()
        # 卷积
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation_rate,
                              padding=padding)
        # 归一化
        self.bn = nn.BatchNorm2d(out_channels)

    # 前行传播
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = nn.GELU()(out)
        return out


# 残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), dilation_rate=(1, 1), padding=1):
        super(ResidualBlock, self).__init__()
        # 卷积
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                               dilation=dilation_rate, padding=padding)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                               dilation=dilation_rate, padding=padding)
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                               dilation=dilation_rate, padding=padding)
        # 归一化
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

    # 前行传播
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.GELU()(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = nn.GELU()(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = nn.GELU()(out)
        out += residual  # 加入残差
        out = nn.GELU()(out)
        return out


class SamplotNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, init_weights=False):
        super(SamplotNet, self).__init__()
        self.features = nn.Sequential(
            # input layers
            nn.Conv1d(in_channels, 32, kernel_size=(7, 7), stride=(1, 1), dilation=(2, 2), padding=0),
            nn.BatchNorm2d(num_features=32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # i = 1..4
            ConvBlock(32, 32 * 1, kernel_size=(1, 1), stride=(1, 1), dilation_rate=(1, 1), padding=1),
            ResidualBlock(32, 32 * 1, kernel_size=(3, 3), stride=(1, 1), dilation_rate=(1, 1), padding=1),
            ResidualBlock(32, 32 * 1, kernel_size=(3, 3), stride=(1, 1), dilation_rate=(1, 1), padding=1),
            ResidualBlock(32, 32 * 1, kernel_size=(3, 3), stride=(1, 1), dilation_rate=(1, 1), padding=1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            ConvBlock(32 * 1, 32 * 2, kernel_size=(1, 1), stride=(1, 1), dilation_rate=(1, 1), padding=1),
            ResidualBlock(32 * 2, 32 * 2, kernel_size=(3, 3), stride=(1, 1), dilation_rate=(1, 1), padding=1),
            ResidualBlock(32 * 2, 32 * 2, kernel_size=(3, 3), stride=(1, 1), dilation_rate=(1, 1), padding=1),
            ResidualBlock(32 * 2, 32 * 2, kernel_size=(3, 3), stride=(1, 1), dilation_rate=(1, 1), padding=1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            ConvBlock(32 * 2, 32 * 3, kernel_size=(1, 1), stride=(1, 1), dilation_rate=(1, 1), padding=1),
            ResidualBlock(32 * 3, 32 * 3, kernel_size=(3, 3), stride=(1, 1), dilation_rate=(1, 1), padding=1),
            ResidualBlock(32 * 3, 32 * 3, kernel_size=(3, 3), stride=(1, 1), dilation_rate=(1, 1), padding=1),
            ResidualBlock(32 * 3, 32 * 3, kernel_size=(3, 3), stride=(1, 1), dilation_rate=(1, 1), padding=1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            ConvBlock(32 * 3, 32 * 4, kernel_size=(1, 1), stride=(1, 1), dilation_rate=(1, 1), padding=1),
            ResidualBlock(32 * 4, 32 * 4, kernel_size=(3, 3), stride=(1, 1), dilation_rate=(1, 1), padding=1),
            ResidualBlock(32 * 4, 32 * 4, kernel_size=(3, 3), stride=(1, 1), dilation_rate=(1, 1), padding=1),
            ResidualBlock(32 * 4, 32 * 4, kernel_size=(3, 3), stride=(1, 1), dilation_rate=(1, 1), padding=1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 1024),
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes),
            nn.Softmax()
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # tf.keras.layers.GlobalAveragePooling2D()
        x = nn.AdaptiveAvgPool2d((1, 1))(x).squeeze()
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
