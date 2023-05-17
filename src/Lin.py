import torch
import torch.nn as nn

input_tensor = torch.randn(17, 512, 768)
conv_layer = nn.Conv1d(768, 64, kernel_size=1)  # 输入通道数为768，输出通道数为64，卷积核大小为1
output_tensor = conv_layer(input_tensor.permute(0, 2, 1))  # 将输入张量维度调整为(16, 768, 512)后进行卷积操作
output_tensor = output_tensor.permute(0, 2, 1)  # 调整输出张量的维度为(16, 512, 64)

print(output_tensor.shape)
