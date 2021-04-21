import torch

in_channels, out_channels = 5, 10
width, height = 100, 100
kernel_size = 3
batch_size = 1

input = torch.randn(batch_size,
                    in_channels,
                    width,
                    height)
conv_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)

output = conv_layer(input)

print(input.shape)  # [1, 5, 100, 100]
print(output.shape)  # [1, 10, 98, 98]
print(conv_layer.weight.shape)  # [10, 5, 3, 3] 卷积核叠加
