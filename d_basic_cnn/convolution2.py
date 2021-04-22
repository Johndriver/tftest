import torch

in_channels, out_channels = 1, 1
kernel_size = 3
batch_size = 1

input = [
    3,4,6,5,7,
    2,4,6,8,2,
    1,6,7,8,4,
    9,7,4,6,2,
    3,7,6,4,1
]
input = torch.Tensor(input).view(1,1,5,5) # batch chanel width height
# kernel_size 卷积核大小 padding几个外圈填充  stride卷积核移动步长（默认1）
conv_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,padding = 1,stride=2,bias = False)

kernel = torch.Tensor([1,2,3,4,5,6,7,8,9]).view(out_channels,in_channels,kernel_size,kernel_size) #out in w h
conv_layer.weight.data = kernel.data


output = conv_layer(input)

print(output)
