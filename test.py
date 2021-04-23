# from tensorflow.python.client import device_lib
import torch

# print(device_lib.list_local_devices())
print(torch.cuda.is_available())
v = torch.version
print(torch.version.cuda)