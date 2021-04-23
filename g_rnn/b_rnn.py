import torch

batch_size = 1
seq_size = 3
input_size = 4
hidden_size = 2
num_layers = 1

cell = torch.nn.RNN(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers)

inputs = torch.randn(seq_size,batch_size,input_size)
hidden = torch.zeros(num_layers,batch_size,hidden_size)

out, hidden = cell(inputs, hidden)

print('out size: ',out.shape)
print('output', out)
print('hidden size', hidden.shape)
print('hidden', hidden)