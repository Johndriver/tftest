import torch
criterion = torch.nn.CrossEntropyLoss()

y = torch.LongTensor([2,0,1])

y_pred1 = torch.Tensor([[0.1,0.2,0.9],
                        [1.1,0.1,0.2],
                        [0.2,2.1,0.1]])
y_pred2 = torch.Tensor([[0.8,0.2,0.3],
                        [0.2,0.3,0.5],
                        [0.2,0.2,0.5]])

l1 = criterion(y_pred1,y)
l2 = criterion(y_pred2,y)

print('batch loss1=',l1.data)
print('batch loss2=',l2.data)