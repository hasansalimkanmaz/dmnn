import torch
from torch import nn


first_layer = nn.Linear(2, 2)
first_layer.bias = nn.Parameter(torch.tensor([0.5, 0.4]), requires_grad=False)
first_layer.weight = nn.Parameter(torch.tensor([[-0.2, -0.7],
                                                [-0.8, 0.6]]), requires_grad=False)
first_act = nn.Sigmoid()
second_layer = nn.Linear(2, 1)
second_layer.bias = nn.Parameter(torch.tensor([-0.5]), requires_grad=False)
second_layer.weight = nn.Parameter(torch.tensor([[-0.2, 1.3]]), requires_grad=False)


in1 = torch.tensor([3, 0], dtype=torch.float)

l1_out = first_layer(in1)
a1_out = first_act(l1_out)

print(a1_out)
l2_out = second_layer(a1_out)
print(l2_out)

