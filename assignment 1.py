import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)
# INTRODUCTION

inputs = torch.tensor([[2, 1, -2, -1], [2, -2, 2, 1]], dtype=torch.float)
target = [0, 1, 0, 1]

perceptron = nn.Linear(4, 1)

# change weights
perceptron.bias = nn.Parameter(torch.zeros_like(perceptron.bias))
perceptron.weight = nn.Parameter(torch.zeros_like(perceptron.weight))

# EXERCISE 1.1.2
x = np.arange(0, 21, 1)
y = - np.sin(0.8 * np.pi * x)
plt.scatter(x, y)
plt.plot(x, y)

perceptron = nn.Linear(1, 1)
criterion = nn.L1Loss()  # any loss for a regression task
optimizer = torch.optim.SGD(perceptron.parameters(), lr=0.001)
n_epochs = 100

x = torch.tensor([[i] for i in x], dtype=torch.float)
y = torch.tensor(y, dtype=torch.long)
for epoch in range(n_epochs):
    perceptron.train()
    optimizer.zero_grad()
    output = perceptron(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

# as can be seen from the plot, a linear model can't capture the relationship between x and y.
plt.plot(x, output.detach().numpy())
plt.legend(["data points", "fitted perceptron"])
# ######################################################################################################################

# EXERCISE 3.1
import scipy.io
import random
from mpl_toolkits import mplot3d

variables = scipy.io.loadmat("data_personal_regression_problem.mat")

x1 = variables["X1"]
x2 = variables["X2"]
t1 = variables["T1"]
t2 = variables["T2"]
t3 = variables["T3"]
t4 = variables["T4"]
t5 = variables["T5"]

t_new = (8*t1 + 7*t2 + 4*t3 + 3*t4 + 2*t5) / (8 + 7 + 4 + 3 + 2)

n_instances = t_new.shape[0]

random.seed(1773)
sample_indexes = random.sample(list(range(n_instances)), 3000)

train_indexes = sample_indexes[:1000]
validation_indexes = sample_indexes[1000:2000]
test_indexes = sample_indexes[2000:]

train_set = np.hstack([x1[train_indexes], x2[train_indexes], t_new[train_indexes]])
validation_set = np.hstack([x1[validation_indexes], x2[validation_indexes], t_new[validation_indexes]])
test_set = np.hstack([x1[test_indexes], x2[test_indexes], t_new[test_indexes]])

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(train_set[:, 0], train_set[:, 1], train_set[:, 2], c=train_set[:, 2])
plt.show()

