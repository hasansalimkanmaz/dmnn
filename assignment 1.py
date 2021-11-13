import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import scipy.io
import random
import matplotlib as mpl
from sklearn.metrics import mean_squared_error

mpl.use('macosx')

# Seeding for the reproducibility of results.
seed = 0
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

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
variables = scipy.io.loadmat("data_personal_regression_problem.mat")

x1 = variables["X1"]
x2 = variables["X2"]
t1 = variables["T1"]
t2 = variables["T2"]
t3 = variables["T3"]
t4 = variables["T4"]
t5 = variables["T5"]

t_new = (8 * t1 + 7 * t2 + 4 * t3 + 3 * t4 + 2 * t5) / (8 + 7 + 4 + 3 + 2)

n_instances = t_new.shape[0]
sample_indexes = random.sample(list(range(n_instances)), 3000)

train_indexes = sample_indexes[:1000]
validation_indexes = sample_indexes[1000:2000]
test_indexes = sample_indexes[2000:]

train_set = np.hstack([x1[train_indexes], x2[train_indexes], t_new[train_indexes]])
validation_set = np.hstack([x1[validation_indexes], x2[validation_indexes], t_new[validation_indexes]])
test_set = np.hstack([x1[test_indexes], x2[test_indexes], t_new[test_indexes]])


model = nn.Sequential(
    nn.Linear(2, 100),
    nn.ReLU(),
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 25),
    nn.ReLU(),
    nn.Linear(25, 1),
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# train_inputs = torch.tensor(train_set[:, [0, 1]], dtype=torch.float)
# train_set = train_set[0:100, :]
train_inputs = torch.tensor(train_set[:, [0, 1]], dtype=torch.float)
train_targets = torch.tensor(train_set[:, 2], dtype=torch.float)
validation_inputs = torch.tensor(validation_set[:, [0, 1]], dtype=torch.float)
validation_targets = torch.tensor(validation_set[:, 2], dtype=torch.float)
test_inputs = torch.tensor(test_set[:, [0, 1]], dtype=torch.float)
test_targets = torch.tensor(test_set[:, 2], dtype=torch.float)
n_epochs = 200

train_losses, validation_losses, test_losses = [], [], []
model.train()
best_val_loss = np.inf
saved_model_dir = "assignment_1_model.bin"
for epoch in range(n_epochs):
    optimizer.zero_grad()

    train_preds = model(train_inputs)
    train_loss = criterion(train_preds.ravel(), train_targets)
    train_losses.append(train_loss.detach().numpy())

    validation_preds = model(validation_inputs)
    validation_loss = criterion(validation_preds, validation_targets)
    validation_losses.append(validation_loss.detach().numpy())

    if validation_loss < best_val_loss:
        print(f"The best model until now after epoch {epoch}")
        torch.save(model, "assignment_1_model.bin")
        best_val_loss = validation_loss

    test_preds = model(test_inputs)
    test_loss = criterion(test_preds, test_targets)
    test_losses.append(test_loss.detach().numpy())

    print(f"Epoch: {epoch}, Train Loss: {train_loss}, Validation Loss: {validation_loss}, Test Loss: {test_loss}.")
    train_loss.backward()
    optimizer.step()

best_model = torch.load(saved_model_dir)
train_preds = model(train_inputs)
test_preds = model(test_inputs)

fig = plt.figure()
plt.plot(train_losses)
plt.plot(validation_losses)
plt.plot(test_losses)
plt.legend(["Train Loss", "Validation Loss", "Test Loss"])
plt.yscale('log')
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(test_set[:, 0], test_set[:, 1], test_preds.detach().numpy(), c="blue")
ax.scatter(test_set[:, 0], test_set[:, 1], test_set[:, 2], c="red")
plt.show()

mse_test = mean_squared_error(test_set[:, 2], test_preds.detach().numpy(), squared=False)
print(f"RMSE on test set: {mse_test}")

