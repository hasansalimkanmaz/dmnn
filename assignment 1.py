import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
from sklearn.metrics import mean_squared_error
from torch import nn

from utils import set_seed

mpl.use("macosx")

set_seed()

# EXERCISE 1.1.2
x = np.arange(0, 21, 1)
y = -np.sin(0.8 * np.pi * x)
plt.scatter(x, y)
plt.plot(x, y)


# EXERCISE 1.1.3
model = nn.Sequential(
    nn.Linear(1, 2),
    nn.Tanh(),
    nn.Linear(2, 1),
)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
n_epochs = 100

x = torch.tensor(np.expand_dims(x, axis=1), dtype=torch.float)
y = torch.tensor(y, dtype=torch.float)
model.train()
for epoch in range(n_epochs):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output.ravel(), y)
    print(loss)
    loss.backward()
    optimizer.step()

# as can be seen from the plot, a linear model can't capture the relationship between x and y.
plt.plot(x, output.detach().numpy())
plt.legend(["data points", "fitted model"])
plt.show()


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
plt.yscale("log")
plt.show()

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter(test_set[:, 0], test_set[:, 1], test_preds.detach().numpy(), c="blue")
ax.scatter(test_set[:, 0], test_set[:, 1], test_set[:, 2], c="red")
plt.show()

mse_test = mean_squared_error(test_set[:, 2], test_preds.detach().numpy(), squared=False)
print(f"RMSE on test set: {mse_test}")
