from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from torch import nn

from utils import set_seed

set_seed()

# SANTA FE DATASET GENERAL CONFIGURATIONS
train_set = pd.read_csv("lasertrain.dat", header=None)
test_set = pd.read_csv("laserpred.dat", header=None)
n_epochs = 1000
n_neurons_to_try = [20, 40, 80, 100, 150, 200]
criterion = nn.MSELoss()


# EXERCISE 1
def get_time_series_train_data(train_set: pd.DataFrame, p: int) -> Tuple[List[List[float]], List[float]]:
    train_raw_inputs = []
    for i in range(len(train_set) - p):
        train_raw_inputs.append(train_set[0][i : i + p].to_list())
    train_raw_targets = train_set[0][p:].to_list()
    return train_raw_inputs, train_raw_targets


def preprocess_inputs(raw_inputs: List[List[float]], raw_targets: List[float]) -> Tuple[torch.tensor, torch.tensor]:
    processed_inputs = torch.tensor(raw_inputs, dtype=torch.float)
    processed_targets = torch.tensor(raw_targets, dtype=torch.float)
    return processed_inputs, processed_targets


def get_model(n_neurons: int, lag: int) -> nn.Sequential:
    model = nn.Sequential(
        nn.Linear(lag, n_neurons),
        nn.ReLU(),
        nn.Linear(n_neurons, 1),
        nn.ReLU(),
    )
    return model


# best_mse = np.inf
# best_params = {}
# for lag in np.arange(25, 200, 5):
#     train_raw_inputs, train_raw_targets = get_time_series_train_data(train_set, lag)
#     train_inputs, train_targets = preprocess_inputs(train_raw_inputs, train_raw_targets)
#
#     for n_neurons in n_neurons_to_try:
#         model = get_model(n_neurons=n_neurons, lag=lag)
#         optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
#         model.train()
#         for epoch in range(n_epochs):
#             optimizer.zero_grad()
#             output = model(train_inputs)
#             loss = criterion(output.ravel(), train_targets)
#             loss.backward()
#             optimizer.step()
#
#             mse_train = mean_squared_error(train_targets, output.detach().numpy())
#             if mse_train < best_mse:
#                 torch.save(model, "santa_fe_model.bin")
#                 print(f"Best MSE: {mse_train}")
#                 best_params = {"n_neurons": n_neurons, "lag": lag}
#                 best_mse = mse_train
#
# print(f"Best MSE: {best_mse}, Best Parameters: {best_params}")
# # Best MSE: 1.2047756910324097, Best Parameters: {'n_neurons': 200, 'lag': 25}

# EXERCISE 2


def validate_model(train_inputs: torch.tensor, validation_targets: torch.tensor, model: nn.Sequential) -> float:
    cur_input = train_inputs[-1, :]
    val_preds = []
    for i in range(validation_targets.size()[0]):
        cur_output = model(cur_input)
        val_preds.append(cur_output.detach().numpy())

        cur_input = torch.cat([cur_input[1:], cur_output])

    mse_val = mean_squared_error(validation_targets, val_preds)
    return mse_val


best_mse = np.inf
best_params = {}
for lag in np.arange(25, 200, 5):
    raw_inputs, raw_targets = get_time_series_train_data(train_set, lag)
    processed_inputs, processed_targets = preprocess_inputs(raw_inputs, raw_targets)

    # Split into train and validation sets.
    n_train = round(processed_inputs.size()[0] * 0.8)
    train_inputs, train_targets = processed_inputs[:n_train, :], processed_targets[:n_train]
    validation_targets = processed_targets[n_train:]

    for n_neurons in n_neurons_to_try:
        model = get_model(n_neurons=n_neurons, lag=lag)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model.train()
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            output = model(train_inputs)
            loss = criterion(output.ravel(), train_targets)
            loss.backward()
            optimizer.step()

            mse_train = validate_model(train_inputs, validation_targets, model)
            if mse_train < best_mse:
                torch.save(model, "santa_fe_model.bin")
                print(f"Best MSE: {mse_train}")
                best_params = {"n_neurons": n_neurons, "lag": lag}
                best_mse = mse_train

print(f"Best MSE: {best_mse}, Best Parameters: {best_params}")
# Best MSE: 213.55189514160156, Best Parameters: {'n_neurons': 80, 'lag': 50}
