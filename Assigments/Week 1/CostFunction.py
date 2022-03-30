import torch


def squared_error_cost_function(prediction, y_data):
    diff = torch.sum(prediction - y_data, dim=1)
    m = len(y_data)
    cost = (1 / (2 * m)) * pow(diff, 2)
    return cost
