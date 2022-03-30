import torch


def univariate_liner_reg_prediction(theta0, theta1, x_data):
    prediction = theta1.mm(x_data)
    return prediction
