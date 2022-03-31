import torch
import numpy as np

from CostFunction import squared_error_cost_function
from Prediction import univariate_liner_reg_prediction
from plotter import theta_cost_plot

device = "cuda" if torch.cuda.is_available() else "cpu"
grad_req = True

x_list = np.asarray([[1, 2, 3]])
y_list = np.asarray([[1, 2, 3]])

# print(x_list.size)

x_data = torch.tensor(x_list, dtype=torch.float32,
                      device=device, requires_grad=grad_req)

y_data = torch.tensor(y_list, dtype=torch.float32,
                      device=device, requires_grad=grad_req)

training_data = torch.stack((x_data, y_data))

theta1_tensor = torch.arange(
    -10, 10, dtype=torch.float32, device=device).unsqueeze(1)

prediction = univariate_liner_reg_prediction(
    theta0=None, theta1=theta1_tensor, x_data=x_data)


# print(x_data.cpu().detach().numpy())

cost = squared_error_cost_function(prediction, y_data)

theta_cost_plot(theta1_tensor, cost)

print("Theta 1", theta1_tensor)
print("Cost Function", cost)
