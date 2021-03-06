import torch
import numpy as np

from modules.CostFunction import squared_error_cost_function
from modules.Prediction import univariate_liner_reg_prediction
from modules.plotter import theta_cost_plot
from modules.GradientDescent import gradient_descent

device = "cuda" if torch.cuda.is_available() else "cpu"
grad_req = True
debug = False

x_list = np.asarray([[1, 1], [1, 2], [1, 3]])
y_list = np.asarray([[1], [2], [3]])

# print(x_list.shape)
# print(y_list.shape)

x_data = torch.tensor(x_list, dtype=torch.float32,
                      device=device, requires_grad=grad_req)

y_data = torch.tensor(y_list, dtype=torch.float32,
                      device=device, requires_grad=grad_req)

training_data = torch.cat((x_data, y_data), dim=1)

# print(training_data)

theta0_tensor = torch.arange(
    -20, 20, 2, dtype=torch.float32, device=device).unsqueeze(1)
theta1_tensor = torch.arange(
    -10, 10, 1, dtype=torch.float32, device=device).unsqueeze(1)

theta = np.zeros((40000, 2))
count = 0
for i in range(-98, 102):
    for j in range(-98, 102):
        theta[count, 0] = i / 100
        theta[count, 1] = j / 100
        count += 1


alpha = 0.01
theta_tensor = torch.tensor([[5, 0]], dtype=torch.float32, device=device)
cost = squared_error_cost_function(x_data, y_data, theta_tensor, debug=debug)
grad = gradient_descent(x_data, y_data, theta_tensor, alpha, debug=debug)


print("Theta", theta_tensor)
print("Cost", cost)
print("Grad", grad)

print("\n\nEnd\n\n")
# theta_tensor = torch.tensor(theta, dtype=torch.float32, device=device)

# print(theta_tensor)

# prediction = univariate_liner_reg_prediction(
#     theta0=None, theta1=theta1_tensor, x_data=x_data)


# print(x_data.cpu().detach().numpy())

cost = squared_error_cost_function(x_data, y_data, theta_tensor, debug=debug)

# theta_cost_plot(theta_tensor, cost, debug=debug)

# print("Theta 1", theta1_tensor)
print("Cost Function", cost)

index = torch.argmin(cost)

print(theta_tensor[index])
