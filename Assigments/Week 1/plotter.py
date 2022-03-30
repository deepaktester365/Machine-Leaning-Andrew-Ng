import matplotlib
import matplotlib.pyplot as plt


def theta_cost_plot(theta_tensor, cost_tensor):
    theta = theta_tensor.cpu().numpy()
    cost = cost_tensor.cpu().detach().numpy()

    plt.plot(theta, cost)
    plt.show()
