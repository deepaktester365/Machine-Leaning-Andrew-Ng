import matplotlib
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np


def visual_plot(x, y):
    x = x.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    plt.scatter(x, y, marker = "x")
    plt.show()


def theta_cost_plot(theta_tensor, cost_tensor, debug=False):
    theta = theta_tensor.cpu().numpy().transpose()
    t0 = theta[0]
    t1 = theta[1]
    cost = cost_tensor.cpu().detach().numpy()

    t0 = np.reshape(t0, (200, 200))
    t1 = np.reshape(t1, (200, 200))
    cost = np.reshape(cost, (200, 200))

    if debug:
        print(theta_tensor, "Theta tensor")
        print(theta)
        print(cost_tensor, "Cost tensor")
        print(t0.shape, "Theta 0")
        print(t1.shape, "Theta 1")
        print(cost.shape, "Cost")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(t0, t1, cost)

    # ax.plot_surface(t0, t1, cost,
    #                 cmap="autumn_r", lw=0, rstride=1, cstride=1)
    # ax.contour(theta[0], theta[1], cost, 10, lw=3,
    #            colours="k", linestyles="solid")

    # plt.plot(theta, cost)
    plt.show()
