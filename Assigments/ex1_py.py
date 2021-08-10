import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from modules.process_data import read_data, compute_cost, gradient_descent


def warm_up():
    print("Warmup exercise")
    print(np.identity(5))


def show_init_plot(X, y):
    plt.scatter(X, y, marker="X", color="red", s=25)
    plt.xlim(5, 25)
    plt.ylim(-5, 25)
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000s")
    plt.show()


def liner_reg_plot(X, y, theta):
    Xval = np.transpose(X)[1]
    Yval = np.transpose(y)
    plt.scatter(Xval, Yval, marker="X", color="red", s=25)
    plt.plot(Xval, np.matmul(X, theta))
    plt.xlim(5, 25)
    plt.ylim(-5, 25)
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000s")
    plt.show()


def visualize_cost(X, y):
    theta0_vals = np.linspace(-10, 10, 100)[:, None]
    theta1_vals = np.linspace(-1, 4, 100)[:, None]

    Cost_vals = np.zeros([len(theta0_vals), len(theta1_vals)])

    np_X = []
    np_Y = []
    np_Z = []
    for i in range(0, len(theta0_vals)):
        for j in range(0, len(theta1_vals)):
            t = [theta0_vals[i], theta1_vals[j]]
            Cost_vals[i][j] = compute_cost(X, y, t)
            np_X.append(theta0_vals[i][0])
            np_Y.append(theta1_vals[j][0])
            np_Z.append(Cost_vals[i][j])

    np_X = np.asarray(np_X)
    np_Y = np.asarray(np_Y)
    np_Z = np.asarray(np_Z)
    fig = plt.figure()
    plt.xlim(-10, 10)
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    surf = ax.plot_trisurf(np_X, np_Y, np_Z, cmap=cm.jet, linewidth=0.1)
    plt.show()


def cost_iter_plot(Cost, iter, title):
    plt.plot(Cost, iter)
    plt.title(title)
    plt.show()
