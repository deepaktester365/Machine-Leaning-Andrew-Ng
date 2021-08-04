import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from modules.process_data import read_data, compute_cost, gradient_descent

exercise = "ex1"


if exercise == "ex1":
    warm_up = False
    show_plot = False

    if warm_up:
        print("Warmup exercise")
        print(np.identity(5))

    print("Linear Regression with one variable")
    X, y = read_data(exercise, "ex1data1", xcol=1, ycol=1)

    if show_plot:
        plt.scatter(X, y, marker="X", color="red", s=25)
        plt.xlim(5, 25)
        plt.ylim(-5, 25)
        plt.xlabel("Population of City in 10,000s")
        plt.ylabel("Profit in $10,000s")
        plt.show()

    m = len(y)
    X0 = np.ones((m, 1))

    X = np.concatenate((X0, X), axis=1)
    theta = np.zeros((2, 1))
    cost = compute_cost(X, y, theta)
    print(cost)

    iterations = 1
    alpha = 0.01

    for i in range(iterations):
        cost = compute_cost(X, y, theta)
        print("Theta", theta, "\t", "Cost", cost)
        theta = gradient_descent(X, y, theta, alpha)
