import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from modules.process_data import read_data, compute_cost, gradient_descent, feature_normalize, normal_eqn
import ex1_py

exercise = "ex1"


if exercise == "ex1":
    warm_up = False
    show_init_plot = False

    if warm_up:
        ex1_py.warm_up()

    print("Linear Regression with one variable")
    X, y = read_data(exercise, "ex1data1", xcol=1, ycol=1)

    if show_init_plot:
        ex1_py.show_init_plot(X, y)

    m = len(y)
    X0 = np.ones((m, 1))

    X = np.concatenate((X0, X), axis=1)
    theta = np.zeros((2, 1))
    cost = compute_cost(X, y, theta)

    iterations = 1500
    alpha = 0.01

    for i in range(iterations):
        cost = compute_cost(X, y, theta)
        theta = gradient_descent(X, y, theta, alpha)

    linear_reg_plot = False
    if linear_reg_plot:
        ex1_py.linear_reg_plot(X, y, theta)

    predictions = False
    if predictions:
        predict1 = np.matmul([1, 3.5], theta)
        predict2 = np.matmul([1, 7], theta)

        print("For population = 35,000, we predict a profit of ", predict1 * 10000)
        print("For population = 70,000, we predict a profit of ", predict2 * 10000)

    visualize_cost = False

    if visualize_cost:
        ex1_py.visualize_cost(X, y)

    linear_reg_mult_var = True

    if linear_reg_mult_var:
        X_mult, y_mult = read_data(exercise, "ex1data2", xcol=2, ycol=1)

        X_norm, mean_x, std_x = feature_normalize(X_mult)

        m = len(y_mult)
        X0 = np.ones((m, 1))

        X_norm = np.concatenate((X0, X_norm), axis=1)
        theta = np.zeros((3, 1))
        cost = compute_cost(X_norm, y_mult, theta)

        iterations = 50
        alpha = 0.1
        iter_list = []
        cost_list = []

        for i in range(iterations):
            cost = compute_cost(X_norm, y_mult, theta)
            theta = gradient_descent(X_norm, y_mult, theta, alpha)
            iter_list.append(i + 1)
            cost_list.append(cost)

        print(theta)

        predict_mult = False
        if predict_mult:
            Z = np.asarray([1650, 3])
            Z_corrected = (Z - mean_x) / std_x
            Z = np.insert(Z_corrected, 0, 1)

            predicted_price = np.matmul(Z, theta)
            print("Predicted price : $", predicted_price)

        alpha_iter = False
        if alpha_iter:
            for alpha in [30, 10, 3, 1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]:
                iter_list = []
                cost_list = []

                for i in range(iterations):
                    cost = compute_cost(X_norm, y_mult, theta)
                    theta = gradient_descent(X_norm, y_mult, theta, alpha)
                    iter_list.append(i + 1)
                    cost_list.append(cost)
                title = "Alpha Value:" + str(alpha)
                ex1_py.cost_iter_plot(iter_list, cost_list, title)

        normal_eqn_mult = False
        if normal_eqn_mult:
            X_mult, y_mult = read_data(exercise, "ex1data2", xcol=2, ycol=1)

            m = len(y_mult)
            X0 = np.ones((m, 1))
            X_mult = np.concatenate((X0, X_mult), axis=1)

            theta_norm = normal_eqn(X_mult, y_mult)
            print(theta_norm)

            Z = np.asarray([1650, 3])
            Z = np.insert(Z, 0, 1)

            predicted_price = np.matmul(Z, theta_norm)
            print("Predicted price : $", predicted_price)
