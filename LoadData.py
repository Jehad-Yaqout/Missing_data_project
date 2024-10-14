import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


def LoadData(PlotFlag):
    iris = datasets.load_iris()
    X = iris.data
    y = np.array([1] * 50 + [2] * 50 + [3] * 50)

    if PlotFlag:
        plt.figure()
        plt.plot(X[y == 1, 0], X[y == 1, 1], 'bx', markersize=16, linewidth=3, label='Class 1')  # First class
        plt.plot(X[y == 2, 0], X[y == 2, 1], 'gx', markersize=16, linewidth=3, label='Class 2')  # Second class
        plt.plot(X[y == 3, 0], X[y == 3, 1], 'rx', markersize=16, linewidth=3, label='Class 3')  # Third class
        plt.legend()
        plt.gca().set_fontsize = 20
        plt.gcf().set_facecolor = 'w'
        plt.show()

    ClassNames = np.unique(y)
    C = len(ClassNames)
    N, dim = X.shape
    Lb = np.min(X, axis=0)
    Ub = np.max(X, axis=0)

    return X, y, ClassNames, C, N, dim, Lb, Ub


# Example usage
if __name__ == "__main__":
    PlotFlag = 1  # or 0 if you do not want to plot
    X, Y, ClassNames, C, N, dim, Lb, Ub = LoadData(PlotFlag)
    print(f"X: {X.shape}, Y: {Y.shape}")
    print(f"ClassNames: {ClassNames}, C: {C}, N: {N}, dim: {dim}")
    print(f"Lb: {Lb}, Ub: {Ub}")