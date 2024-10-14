import numpy as np


def RBF_eval(X, S, lambda_, gamma, flag):
    """
    Calculates the estimated function values at points in X based on known points in S using RBF.

    Parameters:
    X : np.array
        Points where function values should be calculated.
    S : np.array
        Points where the function values are known.
    lambda_ : np.array
        Parameter vector.
    gamma : np.array
        Parameters of the optional polynomial tail.
    flag : str
        A string indicating which RBF to be used ('cubic', 'TPS', 'linear').

    Returns:
    np.array
        The estimated function values at the points in X.
    """

    # --------------------------------------------------------------------------
    # Copyright (c) 2012 by Juliane Mueller
    #
    # This file is part of the surrogate model module toolbox.
    #
    # --------------------------------------------------------------------------
    # Author information
    # Juliane Mueller
    # Tampere University of Technology, Finland
    # juliane.mueller2901@gmail.com
    # --------------------------------------------------------------------------
    #
    # input:
    # X are points where function values should be calculated
    # S are points where the function values are known
    # lambda parameter vector
    # gamma contains the parameters of the optional polynomial tail
    # flag is a string indicating which RBF to be used
    # output:
    # the estimated function values at the points in X
    # --------------------------------------------------------------------------

    mX, nX = X.shape
    mS, nS = S.shape

    if nX != nS:  # Check that both matrices are of the right shape
        X = X.T
        mX, nX = X.shape

    R = np.zeros((mX, mS))  # Compute pairwise distances of points in X and S
    for ii in range(mX):
        for jj in range(mS):
            R[ii, jj] = np.linalg.norm(X[ii, :] - S[jj, :])

    if flag == 'cubic':
        Phi = R ** 3
    elif flag == 'TPS':
        R[R == 0] = 1
        Phi = R ** 2 * np.log(R)
    elif flag == 'linear':
        Phi = R

    Yest1 = Phi.dot(lambda_)  # First part of response surface
    Yest2 = np.dot(np.hstack((X, np.ones((mX, 1)))), gamma)  # Optional polynomial tail
    Yest = Yest1 + Yest2  # Predicted function value

    return Yest


# Example usage
if __name__ == '__main__':
    X = np.random.rand(5, 3)
    S = np.random.rand(10, 3)
    lambda_ = np.random.rand(10)
    gamma = np.random.rand(4)
    flag = 'cubic'
    Yest = RBF_eval(X, S, lambda_, gamma, flag)
    print("Estimated Function Values:", Yest)