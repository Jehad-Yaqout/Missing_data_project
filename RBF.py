import numpy as np


def RBF(S, Y, flag, epsilon=1e-8):
    """
    Radial Basis Function interpolation.

    Parameters:
    S (np.array): Training data input.
    Y (np.array): Training data target values.
    flag (str): The type of RBF to use ('cubic', 'TPS', 'linear').
    epsilon (float): Regularization term to avoid singularity in matrix A.

    Returns:
    tuple: (lambda_, gamma) parameters for the RBF interpolation.
    """
    [m, n] = S.shape
    P = np.hstack((S, np.ones((m, 1))))

    R = np.zeros((m, m))
    for ii in range(m):
        for jj in range(ii, m):
            R[ii, jj] = np.sum((S[ii, :] - S[jj, :]) ** 2)
            R[jj, ii] = R[ii, jj]

    R = np.sqrt(R)

    if flag == 'cubic':
        Phi = R ** 3
    elif flag == 'TPS':
        R[R == 0] = 1
        Phi = R ** 2 * np.log(R)
    elif flag == 'linear':
        Phi = R

    # Regularization addition to ensure A is non-singular
    A = np.vstack((np.hstack((Phi, P)), np.hstack((P.T, np.zeros((n + 1, n + 1))))))
    A += epsilon * np.eye(A.shape[0])

    RHS = np.vstack((Y.reshape(-1, 1), np.zeros((n + 1, 1))))
    params = np.linalg.solve(A, RHS)
    lambda_ = params[:m].flatten()
    gamma = params[m:].flatten()

    return lambda_, gamma


# Testing the RBF function with sample data
TrainingData = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
    [10.0, 11.0, 12.0],
    [13.0, 14.0, 15.0]
])

TrainingTargets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

# Using the RBF function
lambda_, gamma = RBF(TrainingData, TrainingTargets, 'cubic')

print("lambda:", lambda_)
print("gamma:", gamma)
