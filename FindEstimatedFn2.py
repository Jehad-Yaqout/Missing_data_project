import numpy as np
from scipy.stats import qmc
from RBF_eval import RBF_eval


def FindEstimatedFn2(GlobalBest, Positions, lambda_, gamma):
    """
    Find estimated function values at new random points and the original positions.

    Parameters:
    GlobalBest : object
        An object representing the global best solution found so far.
    Positions : np.array
        The positions at which the original function values are known.
    lambda_ : np.array
        Parameter vector returned by the RBF function.
    gamma : np.array
        Parameters of the optional polynomial tail returned by the RBF function.

    Returns:
    EstimatedFn : np.array
        The estimated function values at the new random points.
    EstimatedFn_ours : np.array
        The estimated function values at the original positions.
    AllNew_Solns : np.array
        The matrix of all new solutions.
    """

    sampler = qmc.LatinHypercube(d=Positions.shape[1])
    Random_soln = sampler.random(1000)

    minxrange = 1
    sigma_stdev = [
        0.5 * minxrange,
        0.05 * minxrange,
        0.005 * minxrange
    ]

    Best_position = GlobalBest.Position
    NewPoints = []

    for _ in range(1000):
        p = np.random.permutation(len(sigma_stdev))
        new_point = np.maximum(0,
                               np.minimum(Best_position + sigma_stdev[p[0]] * np.random.randn(len(Best_position)), 1))
        NewPoints.append(new_point)

    NewPoints = np.array(NewPoints)
    AllNew_Solns = np.vstack((NewPoints, Random_soln))

    del Random_soln, NewPoints

    EstimatedFn = RBF_eval(AllNew_Solns, Positions, lambda_, gamma, 'cubic')
    EstimatedFn_ours = RBF_eval(Positions, Positions, lambda_, gamma, 'cubic')

    return EstimatedFn, EstimatedFn_ours, AllNew_Solns


# Example usage
if __name__ == '__main__':
    class GlobalBest:
        def __init__(self, position):
            self.Position = position


    GlobalBest_example = GlobalBest(np.random.rand(10))
    Positions = np.random.rand(10, 3)
    lambda_ = np.random.rand(10)
    gamma = np.random.rand(4)

    EstimatedFn, EstimatedFn_ours, AllNew_Solns = FindEstimatedFn2(GlobalBest_example, Positions, lambda_, gamma)
    print("EstimatedFn:", EstimatedFn)
    print("EstimatedFn_ours:", EstimatedFn_ours)
    print("AllNew_Solns:", AllNew_Solns)
