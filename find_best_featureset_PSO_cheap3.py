import numpy as np
from Fitness_fn import predict_alaa, Fitness_fn  # Import your modifications
from scipy.optimize import differential_evolution
import time


def find_best_featureset_PSO_cheap3(OriginalData, ImputedData, num_features, timeout=300):
    """
    Use Particle Swarm Optimization (PSO) to find the best feature set.

    Parameters:
    OriginalData (np.array): Original data with missing values.
    ImputedData (np.array): Data with imputed values.
    num_features (int): Number of features to select.
    timeout (int): Maximum computation time in seconds. Default is 300 seconds.

    Returns:
    list: Best feature set found.
    """
    num_columns = OriginalData.shape[1]
    start_time = time.time()

    def fitness_wrapper(indices):
        indices = np.argsort(indices)
        selected_columns = indices[:num_features].astype(int)
        current_col = selected_columns[0]
        remaining_cols = selected_columns[1:]

        if time.time() - start_time > timeout:
            raise TimeoutError("Timeout reached during optimization")

        return Fitness_fn(current_col, remaining_cols, OriginalData, ImputedData)

    bounds = [(0, num_columns - 1) for _ in range(num_columns)]

    try:
        result = differential_evolution(fitness_wrapper, bounds, strategy='rand1bin', maxiter=1000, tol=0.01)
    except TimeoutError:
        print("Optimization timed out. Returning partial results.")
        best_indices = np.random.permutation(num_columns)
    else:
        best_indices = np.argsort(result.x)

    best_feature_set = best_indices[:num_features].astype(int)
    return best_feature_set


# Example usage
if __name__ == '__main__':
    np.random.seed(0)
    OriginalData = np.random.rand(100, 10)
    ImputedData = np.random.rand(100, 10)  # Example imputed data
    num_features = 6  # Ensure we are selecting enough features
    best_features = find_best_featureset_PSO_cheap3(OriginalData, ImputedData, num_features)
    print(best_features)
