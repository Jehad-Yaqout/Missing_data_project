import numpy as np


def missing_value_generator(seed, data, missing_rate):
    np.random.seed(seed)
    missing_value_average_each_row = data.shape[1] * (missing_rate / 100)
    r = np.random.poisson(missing_value_average_each_row, size=(data.shape[0], 1))

    # Fixing values out of range
    r[r > data.shape[1]] = data.shape[1]
    r[r < 0] = 0
    r[r >= data.shape[1]] = data.shape[1] - 1

    X_missing = data.copy()

    for i in range(data.shape[0]):
        Temp = np.random.permutation(data.shape[1])
        if r[i, 0] > 0:
            missing_idx = Temp[:r[i, 0]]
            X_missing[i, missing_idx] = np.nan

    return X_missing


# Example usage
if __name__ == "__main__":
    seed = 42
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    missing_rate = 50  # 50% missing rate
    X_missing = missing_value_generator(seed, data, missing_rate)
    print("Original Data:\n", data)
    print("Data with Missing Values:\n", X_missing)