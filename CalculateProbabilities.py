import numpy as np


def CalculateProbabilities(DetailedPredictions):
    # Ensure DetailedPredictions is a numpy array and contains only numeric values
    DetailedPredictions = np.array(DetailedPredictions, dtype=float)

    # Flatten the array and apply unique on numeric values to ensure comparability
    unique_values = np.unique(DetailedPredictions.flatten())
    Prob = np.zeros((DetailedPredictions.shape[0], len(unique_values)))

    for idx, unique_value in enumerate(unique_values):
        Prob[:, idx] = np.sum(DetailedPredictions == unique_value, axis=1) / DetailedPredictions.shape[1]

    return Prob


if __name__ == '__main__':
    # Sample detailed predictions as a list of lists (numerical values)
    DetailedPredictions = [
        [1, 2, 2, 1, 2],
        [2, 2, 2, 2, 1],
        [1, 1, 1, 2, 2]
    ]

    Prob = CalculateProbabilities(DetailedPredictions)
    print("Probabilities:")
    print(Prob)
