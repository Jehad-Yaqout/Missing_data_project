import numpy as np


def AnalyseMissingData(X_unlabeled, X_unlabeled_Full, y_unlabeled):
    # 1- Remove the variable that has more than 50% missing data
    No_missing_Col = np.sum(np.isnan(X_unlabeled), axis=0)
    Rate_missing_Col = No_missing_Col / X_unlabeled.shape[0]
    Idx_MissingRateCol = Rate_missing_Col >= 0.5
    X_unlabeled = X_unlabeled[:, ~Idx_MissingRateCol]
    X_unlabeled_Full = X_unlabeled_Full[:, ~Idx_MissingRateCol]

    # 2- Remove the data point that has more than 50% missing data
    No_missing_Features = np.sum(np.isnan(X_unlabeled), axis=1)
    Idx_MissingFeatures = No_missing_Features >= np.ceil(0.5 * X_unlabeled.shape[1])
    X_unlabeled = X_unlabeled[~Idx_MissingFeatures, :]
    X_unlabeled_Full = X_unlabeled_Full[~Idx_MissingFeatures, :]
    y_unlabeled = y_unlabeled[~Idx_MissingFeatures]

    return X_unlabeled, X_unlabeled_Full, y_unlabeled, Idx_MissingRateCol, Idx_MissingFeatures


# Example usage
if __name__ == "__main__":
    X_unlabeled = np.array([[np.nan, 2, 3], [4, 5, np.nan], [7, 8, 9], [np.nan, np.nan, np.nan]])
    X_unlabeled_Full = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    y_unlabeled = np.array([1, 2, 1, 3])

    result = AnalyseMissingData(X_unlabeled, X_unlabeled_Full, y_unlabeled)

    X_unlabeled, X_unlabeled_Full, y_unlabeled, Idx_MissingRateCol, Idx_MissingFeatures = result

    print("X_unlabeled:\n", X_unlabeled)
    print("X_unlabeled_Full:\n", X_unlabeled_Full)
    print("y_unlabeled:\n", y_unlabeled)
    print("Idx_MissingRateCol:\n", Idx_MissingRateCol)
    print("Idx_MissingFeatures:\n", Idx_MissingFeatures)