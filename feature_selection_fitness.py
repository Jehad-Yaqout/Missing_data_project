import numpy as np
from MultiPolyRegress import MultiPolyRegress


def PredictAlaa(TestData, reg):
    """
    Predict using the polynomial regression model.
    
    Parameters:
    TestData (np.array): Test data.
    reg (dict): Regression model details.
    
    Returns:
    np.array: Predicted values.
    """
    polynomial_exp = reg['PolynomialExpression']
    test_scores = polynomial_exp.fit_transform(TestData)
    yhat = np.dot(test_scores, reg['Coefficients'])
    return yhat


def Fitness_fn(Current_col, Remaining_COLS, OriginalData, ImputedData):
    """
    Calculate fitness function for feature selection.

    Parameters:
    Current_col (int): Current column index.
    Remaining_COLS (array): Array of remaining columns.
    OriginalData (np.array): Original data with missing values.
    ImputedData (np.array): Data with imputed values.
    
    Returns:
    float: Mean error over 5 iterations.
    """
    IDx = np.isnan(OriginalData[:, Current_col])
    TempCol = Remaining_COLS

    for k in range(len(TempCol)):
        IDx = IDx | np.isnan(OriginalData[:, TempCol[k]])

    Targets = ImputedData[:, Current_col]
    Data = ImputedData[:, TempCol]
    errors = []

    for _ in range(5):
        RIDX = np.random.permutation(len(Data))
        NewData = Data[RIDX, :]
        NewTargets = Targets[RIDX]

        split_idx = int(np.ceil(0.6 * len(Data)))
        TrainingData = NewData[:split_idx, :]
        TrainingTargets = NewTargets[:split_idx]
        TestData = NewData[split_idx:, :]
        TestTargets = NewTargets[split_idx:]

        reg = MultiPolyRegress(TrainingData, TrainingTargets, 2)
        yhatNew = PredictAlaa(TestData, reg)
        error = np.sum(np.abs(yhatNew - TestTargets))
        errors.append(error)

    return np.mean(errors)