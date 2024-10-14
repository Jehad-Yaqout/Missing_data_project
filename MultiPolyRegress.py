import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def PredictHelper(TestData, reg):
    """
    Predict using the polynomial regression model.

    Parameters:
    TestData (np.array): Test data.
    reg (dict): Regression model details.

    Returns:
    np.array: Predicted values.
    """
    PowerMatrix = reg['PowerMatrix']
    Coefficients = reg['Coefficients']

    # Generate polynomial features manually based on PowerMatrix
    num_samples = TestData.shape[0]
    num_terms = PowerMatrix.shape[0]
    test_features = np.ones((num_samples, num_terms))

    for i in range(num_terms):
        for j in range(TestData.shape[1]):
            test_features[:, i] *= TestData[:, j] ** PowerMatrix[i, j]

    yhat = np.dot(test_features, Coefficients)
    return yhat


def MultiPolyRegress(Data, R, PW, *args):
    """
    Perform multi-variable polynomial regression analysis.

    Parameters:
    Data (np.array): m-by-n matrix where m is the number of data points and
                     n is the number of independent variables.
    R (np.array): m-by-1 response column vector.
    PW (int): Degree of the polynomial fit.
    *args: Additional arguments:
           - PV (list): Restricts individual dimensions of Data to particular powers.
           - 'figure': Adds a scatter plot for the fit.
           - 'range': Adjusts the normalization of goodness of fit measures.

    Returns:
    dict: Result containing regression details including:
          - FitParameters
          - PowerMatrix
          - Scores
          - PolynomialExpression
          - Coefficients
          - yhat
          - Residuals
          - GoodnessOfFit
          - RSquare
          - MAE
          - MAESTD
          - LOOCVGoodnessOfFit
          - CVRSquare
          - CVMAE
          - CVMAESTD
    """
    # Align Data
    if Data.shape[1] > Data.shape[0]:
        Data = Data.T

    # Arrange Input Arguments
    PV = np.repeat(PW, Data.shape[1])
    FigureSwitch = 'figureoff'
    NormalizationSwitch = '1-to-1 (Default)'

    for arg in args:
        if arg == 'figure':
            FigureSwitch = 'figureon'
        if arg == 'range':
            NormalizationSwitch = 'Range'
        if isinstance(arg, (list, np.ndarray)):
            PV = np.array(arg)

    # Function Parameters
    NData, NVars = Data.shape

    # Initialize Polynomial Features
    poly = PolynomialFeatures(PW, include_bias=False)
    Scores = poly.fit_transform(Data)

    # Restrict polynomial features based on PV
    powers = poly.powers_
    mask = np.all(powers <= PV, axis=1)
    Scores = Scores[:, mask]
    PowerMatrix = powers[mask]

    # Create a legend
    Legend = []
    for power in PowerMatrix:
        term = []
        for var_idx, exponent in enumerate(power):
            if exponent == 0:
                continue
            elif exponent == 1:
                term.append(f"x{var_idx + 1}")
            else:
                term.append(f"x{var_idx + 1}^{exponent}")
        Legend.append(" * ".join(term) if term else "1")

    # Ordinary Least Squares Regression
    model = LinearRegression(fit_intercept=False)
    model.fit(Scores, R)
    yhat = model.predict(Scores)
    Residuals = R - yhat

    # Polynomial Expression
    Coefficients = model.coef_

    PolyStr = " + ".join(f"{coef}*({term})" for coef, term in zip(Coefficients, Legend))
    variablesexp = " and ".join(f"x{i + 1}" for i in range(Data.shape[1]))

    PolynomialExpression = f"lambda {variablesexp}: {PolyStr}"

    # Goodness of Fit
    SS_Residual = np.sum(Residuals ** 2)
    SS_Total = np.sum((R - np.mean(R)) ** 2)
    RSquare = 1 - (float(SS_Residual) / SS_Total)

    if NormalizationSwitch == 'Range':
        normalized_factor = np.ptp(R)
    else:
        normalized_factor = R

    MAE = np.mean(np.abs(Residuals) / normalized_factor)
    MAESTD = np.std(np.abs(Residuals) / normalized_factor)

    # Leave-One-Out Cross-Validation (LOOCV)
    loo_errors = []
    for i in range(NData):
        Scores_loo = np.concatenate((Scores[:i], Scores[i + 1:]), axis=0)
        R_loo = np.concatenate((R[:i], R[i + 1:]), axis=0)
        model.fit(Scores_loo, R_loo)
        y_pred_loo = model.predict(Scores[i].reshape(1, -1))
        loo_errors.append(R[i] - y_pred_loo[0])

    loo_errors = np.array(loo_errors)
    SSE_CV = np.sum(loo_errors ** 2)
    CVRSquare = 1 - (float(SSE_CV) / SS_Total)

    CVMAE = np.mean(np.abs(loo_errors) / normalized_factor)
    CVMAESTD = np.std(np.abs(loo_errors) / normalized_factor)

    # Construct Output
    reg = {
        'FitParameters': '-----------------',
        'PowerMatrix': PowerMatrix,
        'Scores': Scores,
        'PolynomialExpression': poly,
        'Coefficients': Coefficients,
        'Legend': Legend,
        'yhat': yhat,
        'Residuals': Residuals,
        'GoodnessOfFit': '-----------------',
        'RSquare': RSquare,
        'MAE': MAE,
        'MAESTD': MAESTD,
        'Normalization': NormalizationSwitch,
        'LOOCVGoodnessOfFit': '-----------------',
        'CVRSquare': CVRSquare,
        'CVMAE': CVMAE,
        'CVMAESTD': CVMAESTD,
        'CVNormalization': NormalizationSwitch
    }

    # Optional Figure
    if FigureSwitch == 'figureon':
        plt.figure()
        plt.scatter(yhat, R, c='r', marker='o')
        plt.xlabel('yhat')
        plt.ylabel('y')
        plt.title('Goodness of Fit Scatter Plot')
        xlim = plt.gca().get_xlim()
        ylim = plt.gca().get_ylim()
        lims = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]
        plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        plt.grid(True)
        plt.show()

    return reg


# Example usage
if __name__ == '__main__':
    np.random.seed(0)
    Data = np.random.rand(100, 2)
    R = np.random.rand(100)
    PW = 2
    reg = MultiPolyRegress(Data, R, PW, 'figure')
    print(reg)
