import os
import numpy as np
import pandas as pd  # Import pandas for saving data to CSV
from RBF import RBF  # Import the RBF function
from RBF_eval import RBF_eval  # Import the RBF_eval function
from ApplyImputation import ApplyImputation
from FlexibleClassifier import FlexibleClassifier
from CalculateProbabilities import CalculateProbabilities
from LoadData import LoadData
from missing_value_generator import missing_value_generator
from AnalyseMissingData import AnalyseMissingData
from find_best_featureset_PSO_cheap3 import find_best_featureset_PSO_cheap3
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def save_to_csv(file_path, data):
    """Utility function to save data to a CSV file."""
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)


def GenerateDataNew(SDataset, Runs, NInitialPer, idx):
    PlotFlag = 0
    Threshold = 0.2
    Scenario = 3
    Nof = 0
    missing_rate_list = [0, 10, 20, 30, 40, 50]
    m_imputations = 5
    seed_list = range(1, Runs + 1)

    results_folder = "ResultOfGenerateDataNew"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    for r in range(Runs):
        data, Y, _, _, _, _, _, _ = LoadData(PlotFlag)
        if data.shape[1] > 25:
            pca = PCA(n_components=25)
            data = pca.fit_transform(data)
        X_missing = missing_value_generator(seed_list[r - 1], data, missing_rate_list[idx])
        X_unlabeled, X_tst, y_unlabeled, y_tst = train_test_split(X_missing, Y, test_size=0.5,
                                                                  random_state=seed_list[r - 1])
        X_unlabeled_Full, X_tst_Full, _, _ = train_test_split(data, Y, test_size=0.5, random_state=seed_list[r - 1])
        result = AnalyseMissingData(X_unlabeled, X_unlabeled_Full, y_unlabeled)
        New_X_unlabeled, New_X_unlabeled_Full, New_y_unlabeled, Idx_MissingRateCol, Idx_MissingFeatures = result
        IDX = np.isnan(New_X_unlabeled).any(axis=1)

        if missing_rate_list[idx] > 0:
            TrainingImputedData, TestImputedData, Imputed_uncertainity, Imputed_Indices = ApplyImputation(
                New_X_unlabeled, New_X_unlabeled_Full,
                X_tst, X_tst_Full, IDX, Idx_MissingRateCol, m_imputations)
        else:
            TrainingImputedData = New_X_unlabeled
            Imputed_uncertainity = np.zeros((TrainingImputedData.shape[0], 1))
            Imputed_Indices = np.zeros((TrainingImputedData.shape[0], 1))
            TestImputedData = X_tst

        # Utilize the feature selection before reducing dimensionality
        num_features = 10  # Define the number of features to select, adjust as needed
        best_features = find_best_featureset_PSO_cheap3(data, TrainingImputedData, num_features)
        TrainingImputedData = TrainingImputedData[:, best_features]
        TestImputedData = TestImputedData[:, best_features]

        Me = np.mean(TrainingImputedData, axis=0)
        NewTempdata = TrainingImputedData - Me
        cov = np.dot(NewTempdata.T, NewTempdata)
        eigVal, eigVec = np.linalg.eig(cov)
        idxTemp = np.argsort(eigVal)[::-1]
        NewEigVal2 = eigVal / sum(eigVal)
        CS = np.cumsum(NewEigVal2)
        iidx = np.where(CS >= 0.95)[0][0]
        transformation_matrix = eigVec[:, idxTemp[:iidx + 1]]
        Newdata = np.dot(TrainingImputedData, transformation_matrix)
        NewTestdata = np.dot(TestImputedData, transformation_matrix)

        N, dim = Newdata.shape
        NLb = np.min(Newdata, axis=0)
        NUb = np.max(Newdata, axis=0)
        X = Newdata
        UC = np.unique(New_y_unlabeled)
        cn = [sum(New_y_unlabeled == i) for i in UC]
        IR = max(cn) / min(cn)
        Budget = int(np.ceil(0.05 * X.shape[0]))
        NInitialPnts = int(np.ceil((NInitialPer / 100) * X.shape[0]))
        R = np.random.permutation(X.shape[0])

        # Ensure NInitialPnts does not exceed the size of Imputed_uncertainity or R
        NInitialPnts = min(NInitialPnts, Imputed_uncertainity.shape[0], R.shape[0])

        # Ensure that R values do not exceed the bounds of Imputed_uncertainity
        R = R[R < Imputed_uncertainity.shape[0]]

        # Debugging information
        # Removed print statement

        # Ensure that R[:NInitialPnts] does not exceed Imputed_uncertainity size
        if NInitialPnts > 0 and Imputed_uncertainity.shape[0] > 0:
            max_index = min(NInitialPnts, Imputed_uncertainity.shape[0])
            # Removed print statement

            # Select the Imputed_uncertainity elements safely
            Selected_Imputed_uncertainity = Imputed_uncertainity[:max_index].reshape(-1, 1)
        else:
            Selected_Imputed_uncertainity = np.zeros((NInitialPnts, 1))  # Handle empty or 1D case

        Dl = X[R[:NInitialPnts], :]
        LabelsDl = New_y_unlabeled[R[:NInitialPnts]]
        Du = np.delete(X, R[:NInitialPnts], axis=0)
        LabelsDu = np.delete(New_y_unlabeled, R[:NInitialPnts], axis=0)

        Selected_Imputed_Indices = Imputed_Indices[R[:NInitialPnts]] if Imputed_Indices.ndim == 1 else Imputed_Indices[
                                                                                                       R[:NInitialPnts],
                                                                                                       :]

        Imputed_Indices = np.delete(Imputed_Indices, R[:NInitialPnts], axis=0)

        if Imputed_uncertainity.ndim == 1:
            Imputed_uncertainity = np.expand_dims(Imputed_uncertainity, axis=1)

        Imputed_uncertainity = np.delete(Imputed_uncertainity, R[:NInitialPnts], axis=0)

        # W1 must be 1D
        W1 = np.ones(LabelsDl.shape[0])

        try:
            lambda_, gamma = RBF(Dl, LabelsDl, 'cubic')
            # Removed print statement

            # Use the RBF evaluation to compute predictions
            eval_result = RBF_eval(Du, Dl, lambda_, gamma, 'cubic')
        except np.linalg.LinAlgError as err:
            # Removed print statement
            lambda_, gamma, eval_result = None, None, None

        if eval_result is not None:
            # Removed print statement
            pass
        else:
            eval_result = None  # Handling the case when the evaluation fails or is not computed

        if NInitialPnts > 0:
            Accuracy, RLabels, DetailedPredictions = FlexibleClassifier(Dl, LabelsDl, W1, Du, LabelsDu, 100)

            # Ensure DetailedPredictions is all numeric
            DetailedPredictions = [[float(value) if isinstance(value, (int, float, str)) else 0 for value in prediction]
                                   for prediction in DetailedPredictions]

            Prob = CalculateProbabilities(DetailedPredictions)
        else:
            Accuracy = []
            Prob = []

        if NInitialPer == 0:
            FileNameBase = f"{results_folder}/NoInitDataset{SDataset}Iter{r}"
        else:
            FileNameBase = f"{results_folder}/Dataset{SDataset}Iter{r}"

        variables_to_save = {
            'Y': Y,
            'Dl': Dl,
            'LabelsDu': LabelsDu,
            'LabelsDl': LabelsDl,
            'Du': Du,
            'data': data,
            'Data': X,
            'Labels': New_y_unlabeled,
            'NLb': NLb,
            'NUb': NUb,
            "Selected_Imputed_Indices": Selected_Imputed_Indices,
            "Selected_Imputed_uncertainity": Selected_Imputed_uncertainity,
            "Imputed_uncertainity": Imputed_uncertainity,
            'Imputed_Indices': Imputed_Indices,
            'NewTestdata': NewTestdata,
            'y_tst': y_tst,
            'lambda': lambda_,
            'gamma': gamma,
            'RBF_eval_result': eval_result,  # Save the RBF evaluation result
        }

        for var_name, data in variables_to_save.items():
            save_to_csv(f"{FileNameBase}_{var_name}.csv", data)

        if NInitialPnts > 0:
            save_to_csv(f"{FileNameBase}_Accuracy.csv", Accuracy)
            save_to_csv(f"{FileNameBase}_Prob.csv", Prob)


if __name__ == "__main__":
    GenerateDataNew(1, 10, 20, 2)  # Example function call for testing
