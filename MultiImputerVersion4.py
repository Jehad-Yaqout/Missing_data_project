import numpy as np
from find_best_featureset_PSO_cheap3 import find_best_featureset_PSO_cheap3


def compute_imputation_error(selected_features, OriginalData, ImputedData):
    selected_indices = np.where(selected_features == 1)[0]
    if len(selected_indices) == 0:
        return float('inf')
    imputed_values = impute_with_selected_features(selected_indices, OriginalData, ImputedData)
    error = np.sum((imputed_values - ImputedData) ** 2)
    return error


def impute_with_selected_features(selected_indices, OriginalData, ImputedData):
    imputed_data = ImputedData.copy()
    for i in range(ImputedData.shape[0]):
        for j in selected_indices:
            if np.isnan(ImputedData[i, j]):
                imputed_data[i, j] = np.nanmean(OriginalData[:, j])
    return imputed_data


def MultiImputerVersion4(New_X_unlabeled, NewImputedData, m_imputations):
    num_features = 3
    result = find_best_featureset_PSO_cheap3(New_X_unlabeled, NewImputedData, num_features)
    if result is None or len(result) == 0:
        print("Error: find_best_featureset_PSO_cheap3 did not return a valid feature set.")
        return [], NewImputedData, np.zeros(New_X_unlabeled.shape[0])
    GlobalBest = result
    List_unique_best_solns_Positions = np.array([GlobalBest])
    List_unique_best_solns_Costs = np.array([compute_imputation_error(GlobalBest, New_X_unlabeled, NewImputedData)])
    return GlobalBest, List_unique_best_solns_Positions, List_unique_best_solns_Costs
