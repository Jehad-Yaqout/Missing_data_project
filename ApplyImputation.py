import numpy as np
from sklearn.impute import SimpleImputer
from MultiImputerVersion4 import MultiImputerVersion4  # Ensure correct import


def ApplyImputation(New_X_unlabeled, New_X_unlabeled_Full, X_tst, X_tst_Full, IDX, Idx_MissingRateCol, m_imputations):
    imputer = SimpleImputer(strategy='median')
    NewImputedData = imputer.fit_transform(New_X_unlabeled)

    # Ensure the imputation process retains the correct shape
    TrainingImputedData = np.copy(NewImputedData)
    if TrainingImputedData.shape != New_X_unlabeled.shape:
        raise ValueError(f"TrainingImputedData shape {TrainingImputedData.shape} "
                         f"does not match New_X_unlabeled shape {New_X_unlabeled.shape}")

    TestImputedData = imputer.transform(X_tst)

    # Validate imputation correctness
    if TestImputedData.shape != X_tst.shape:
        raise ValueError(f"TestImputedData shape {TestImputedData.shape} does not match X_tst shape {X_tst.shape}")

    # Debug prints
    print("Running MultiImputerVersion4...")
    print("New_X_unlabeled shape:", New_X_unlabeled.shape)
    print("NewImputedData shape:", NewImputedData.shape)
    print("m_imputations:", m_imputations)

    # Apply MultiImputerVersion4 to get imputed uncertainty and improved imputed data
    GlobalBest, List_unique_best_solns_Positions, List_unique_best_solns_Costs = MultiImputerVersion4(
        New_X_unlabeled, NewImputedData, m_imputations)

    # Debug prints
    print("GlobalBest:", GlobalBest)
    print("List_unique_best_solns_Positions:", List_unique_best_solns_Positions)
    print("List_unique_best_solns_Costs:", List_unique_best_solns_Costs)

    # Here, you could use List_unique_best_solns_Costs or something relevant to represent uncertainty
    Imputed_uncertainity = List_unique_best_solns_Costs if List_unique_best_solns_Costs.size else np.zeros(
        New_X_unlabeled.shape[0])
    Imputed_Indices = IDX

    return TrainingImputedData, TestImputedData, Imputed_uncertainity, Imputed_Indices
