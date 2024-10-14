# imputation_utils.py
import numpy as np
from sklearn.impute import SimpleImputer


def PredictMissingValuesTest(X_tst, TrainingImputedData):
    imputer = SimpleImputer(strategy='median')
    imputer.fit(TrainingImputedData)
    TestImputedData = imputer.transform(X_tst)
    return TestImputedData, TestImputedData