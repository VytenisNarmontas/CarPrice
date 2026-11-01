
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

train = pd.read_csv("data/processed/bmw_train.csv")
validation = pd.read_csv("data/processed/bmw_validation.csv")

engine_train = train[["engineSize"]]
engine_val = validation[["engineSize"]]

print("Missing before:", engine_train.isna().sum())

imputer = SimpleImputer(strategy="median")

imputer.fit(engine_train)

print("Learned median:", imputer.statistics_)

engine_train_clean = imputer.transform(engine_train)
engine_val_clean = imputer.transform(engine_val)

print("First 10 transformed values:")
print(engine_train_clean[:10])


print("Missing after:", np.isnan(engine_train_clean).sum())
print("Validation missing after:", np.isnan(engine_val_clean).sum())