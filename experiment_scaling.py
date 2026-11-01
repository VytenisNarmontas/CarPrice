
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

train = pd.read_csv("data/processed/bmw_train.csv")
validation = pd.read_csv("data/processed/bmw_validation.csv")

numeric_features = ["year", "mileage", "engineSize"]

X_train = train[numeric_features]
X_val = validation[numeric_features]

# First, fill missing values
imputer = SimpleImputer(strategy="median")

train_imputed = imputer.fit_transform(X_train)
val_imputed = imputer.transform(X_val)

# Then, scale numeric values
scaler = StandardScaler()

train_scaled = scaler.fit_transform(train_imputed)
val_scaled = scaler.transform(val_imputed)

print("Original values:")
print(X_train.head())

print("\nScaled values:")
print(train_scaled[:5])

print("\nTraining means:", train_scaled.mean(axis=0))
print("Training std:", train_scaled.std(axis=0))
