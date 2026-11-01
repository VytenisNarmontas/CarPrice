
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

cars = pd.read_csv("data/processed/bmw_train.csv")

encoder = OneHotEncoder(sparse_output=False)

fuel = cars[["fuelType"]]

encoded = encoder.fit_transform(fuel)

print("Original shape:", fuel.shape)
print("Encoded shape:", encoded.shape)
print("Categories:", encoder.categories_)
print("First 5 encoded rows:")
print(encoded[:5])
