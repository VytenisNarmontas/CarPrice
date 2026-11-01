import pandas as pd
from sklearn.model_selection import train_test_split

cars = pd.read_csv("data/processed/bmw_clean.csv")

train, holdout = train_test_split(
    cars,
    test_size=0.30,
    random_state=42
)

print("Training rows:", len(train))
print("Holdout rows:", len(holdout))

validation, test = train_test_split(
    holdout,
    test_size=0.50,
    random_state=42
)

print("Validation rows:", len(validation))
print("Test rows:", len(test))


train.to_csv("data/processed/bmw_train.csv", index=False)
validation.to_csv("data/processed/bmw_validation.csv", index=False)
test.to_csv("data/processed/bmw_test.csv", index=False)