
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge

from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone


# 1. Load datasets

train = pd.read_csv("data/processed/bmw_train.csv")
validation = pd.read_csv("data/processed/bmw_validation.csv")


# 2. Separate features and target

features = [
    "model",
    "year",
    "mileage",
    "fuelType",
    "transmission",
    "engineSize"
]

X_train = train[features]
y_train = train["price"]

X_val = validation[features]
y_val = validation["price"]


# 3. Train baseline model

baseline = DummyRegressor(strategy="median")

baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_val)

baseline_mae = mean_absolute_error(
    y_val, baseline_predictions
)

print("Baseline MAE:", baseline_mae)


# 4. Define preprocessing

numeric_features = ["year", "mileage", "engineSize"]
categorical_features = ["model", "fuelType", "transmission"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# 5. Create Ridge regression pipeline

ridge_model = Pipeline([
    ("preprocessing", preprocessor),
    ("regression", Ridge(alpha=1.0))
])


# 6. Train Ridge model

ridge_model.fit(X_train, y_train)


# 7. Evaluate Ridge model

ridge_predictions = ridge_model.predict(X_val)

ridge_mae = mean_absolute_error(
    y_val, ridge_predictions
)

print("Ridge MAE:", ridge_mae)


# 8. Compare results

improvement = (
    (baseline_mae - ridge_mae) / baseline_mae
) * 100

print(f"Improvement over baseline: {improvement:.2f}%")


# 9. Save comparison results

results = pd.DataFrame({
    "model": ["Median baseline", "Ridge regression"],
    "validation_mae_gbp": [baseline_mae, ridge_mae]
})

Path("reports").mkdir(exist_ok=True)

results.to_csv("reports/model_comparison.csv", index=False)

print("Model comparison saved!")



# Compare training and validation errors
ridge_train_predictions = ridge_model.predict(X_train)

ridge_train_mae = mean_absolute_error(
    y_train, ridge_train_predictions
)

print("Ridge training MAE:", ridge_train_mae)
print("Ridge validation MAE:", ridge_mae)


# Find the 10 largest validation errors
errors = validation[
    ["model", "year", "mileage", "price"]
].copy()

errors["predicted_price"] = ridge_predictions

errors["absolute_error"] = (
    errors["price"] - errors["predicted_price"]
).abs()

print("\n10 largest prediction errors:")
print(
    errors.nlargest(10, "absolute_error")
    .round(0)
    .to_string(index=False)
)





# Random Forest model
forest_model = Pipeline([
    ("preprocessing", clone(preprocessor)),
    ("regression", RandomForestRegressor(
        n_estimators=150,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

forest_model.fit(X_train, y_train)

forest_train_predictions = forest_model.predict(X_train)
forest_val_predictions = forest_model.predict(X_val)

forest_train_mae = mean_absolute_error(
    y_train, forest_train_predictions
)

forest_val_mae = mean_absolute_error(
    y_val, forest_val_predictions
)

print("Forest training MAE:", forest_train_mae)
print("Forest validation MAE:", forest_val_mae)




# for leaf_size in [1, 2, 5, 10, 20]:
#     model = Pipeline([
#         ("preprocessing", clone(preprocessor)),
#         ("regression", RandomForestRegressor(
#             n_estimators=150,
#             min_samples_leaf=leaf_size,
#             random_state=42,
#             n_jobs=-1
#         ))
#     ])

#     model.fit(X_train, y_train)

#     train_predictions = model.predict(X_train)
#     val_predictions = model.predict(X_val)

#     train_mae = mean_absolute_error(
#         y_train, train_predictions
#     )

#     val_mae = mean_absolute_error(
#         y_val, val_predictions
#     )

#     print(
#         f"Leaf size: {leaf_size:2d} | "
#         f"Train MAE: £{train_mae:.2f} | "
#         f"Validation MAE: £{val_mae:.2f}"
#     )


# Find the 10 largest validation errors
errors = validation[
    ["model", "year", "mileage", "price"]
].copy()

errors["predicted_price"] = forest_val_predictions

errors["absolute_error"] = (
    errors["price"] - errors["predicted_price"]
).abs()

print("\n10 largest prediction errors:")
print(
    errors.nlargest(10, "absolute_error")
    .round(0)
    .to_string(index=False)
)



errors = validation[["year", "price"]].copy()

errors["predicted_price"] = forest_val_predictions

errors["absolute_error"] = (
    errors["price"] - errors["predicted_price"]
).abs()

year_errors = errors.groupby("year").agg(
    car_count=("absolute_error", "size"),
    mae=("absolute_error", "mean")
)

print(year_errors.tail(8).round(2))

print(
    "2020 training cars:",
    (train["year"] == 2020).sum()
)

print(
    "2020 validation cars:",
    (validation["year"] == 2020).sum()
)
