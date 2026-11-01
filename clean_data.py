
import pandas as pd
from pathlib import Path

output_path = Path("data/processed/bmw_clean.csv")

cars = pd.read_csv("data/raw/bmw.csv")

original_count = len(cars)

# Normalize BMW model names
cars["model"] = cars["model"].str.strip()

# Remove exact duplicates
cars = cars.drop_duplicates().copy()

print("Original rows:", original_count)
print("Duplicates removed:", original_count - len(cars))
print("Remaining rows:", len(cars))



bad_price = (
    (cars["model"] == "2 Series") &
    (cars["year"] == 2015) &
    (cars["price"] == 123456)
)

print("Suspicious prices:", bad_price.sum())


bad_year = (
    (cars["model"] == "1 Series") &
    (cars["year"] < 2004)
)

print("Suspicious years:", bad_year.sum())


invalid_rows = bad_price | bad_year

print("Invalid rows:", invalid_rows.sum())

cars = cars.loc[~invalid_rows].copy()


bad_engine = (
    (cars["model"] != "i3") &
    (cars["engineSize"] == 0.0)
)
print("Suspicious engine sizes:", bad_engine.sum())



cars.loc[bad_engine, "engineSize"] = float("nan")

print("Missing engine sizes:", cars["engineSize"].isna().sum())



output_path.parent.mkdir(parents=True, exist_ok=True)

cars.to_csv(output_path, index=False)

print("Cleaned dataset saved!")