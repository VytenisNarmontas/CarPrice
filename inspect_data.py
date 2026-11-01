import pandas as pd

cars = pd.read_csv("data/raw/bmw.csv")


print(cars[cars["year"] < 2000].sort_values("price", ascending=False).head(100))

#print(cars.head())
#print(cars.shape)
#print(cars.columns)

#print(cars["model"].value_counts())
#print(cars["engineSize"].value_counts().sort_index())

#print(cars[cars["engineSize"] == 0.0])
#print(cars[cars["engineSize"] == 0.6])


#print(cars.isna().sum())
#print("Duplicate rows:", cars.duplicated().sum())



# # Check missing values
# print("Missing values:")
# print(cars.isna().sum())

# # Check duplicates
# print("\nDuplicate rows:")
# print(cars.duplicated().sum())

# # Check numeric ranges
# print("\nNumeric statistics:")
# print(cars[["year", "price", "mileage", "engineSize"]].describe())

# # Check fuel type distribution
# print("\nFuel types:")
# print(cars["fuelType"].value_counts())

# # Check model name formatting
# print("\nModel names:")
# print(cars["model"].head().map(repr).tolist())




# print("\n10 most expensive cars:")
# print(cars.nlargest(10, "price"))

# print("\n10 cheapest cars:")
# print(cars.nsmallest(10, "price"))

# print("\n10 oldest cars:")
# print(cars.nsmallest(10, "year"))

# print("\nCars registered before 2004:")
# print(cars[cars["year"] < 2004])




# # Remove leading/trailing spaces from model names
# cars["model"] = cars["model"].str.strip()

# # Find cars with similar specifications
# similar = cars[
#     (cars["model"] == "2 Series") &
#     (cars["year"] == 2015) &
#     (cars["fuelType"] == "Diesel") &
#     (cars["engineSize"] == 2.0)
# ]

# print("Number of similar cars:", len(similar))

# print("\nPrice statistics:")
# print(similar["price"].describe())

# print("\n10 most expensive comparable cars:")
# print(similar.nlargest(10, "price")[
#     ["price", "mileage", "transmission"]
# ])




