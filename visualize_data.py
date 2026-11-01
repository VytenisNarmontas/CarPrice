import pandas as pd
import matplotlib.pyplot as plt

cars = pd.read_csv("data/processed/bmw_clean.csv")

numberOfBins = 100

plt.hist(cars["price"], bins=numberOfBins)

plt.xlabel("Price (GBP)")
plt.ylabel("Number of Cars")
plt.title("BMW asking price distribution")

plt.show()


plt.figure(figsize=(10, 6))

plt.scatter(
    cars["mileage"],
    cars["price"],
    alpha=0.3,
    s=10
)

plt.xlabel("Mileage (miles)")
plt.ylabel("Price (GBP)")
plt.title("BMW mileage vs price")

plt.show()


series3 = cars[cars["model"] == "3 Series"]

plt.figure(figsize=(10, 6))

plt.scatter(
    series3["mileage"],
    series3["price"],
    alpha=0.3,
    s=10
)

plt.xlabel("Mileage (miles)")
plt.ylabel("Price (GBP)")
plt.title("BMW 3 Series mileage vs price")

plt.show()



series3_2018 = cars[
    (cars["model"] == "3 Series") &
    (cars["year"] == 2018)
]

print("Cars found:", len(series3_2018))

plt.figure(figsize=(10, 6))

plt.scatter(
    series3_2018["mileage"],
    series3_2018["price"],
    alpha=0.4,
    s=15
)

plt.xlabel("Mileage (miles)")
plt.ylabel("Price (GBP)")
plt.title("2018 BMW 3 Series: Mileage vs Price")

plt.show()



print("All BMWs:",
      cars["mileage"].corr(cars["price"]))

print("2018 BMW 3 Series:",
      series3_2018["mileage"].corr(series3_2018["price"]))
