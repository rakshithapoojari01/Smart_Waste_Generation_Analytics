import pandas as pd
import numpy as np
from pathlib import Path

# Project location
project = Path(r"E:\Smart_Waste_Generation_Analytics")
dataset_folder = project / "dataset"
dataset_folder.mkdir(parents=True, exist_ok=True)

# 25 Hubli-Dharwad locations
areas = [
    "Vidyanagar", "Gokul Road", "Keshwapur", "Unkal", "Old Hubli",
    "Deshpande Nagar", "Navanagar", "Rajendra Nagar", "Channamma Circle",
    "CBT", "Hosur", "Shirur Park",
    "Saptapur", "Malmaddi", "Saraswatpur", "Vidyagiri",
    "Gandhinagar", "Jubilee Circle", "KCD Road", "Toll Naka",
    "Lakamanahalli", "Navalur", "Kelageri", "UAS Dharwad", "Dharwad Town"
]

# 10 waste types
waste_types = [
    "Organic",
    "Plastic",
    "Paper",
    "Glass",
    "Metal",
    "E-Waste",
    "Textile",
    "Wood",
    "Construction Waste",
    "Hazardous Waste"
]

# One full year
dates = pd.date_range(
    start="2026-01-01",
    end="2026-12-31",
    freq="D"
)

rng = np.random.default_rng(42)

# Population and collection vehicles for each area
population = {
    area: int(rng.integers(18000, 70001))
    for area in areas
}

vehicles = {
    area: int(rng.integers(1, 6))
    for area in areas
}

# Average waste generated per day
base_waste = {
    "Organic": 45,
    "Plastic": 18,
    "Paper": 13,
    "Glass": 7,
    "Metal": 6,
    "E-Waste": 3,
    "Textile": 5,
    "Wood": 4,
    "Construction Waste": 10,
    "Hazardous Waste": 2
}

records = []

for date in dates:

    day = date.dayofyear

    # Simulated weather
    temperature = (
        26
        + 7 * np.sin(2 * np.pi * (day - 80) / 365)
        + rng.normal(0, 2)
    )

    rainfall = max(
        0,
        7 + 6 * np.sin(2 * np.pi * (day - 160) / 365)
        + rng.normal(0, 6)
    )

    for area in areas:

        area_population = population[area]
        area_vehicles = vehicles[area]

        population_factor = area_population / 40000
        vehicle_factor = 0.90 + (0.04 * area_vehicles)

        for waste_type in waste_types:

            seasonal_factor = (
                1 + 0.08 * np.sin(2 * np.pi * day / 365)
            )

            weather_factor = 1

            if rainfall > 20 and waste_type == "Organic":
                weather_factor = 1.03

            waste = (
                base_waste[waste_type]
                * population_factor
                * vehicle_factor
                * seasonal_factor
                * weather_factor
            )

            # Random variation
            waste += rng.normal(0, max(0.5, waste * 0.12))

            waste = max(0.5, round(waste, 2))

            records.append([
                date.strftime("%Y-%m-%d"),
                area,
                waste_type,
                waste,
                area_population,
                area_vehicles,
                round(float(temperature), 1),
                round(float(rainfall), 1)
            ])

# Create DataFrame
df = pd.DataFrame(records, columns=[
    "Date",
    "Area",
    "Waste_Type",
    "Waste_Collected_kg",
    "Population",
    "Collection_Vehicles",
    "Temperature_C",
    "Rainfall_mm"
])

# Save dataset
output_file = dataset_folder / "waste_data.csv"
df.to_csv(output_file, index=False)

print()
print("=" * 60)
print("SMART WASTE GENERATION DATASET CREATED")
print("=" * 60)
print(f"File: {output_file}")
print(f"Records: {len(df):,}")
print(f"Locations: {df['Area'].nunique()}")
print(f"Waste Types: {df['Waste_Type'].nunique()}")
print(f"Date Range: {df['Date'].min()} to {df['Date'].max()}")
print("=" * 60)
print()
print("Waste Types:")
print(df["Waste_Type"].value_counts().sort_index())