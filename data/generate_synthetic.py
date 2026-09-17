import pandas as pd
import numpy as np
from pathlib import Path

rng = np.random.default_rng(42)

SUBURBS = {
    "Surry Hills": {
        "postcode": 2010,
        "base_price": 1_250_000,
        "price_std": 350_000,
        "distance_cbd": (2.5, 4.5),
        "prop_types": ["Unit", "Townhouse", "House"],
        "type_weights": [0.65, 0.20, 0.15],
        "beds_range": (1, 4),
        "land_size": (None, None),  # units mostly no land
        "floor_range": (55, 140),
        "year_range": (1960, 2022),
        "n": 40,
    },
    "Parramatta": {
        "postcode": 2150,
        "base_price": 900_000,
        "price_std": 230_000,
        "distance_cbd": (23, 27),
        "prop_types": ["House", "Townhouse", "Unit"],
        "type_weights": [0.55, 0.25, 0.20],
        "beds_range": (2, 5),
        "land_size": (300, 700),
        "floor_range": (100, 250),
        "year_range": (1970, 2018),
        "n": 40,
    },
    "Penrith": {
        "postcode": 2750,
        "base_price": 680_000,
        "price_std": 160_000,
        "distance_cbd": (50, 58),
        "prop_types": ["House", "Townhouse", "Unit"],
        "type_weights": [0.70, 0.20, 0.10],
        "beds_range": (2, 5),
        "land_size": (400, 900),
        "floor_range": (110, 280),
        "year_range": (1985, 2024),
        "n": 40,
    },
}

AGENT_PHRASES = [
    "Stunning renovation with modern finishes",
    "Original condition, great potential",
    "Freshly painted and move-in ready",
    "Renovated kitchen and bathrooms",
    "Spacious living with natural light",
    "Charming character home",
    "Contemporary design throughout",
    "As-is, investor special",
    "Premium location, walk to transport",
    "Quiet street, close to schools",
    "Rare offering in tightly held street",
    "North-facing aspect, excellent light",
    "Distressed sale, priced to move",
    "Architect designed with quality fixtures",
    "Solid brick construction",
]

rows = []
sale_date_pool = pd.date_range("2023-01-01", "2024-12-31", freq="D")

for suburb, cfg in SUBURBS.items():
    for i in range(cfg["n"]):
        ptype = rng.choice(cfg["prop_types"], p=cfg["type_weights"])
        beds = int(rng.integers(cfg["beds_range"][0], cfg["beds_range"][1] + 1))
        baths = max(1, beds - int(rng.integers(0, 2)))
        car = int(rng.integers(0, 3))

        # Land size (NA for units in Surry Hills)
        if ptype == "Unit" and suburb == "Surry Hills":
            land = np.nan
        elif ptype == "Unit":
            land = round(float(rng.uniform(80, 200)), 0)
        else:
            ls = cfg["land_size"]
            land = round(float(rng.uniform(ls[0], ls[1])), 0) if ls[0] else np.nan

        floor = round(float(rng.uniform(cfg["floor_range"][0], cfg["floor_range"][1])), 0)
        year_built = int(rng.integers(cfg["year_range"][0], cfg["year_range"][1] + 1))
        sale_date = str(rng.choice(sale_date_pool))[:10]
        distance = round(float(rng.uniform(cfg["distance_cbd"][0], cfg["distance_cbd"][1])), 1)

        # Price model
        price = cfg["base_price"]
        price += (beds - 3) * 90_000
        price += (baths - 1) * 40_000
        if not np.isnan(land if isinstance(land, float) else float('nan')):
            price += land * 800
        price += (2024 - year_built) * (-3000)
        price -= (distance - min(cfg["distance_cbd"])) * 12_000
        if ptype == "House":
            price *= 1.10
        elif ptype == "Unit":
            price *= 0.85
        price += float(rng.normal(0, cfg["price_std"] * 0.4))
        price = max(300_000, round(price / 1000) * 1000)

        desc = rng.choice(AGENT_PHRASES)
        url = f"https://www.domain.com.au/example-{suburb.lower().replace(' ', '-')}-{i+1:03d}"

        rows.append({
            "suburb": suburb,
            "postcode": cfg["postcode"],
            "property_type": ptype,
            "bedrooms": beds,
            "bathrooms": baths,
            "car_spaces": car,
            "land_size_m2": land if not (isinstance(land, float) and np.isnan(land)) else np.nan,
            "floor_area_m2": floor,
            "year_built": year_built,
            "sale_date": sale_date,
            "distance_to_cbd_km": distance,
            "agent_description": desc,
            "sale_price_aud": int(price),
            "source_url": url,
        })

df = pd.DataFrame(rows)
out = Path(__file__).parent / "raw_listings.csv"
df.to_csv(out, index=False)
print(f"Saved {len(df)} rows to {out}")
print(df.groupby("suburb").size())
print(df.describe())
