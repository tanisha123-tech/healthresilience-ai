from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
OUT = ROOT / "data" / "phc_data.csv"

rng = np.random.default_rng(42)

# ---------------------------------------------------------
# PAN-BRICS SIMULATED PHC LOCATIONS
# ---------------------------------------------------------

states = {
    "India - Jammu & Kashmir": (33.78, 76.58),
    "India - Himachal Pradesh": (31.10, 77.17),
    "India - Punjab": (30.90, 75.85),
    "India - Haryana": (29.06, 76.08),
    "India - Rajasthan": (26.91, 75.79),
    "India - Uttar Pradesh": (26.85, 80.95),
    "India - Bihar": (25.60, 85.13),
    "India - West Bengal": (22.57, 88.36),
    "India - Gujarat": (23.02, 72.57),
    "India - Madhya Pradesh": (23.25, 77.41),
    "India - Maharashtra": (19.08, 72.88),
    "India - Goa": (15.49, 73.83),
    "India - Karnataka": (12.97, 77.59),
    "India - Kerala": (8.52, 76.94),
    "India - Tamil Nadu": (13.08, 80.27),
    "India - Telangana": (17.39, 78.49),
    "India - Andhra Pradesh": (16.51, 80.64),
    "India - Odisha": (20.30, 85.82),
    "India - Jharkhand": (23.36, 85.33),
    "India - Chhattisgarh": (21.25, 81.63),
    "India - Assam": (26.14, 91.74),
    "India - Uttarakhand": (30.32, 78.03),
    "India - Sikkim": (27.33, 88.61),
    "India - Tripura": (23.83, 91.28),
    "Brazil - Sao Paulo": (-23.55, -46.63),
    "Brazil - Rio de Janeiro": (-22.90, -43.17),
    "Brazil - Brasilia": (-15.79, -47.88),
    "Russia - Moscow": (55.75, 37.61),
    "Russia - Saint Petersburg": (59.93, 30.31),
    "Russia - Novosibirsk": (55.00, 82.93),
    "China - Beijing": (39.90, 116.40),
    "China - Shanghai": (31.23, 121.47),
    "China - Shenzhen": (22.54, 114.05),
    "South Africa - Johannesburg": (-26.20, 28.04),
    "South Africa - Cape Town": (-33.92, 18.42),
    "South Africa - Durban": (-29.85, 31.02),
}

medicines = [
    "ORS",
    "Paracetamol",
    "Amoxicillin",
    "Insulin",
    "Azithromycin"
]

# Number of simulated PHCs
num_phcs = 1000

phc_locations = []

state_names = list(states.keys())

for i in range(num_phcs):

    state = rng.choice(state_names)

    base_lat, base_lon = states[state]

    # Small geographical variation around each state centre
    lat = base_lat + rng.normal(0, 0.35)
    lon = base_lon + rng.normal(0, 0.35)

    phc_locations.append({
        "phc_id": f"PHC-{i + 1:04d}",
        "state": state,
        "latitude": lat,
        "longitude": lon
    })


# ---------------------------------------------------------
# CREATE MEDICINE / RESOURCE RECORDS
# ---------------------------------------------------------

records = []

for phc in phc_locations:

    # Each PHC handles several medicines
    selected_medicines = rng.choice(
        medicines,
        size=len(medicines),
        replace=False
    )

    for medicine in selected_medicines:

        patient_footfall = int(
            rng.integers(60, 700)
        )

        daily_consumption = max(
            10,
            patient_footfall * rng.uniform(0.12, 0.75)
        )

        current_stock = max(
            50,
            daily_consumption * rng.uniform(2, 25)
        )

        bed_occupancy = rng.uniform(
            30, 98
        )

        staff_availability = rng.uniform(
            55, 100
        )

        lead_time = int(
            rng.integers(1, 15)
        )

        season_index = rng.uniform(
            0.70, 1.40
        )

        emergency_index = rng.choice(
            [0, 1],
            p=[0.88, 0.12]
        )

        # Future demand
        forecast_7d = (
            daily_consumption
            * 7
            * season_index
            * (1 + 0.55 * emergency_index)
            * (0.75 + patient_footfall / 1000)
        )

        # Estimated stock-out time
        stockout_days = (
            current_stock
            /
            max(
                daily_consumption
                * season_index
                * (1 + 0.55 * emergency_index),
                1
            )
        )

        stockout = int(
            stockout_days < 7
        )

        records.append({

            "phc_id": phc["phc_id"],

            "state": phc["state"],

            "medicine": medicine,

            "current_stock": round(
                current_stock
            ),

            "daily_consumption": round(
                daily_consumption,
                1
            ),

            "patient_footfall":
                patient_footfall,

            "bed_occupancy_pct":
                round(
                    bed_occupancy,
                    1
                ),

            "staff_availability_pct":
                round(
                    staff_availability,
                    1
                ),

            "lead_time_days":
                lead_time,

            "season_index":
                round(
                    season_index,
                    2
                ),

            "emergency_index":
                emergency_index,

            "forecast_7d":
                round(
                    forecast_7d
                ),

            "stockout_days":
                round(
                    stockout_days,
                    1
                ),

            "stockout":
                stockout,

          "latitude":
    round(
        phc["latitude"],
        5
    ),

"longitude":
    round(
        phc["longitude"],
        5
    )
        })


df = pd.DataFrame(records)

OUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUT,
    index=False
)

print(
    f"Created India-wide simulated dataset: "
    f"{len(df)} records across "
    f"{num_phcs} PHCs and "
    f"{len(state_names)} states."
)

print(
    "\nStates included:"
)

for state in state_names:
    print(
        f" - {state}"
    )