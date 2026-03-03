import numpy as np
import pandas as pd
from pathlib import Path


NEOM_ZONES = [
    "the_line_north", "the_line_central", "the_line_south",
    "oxagon_port", "oxagon_research", "trojena_resort",
    "trojena_village", "sindalah_marina", "leyja_nature",
    "neom_bay", "epicon_coastal", "xaynor_logistics",
]

TRANSPORT_MODES = {
    "autonomous_pod": {
        "avg_speed_kmh": 45,
        "capacity": 6,
        "carbon_g_per_km": 0,
        "cost_per_km": 1.2,
        "comfort_score": 4.2,
        "availability_hours": (0, 24),
    },
    "high_speed_rail": {
        "avg_speed_kmh": 250,
        "capacity": 400,
        "carbon_g_per_km": 0,
        "cost_per_km": 0.8,
        "comfort_score": 4.5,
        "availability_hours": (5, 23),
    },
    "hyperloop": {
        "avg_speed_kmh": 700,
        "capacity": 28,
        "carbon_g_per_km": 0,
        "cost_per_km": 2.5,
        "comfort_score": 4.0,
        "availability_hours": (6, 22),
    },
    "walking": {
        "avg_speed_kmh": 5,
        "capacity": 1,
        "carbon_g_per_km": 0,
        "cost_per_km": 0.0,
        "comfort_score": 3.0,
        "availability_hours": (0, 24),
    },
    "cycling": {
        "avg_speed_kmh": 18,
        "capacity": 1,
        "carbon_g_per_km": 0,
        "cost_per_km": 0.1,
        "comfort_score": 3.2,
        "availability_hours": (0, 24),
    },
    "e_scooter": {
        "avg_speed_kmh": 25,
        "capacity": 1,
        "carbon_g_per_km": 0,
        "cost_per_km": 0.5,
        "comfort_score": 3.5,
        "availability_hours": (5, 23),
    },
    "vertical_transit": {
        "avg_speed_kmh": 15,
        "capacity": 20,
        "carbon_g_per_km": 0,
        "cost_per_km": 0.3,
        "comfort_score": 4.0,
        "availability_hours": (5, 24),
    },
    "water_taxi": {
        "avg_speed_kmh": 35,
        "capacity": 12,
        "carbon_g_per_km": 5,
        "cost_per_km": 3.0,
        "comfort_score": 4.3,
        "availability_hours": (6, 21),
    },
}

LINE_ZONES = ["the_line_north", "the_line_central", "the_line_south"]
COASTAL_ZONES = ["sindalah_marina", "neom_bay", "epicon_coastal", "oxagon_port"]


def _compute_zone_distance(origin, dest):
    """Compute approximate distance between two NEOM zones.

    Args:
        origin: Origin zone name.
        dest: Destination zone name.

    Returns:
        Distance in kilometers.
    """
    zone_coords = {
        "the_line_north": (28.0, 35.0),
        "the_line_central": (27.5, 35.5),
        "the_line_south": (27.0, 36.0),
        "oxagon_port": (27.8, 35.2),
        "oxagon_research": (27.7, 35.3),
        "trojena_resort": (28.2, 36.0),
        "trojena_village": (28.1, 35.9),
        "sindalah_marina": (27.9, 34.8),
        "leyja_nature": (27.6, 35.8),
        "neom_bay": (28.0, 34.9),
        "epicon_coastal": (27.3, 35.1),
        "xaynor_logistics": (27.4, 35.6),
    }
    lat1, lon1 = zone_coords[origin]
    lat2, lon2 = zone_coords[dest]
    dlat = abs(lat1 - lat2) * 111
    dlon = abs(lon1 - lon2) * 85
    return round(np.sqrt(dlat ** 2 + dlon ** 2), 1)


def generate_residents(n_residents=10000, seed=42):
    """Generate synthetic NEOM resident profiles.

    Args:
        n_residents: Number of residents to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with resident profiles.
    """
    rng = np.random.RandomState(seed)

    residents = pd.DataFrame({
        "resident_id": [f"R{str(i).zfill(5)}" for i in range(n_residents)],
        "age": rng.randint(18, 75, n_residents),
        "fitness_level": rng.randint(1, 6, n_residents),
        "sustainability_priority": rng.randint(1, 6, n_residents),
        "comfort_preference": rng.randint(1, 6, n_residents),
        "budget_sensitivity": rng.randint(1, 6, n_residents),
        "has_mobility_limitation": rng.random(n_residents) < 0.08,
        "home_zone": rng.choice(NEOM_ZONES, n_residents),
        "work_zone": rng.choice(NEOM_ZONES, n_residents),
    })

    print(f"Generated {len(residents)} resident profiles")
    return residents


def _get_available_modes(distance, temperature, origin, dest, hour):
    """Determine which transport modes are available given trip context.

    Args:
        distance: Trip distance in km.
        temperature: Current temperature in Celsius.
        origin: Origin zone name.
        dest: Destination zone name.
        hour: Hour of day (0-23).

    Returns:
        List of available mode names.
    """
    available = []

    for mode_name, props in TRANSPORT_MODES.items():
        start_h, end_h = props["availability_hours"]
        if not (start_h <= hour < end_h):
            continue

        if mode_name == "hyperloop" and distance < 50:
            continue

        if mode_name == "cycling" and temperature > 45:
            continue

        if mode_name == "vertical_transit":
            if origin not in LINE_ZONES and dest not in LINE_ZONES:
                continue

        if mode_name == "water_taxi":
            if origin not in COASTAL_ZONES and dest not in COASTAL_ZONES:
                continue

        if mode_name == "walking" and distance > 5:
            continue

        if mode_name == "cycling" and distance > 20:
            continue

        if mode_name == "e_scooter" and distance > 15:
            continue

        available.append(mode_name)

    if not available:
        available = ["autonomous_pod"]

    return available


def _choose_mode(resident, available_modes, distance, rng):
    """Choose a transport mode based on resident preferences.

    Args:
        resident: Series with resident profile data.
        available_modes: List of available mode names.
        distance: Trip distance in km.
        rng: Numpy random state.

    Returns:
        Chosen mode name.
    """
    scores = {}

    for mode in available_modes:
        props = TRANSPORT_MODES[mode]
        score = 1.0

        cost = props["cost_per_km"] * distance
        if resident["budget_sensitivity"] >= 4:
            score *= max(0.1, 1.0 - cost / 50.0)

        if resident["comfort_preference"] >= 4:
            score *= props["comfort_score"] / 5.0

        if resident["sustainability_priority"] >= 4:
            score *= max(0.5, 1.0 - props["carbon_g_per_km"] / 20.0)

        if resident["fitness_level"] >= 4 and mode in ["walking", "cycling"]:
            score *= 1.8

        if resident["has_mobility_limitation"] and mode in ["walking", "cycling", "e_scooter"]:
            score *= 0.05

        if resident["age"] > 60 and mode in ["e_scooter", "cycling"]:
            score *= 0.3

        speed = props["avg_speed_kmh"]
        if distance > 30 and speed < 30:
            score *= 0.1

        scores[mode] = max(score, 0.01)

    total = sum(scores.values())
    probs = [scores[m] / total for m in available_modes]
    chosen = rng.choice(available_modes, p=probs)
    return chosen


def _compute_satisfaction(resident, mode, distance, temperature):
    """Compute satisfaction rating for a trip.

    Args:
        resident: Series with resident profile data.
        mode: Chosen transport mode name.
        distance: Trip distance in km.
        temperature: Temperature in Celsius.

    Returns:
        Satisfaction rating (1-5).
    """
    props = TRANSPORT_MODES[mode]
    base = props["comfort_score"]

    if resident["sustainability_priority"] >= 4 and props["carbon_g_per_km"] == 0:
        base += 0.3

    if resident["budget_sensitivity"] >= 4 and props["cost_per_km"] > 2.0:
        base -= 0.5

    if temperature > 40 and mode in ["walking", "cycling"]:
        base -= 1.0

    travel_time = (distance / props["avg_speed_kmh"]) * 60
    if travel_time > 60:
        base -= 0.4

    base += np.random.normal(0, 0.3)
    return int(np.clip(round(base), 1, 5))


def generate_trip_records(residents, n_trips=500000, seed=42):
    """Generate synthetic historical trip records for NEOM residents.

    Mode choices are influenced by distance, time of day, weather,
    resident preferences, and historical usage patterns.

    Args:
        residents: DataFrame with resident profiles.
        n_trips: Number of trip records to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with trip records.
    """
    rng = np.random.RandomState(seed)

    resident_ids = rng.choice(residents["resident_id"].values, n_trips)
    start_date = pd.Timestamp("2025-06-01")
    timestamps = [
        start_date + pd.Timedelta(hours=int(h))
        for h in rng.uniform(0, 180 * 24, n_trips)
    ]

    records = []
    print("Generating trip records...")

    for i in range(n_trips):
        if (i + 1) % 100000 == 0:
            print(f"  {i + 1:,} / {n_trips:,} trips generated")

        resident = residents[residents["resident_id"] == resident_ids[i]].iloc[0]
        ts = timestamps[i]
        hour = ts.hour

        if rng.random() < 0.6:
            origin = resident["home_zone"]
            dest = resident["work_zone"]
        else:
            origin = rng.choice(NEOM_ZONES)
            dest = rng.choice([z for z in NEOM_ZONES if z != origin])

        distance = _compute_zone_distance(origin, dest)
        if distance < 0.5:
            distance = round(rng.uniform(0.5, 3.0), 1)

        month = ts.month
        if month in [6, 7, 8]:
            temperature = rng.uniform(35, 50)
        elif month in [12, 1, 2]:
            temperature = rng.uniform(12, 25)
        else:
            temperature = rng.uniform(22, 38)

        available = _get_available_modes(distance, temperature, origin, dest, hour)
        mode = _choose_mode(resident, available, distance, rng)

        speed = TRANSPORT_MODES[mode]["avg_speed_kmh"]
        trip_time = round((distance / speed) * 60, 1)

        satisfaction = _compute_satisfaction(resident, mode, distance, temperature)

        records.append({
            "resident_id": resident_ids[i],
            "origin_zone": origin,
            "dest_zone": dest,
            "mode_chosen": mode,
            "trip_distance_km": distance,
            "trip_time_min": trip_time,
            "timestamp": ts,
            "satisfaction_rating": satisfaction,
            "temperature": round(temperature, 1),
        })

    trips = pd.DataFrame(records)
    print(f"Generated {len(trips):,} trip records across {len(TRANSPORT_MODES)} modes")

    mode_dist = trips["mode_chosen"].value_counts()
    for mode, count in mode_dist.items():
        print(f"  {mode}: {count:,} trips ({count / len(trips):.1%})")

    return trips


def save_datasets(residents, trips, data_dir="data"):
    """Save generated datasets to CSV files.

    Args:
        residents: DataFrame with resident profiles.
        trips: DataFrame with trip records.
        data_dir: Directory to save files.
    """
    output = Path(data_dir)
    output.mkdir(parents=True, exist_ok=True)

    residents_path = output / "residents.csv"
    trips_path = output / "trips.csv"

    residents.to_csv(residents_path, index=False)
    trips.to_csv(trips_path, index=False)

    print(f"Residents saved to {residents_path}")
    print(f"Trips saved to {trips_path}")


def get_mode_properties():
    """Return transport mode properties as a DataFrame.

    Returns:
        DataFrame with mode properties.
    """
    rows = []
    for name, props in TRANSPORT_MODES.items():
        row = {"mode": name}
        row.update(props)
        row["availability_start"] = props["availability_hours"][0]
        row["availability_end"] = props["availability_hours"][1]
        del row["availability_hours"]
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    residents = generate_residents()
    trips = generate_trip_records(residents)
    save_datasets(residents, trips)
