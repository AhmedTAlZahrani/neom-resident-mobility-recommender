import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .data_generator import TRANSPORT_MODES, _compute_zone_distance, _get_available_modes


class TripFeatureBuilder:

    def __init__(self):
        self._scaler = MinMaxScaler()
        self._is_fitted = False

    def build_trip_features(self, origin, dest, hour, temperature):
        """Build feature vectors for all available trip options.

        Args:
            origin: Origin zone name.
            dest: Destination zone name.
            hour: Hour of day (0-23).
            temperature: Current temperature in Celsius.

        Returns:
            DataFrame with one row per available mode, columns are features.
        """
        distance = _compute_zone_distance(origin, dest)
        if distance < 0.5:
            distance = 1.0

        available = _get_available_modes(distance, temperature, origin, dest, hour)
        is_rush_hour = hour in [7, 8, 9, 17, 18, 19]
        day_type = "workday"

        rows = []
        for mode in available:
            props = TRANSPORT_MODES[mode]
            speed = props["avg_speed_kmh"]
            travel_time = round((distance / speed) * 60, 1)

            num_transfers = self._estimate_transfers(mode, distance)
            walking_distance = self._estimate_walking(mode, distance)
            carbon = props["carbon_g_per_km"] * distance
            crowd_level = self._estimate_crowd(mode, hour, is_rush_hour)
            cost = round(props["cost_per_km"] * distance, 2)
            comfort = props["comfort_score"]
            accessibility = self._compute_accessibility(mode)

            rows.append({
                "mode": mode,
                "travel_time_min": travel_time,
                "num_transfers": num_transfers,
                "walking_distance_m": walking_distance,
                "carbon_footprint_g": round(carbon, 1),
                "crowd_level": crowd_level,
                "cost_sar": cost,
                "comfort_score": comfort,
                "accessibility_rating": accessibility,
                "time_of_day": hour,
                "temperature": temperature,
                "is_rush_hour": int(is_rush_hour),
                "day_type": day_type,
                "trip_distance_km": distance,
            })

        return pd.DataFrame(rows)

    def _estimate_transfers(self, mode, distance):
        """Estimate number of transfers needed for a trip.

        Args:
            mode: Transport mode name.
            distance: Trip distance in km.

        Returns:
            Estimated number of transfers.
        """
        if mode in ["walking", "cycling", "e_scooter"]:
            return 0
        if mode == "hyperloop":
            return 1
        if distance > 30:
            return 2
        if distance > 10:
            return 1
        return 0

    def _estimate_walking(self, mode, distance):
        """Estimate walking distance to access a transport mode.

        Args:
            mode: Transport mode name.
            distance: Trip distance in km.

        Returns:
            Walking distance in meters.
        """
        walk_distances = {
            "autonomous_pod": 50,
            "high_speed_rail": 400,
            "hyperloop": 600,
            "walking": 0,
            "cycling": 100,
            "e_scooter": 80,
            "vertical_transit": 150,
            "water_taxi": 300,
        }
        return walk_distances.get(mode, 200)

    def _estimate_crowd(self, mode, hour, is_rush_hour):
        """Estimate crowd level for a transport mode at a given time.

        Args:
            mode: Transport mode name.
            hour: Hour of day (0-23).
            is_rush_hour: Whether it is rush hour.

        Returns:
            Crowd level score (1-5).
        """
        base_crowd = {
            "autonomous_pod": 1,
            "high_speed_rail": 3,
            "hyperloop": 2,
            "walking": 1,
            "cycling": 1,
            "e_scooter": 1,
            "vertical_transit": 2,
            "water_taxi": 2,
        }
        crowd = base_crowd.get(mode, 2)
        if is_rush_hour:
            crowd = min(5, crowd + 2)
        if 22 <= hour or hour < 6:
            crowd = max(1, crowd - 1)
        return crowd

    def _compute_accessibility(self, mode):
        """Compute accessibility rating for a transport mode.

        Args:
            mode: Transport mode name.

        Returns:
            Accessibility rating (1-5).
        """
        ratings = {
            "autonomous_pod": 5,
            "high_speed_rail": 4,
            "hyperloop": 4,
            "walking": 2,
            "cycling": 1,
            "e_scooter": 2,
            "vertical_transit": 5,
            "water_taxi": 3,
        }
        return ratings.get(mode, 3)

    def fit_scaler(self, trips_df):
        """Fit the feature scaler on historical trip data.

        Args:
            trips_df: DataFrame with historical trip features.

        Returns:
            self
        """
        feature_cols = [
            "travel_time_min", "num_transfers", "walking_distance_m",
            "carbon_footprint_g", "crowd_level", "cost_sar",
            "comfort_score", "accessibility_rating",
        ]
        available_cols = [c for c in feature_cols if c in trips_df.columns]
        self._scaler.fit(trips_df[available_cols])
        self._is_fitted = True
        print(f"Scaler fitted on {len(available_cols)} features")
        return self

    def normalize_features(self, trip_options):
        """Normalize trip option features to 0-1 range.

        Args:
            trip_options: DataFrame with trip option features.

        Returns:
            DataFrame with normalized feature columns.
        """
        feature_cols = [
            "travel_time_min", "num_transfers", "walking_distance_m",
            "carbon_footprint_g", "crowd_level", "cost_sar",
            "comfort_score", "accessibility_rating",
        ]
        available_cols = [c for c in feature_cols if c in trip_options.columns]

        result = trip_options.copy()
        if self._is_fitted:
            result[available_cols] = self._scaler.transform(result[available_cols])
        else:
            scaler = MinMaxScaler()
            result[available_cols] = scaler.fit_transform(result[available_cols])

        return result

    def build_candidate_matrix(self, origin, dest, hour, temperature):
        """Generate and normalize candidate trip options for an OD pair.

        Args:
            origin: Origin zone name.
            dest: Destination zone name.
            hour: Hour of day (0-23).
            temperature: Current temperature in Celsius.

        Returns:
            Tuple of (raw features DataFrame, normalized features DataFrame).
        """
        raw = self.build_trip_features(origin, dest, hour, temperature)
        if raw.empty:
            return raw, raw

        normalized = self.normalize_features(raw)
        return raw, normalized

    def get_feature_names(self):
        """Return the list of feature column names.

        Returns:
            List of feature name strings.
        """
        return [
            "travel_time_min", "num_transfers", "walking_distance_m",
            "carbon_footprint_g", "crowd_level", "cost_sar",
            "comfort_score", "accessibility_rating",
            "time_of_day", "temperature", "is_rush_hour",
        ]
