import numpy as np
import pandas as pd


class ContentRanker:

    def __init__(self):
        self._weight_configs = None

    def derive_weights(self, resident):
        """Derive scoring weights from a resident's profile."""
        sustainability = resident.get("sustainability_priority", 3)
        comfort = resident.get("comfort_preference", 3)
        budget = resident.get("budget_sensitivity", 3)
        fitness = resident.get("fitness_level", 3)

        raw_weights = {
            "time_weight": 3.0,
            "carbon_weight": sustainability * 0.8,
            "comfort_weight": comfort * 0.7,
            "cost_weight": budget * 0.6,
            "accessibility_weight": 2.0,
            "crowd_weight": comfort * 0.3,
            "fitness_weight": fitness * 0.4,
        }

        if resident.get("has_mobility_limitation", False):
            raw_weights["accessibility_weight"] = 5.0
            raw_weights["comfort_weight"] *= 1.5

        total = sum(raw_weights.values())
        weights = {k: round(v / total, 4) for k, v in raw_weights.items()}
        return weights

    def score_trip_option(self, trip_features, weights):
        """Score a single trip option using weighted feature matching."""
        score = 0.0

        travel_time = trip_features.get("travel_time_min", 30)
        time_score = max(0, 1.0 - travel_time / 120.0)
        score += weights["time_weight"] * time_score

        carbon = trip_features.get("carbon_footprint_g", 0)
        carbon_score = max(0, 1.0 - carbon / 500.0)
        score += weights["carbon_weight"] * carbon_score

        comfort = trip_features.get("comfort_score", 3.0)
        comfort_score = comfort / 5.0
        score += weights["comfort_weight"] * comfort_score

        cost = trip_features.get("cost_sar", 5.0)
        cost_score = max(0, 1.0 - cost / 50.0)
        score += weights["cost_weight"] * cost_score

        accessibility = trip_features.get("accessibility_rating", 3)
        accessibility_score = accessibility / 5.0
        score += weights["accessibility_weight"] * accessibility_score

        crowd = trip_features.get("crowd_level", 3)
        crowd_score = max(0, 1.0 - crowd / 5.0)
        score += weights["crowd_weight"] * crowd_score

        return round(float(score), 4)

    def rank_trip_options(self, trip_options, resident):
        """Rank all candidate trip options for a resident."""
        weights = self.derive_weights(resident)
        scores = []

        for _, row in trip_options.iterrows():
            score = self.score_trip_option(row.to_dict(), weights)
            scores.append(score)

        result = trip_options.copy()
        result["content_score"] = scores
        result = result.sort_values("content_score", ascending=False).reset_index(drop=True)
        return result

    def compute_preference_vector(self, resident):
        """Build a preference feature vector for a resident."""
        weights = self.derive_weights(resident)
        vector = np.array([
            weights["time_weight"],
            weights["carbon_weight"],
            weights["comfort_weight"],
            weights["cost_weight"],
            weights["accessibility_weight"],
            weights["crowd_weight"],
        ])
        return vector

    def compute_option_vector(self, trip_features):
        """Build a feature vector for a trip option."""
        travel_time = trip_features.get("travel_time_min", 30)
        time_score = max(0, 1.0 - travel_time / 120.0)

        carbon = trip_features.get("carbon_footprint_g", 0)
        carbon_score = max(0, 1.0 - carbon / 500.0)

        comfort = trip_features.get("comfort_score", 3.0) / 5.0
        cost = trip_features.get("cost_sar", 5.0)
        cost_score = max(0, 1.0 - cost / 50.0)

        accessibility = trip_features.get("accessibility_rating", 3) / 5.0
        crowd = trip_features.get("crowd_level", 3)
        crowd_score = max(0, 1.0 - crowd / 5.0)

        return np.array([
            time_score, carbon_score, comfort, cost_score,
            accessibility, crowd_score,
        ])

    def cosine_score(self, resident, trip_features):
        """Compute cosine similarity between resident preferences and trip."""
        pref = self.compute_preference_vector(resident)
        option = self.compute_option_vector(trip_features)

        dot = np.dot(pref, option)
        norm_pref = np.linalg.norm(pref)
        norm_option = np.linalg.norm(option)

        if norm_pref == 0 or norm_option == 0:
            return 0.0

        return round(float(dot / (norm_pref * norm_option)), 4)

    def explain_score(self, trip_features, weights):
        """Generate a breakdown of score components."""
        travel_time = trip_features.get("travel_time_min", 30)
        time_score = max(0, 1.0 - travel_time / 120.0)

        carbon = trip_features.get("carbon_footprint_g", 0)
        carbon_score = max(0, 1.0 - carbon / 500.0)

        comfort = trip_features.get("comfort_score", 3.0) / 5.0
        cost = trip_features.get("cost_sar", 5.0)
        cost_score = max(0, 1.0 - cost / 50.0)

        accessibility = trip_features.get("accessibility_rating", 3) / 5.0
        crowd = trip_features.get("crowd_level", 3)
        crowd_score = max(0, 1.0 - crowd / 5.0)

        return {
            "time": round(weights["time_weight"] * time_score, 4),
            "carbon": round(weights["carbon_weight"] * carbon_score, 4),
            "comfort": round(weights["comfort_weight"] * comfort_score, 4),
            "cost": round(weights["cost_weight"] * cost_score, 4),
            "accessibility": round(weights["accessibility_weight"] * accessibility_score, 4),
            "crowd": round(weights["crowd_weight"] * crowd_score, 4),
        }
