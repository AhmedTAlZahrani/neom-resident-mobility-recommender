import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from .trip_feature_builder import TripFeatureBuilder
from .collaborative_filter import CollaborativeFilter
from .content_ranker import ContentRanker
from .data_generator import TRANSPORT_MODES


class HybridRecommender:

    def __init__(self, collab_weight=0.4, content_weight=0.6):
        self.collab_weight = collab_weight
        self.content_weight = content_weight
        self.feature_builder = TripFeatureBuilder()
        self.collab_filter = CollaborativeFilter()
        self.content_ranker = ContentRanker()
        self._is_fitted = False

    def fit(self, residents, trips):
        """Fit all recommender components on historical data."""
        print("Fitting hybrid recommender...")
        self.collab_filter.fit(trips)

        sample_features = []
        zones = residents["home_zone"].unique()[:4]
        for origin in zones:
            for dest in zones:
                if origin != dest:
                    features = self.feature_builder.build_trip_features(
                        origin, dest, 12, 30
                    )
                    sample_features.append(features)

        if sample_features:
            combined = pd.concat(sample_features, ignore_index=True)
            self.feature_builder.fit_scaler(combined)

        self._residents = residents.set_index("resident_id")
        self._is_fitted = True
        print("Hybrid recommender fitted successfully")
        return self

    def recommend(self, resident_id, origin, dest, hour=12, temperature=30, top_k=3):
        """Generate top-K trip recommendations for a resident."""
        raw_options, norm_options = self.feature_builder.build_candidate_matrix(
            origin, dest, hour, temperature
        )

        if raw_options.empty:
            return []

        available_modes = raw_options["mode"].tolist()

        collab_scores = self.collab_filter.predict_combined(
            resident_id, available_modes
        )

        if resident_id in self._residents.index:
            resident = self._residents.loc[resident_id]
        else:
            resident = {
                "sustainability_priority": 3,
                "comfort_preference": 3,
                "budget_sensitivity": 3,
                "fitness_level": 3,
                "has_mobility_limitation": False,
                "age": 35,
            }

        content_ranked = self.content_ranker.rank_trip_options(raw_options, resident)

        content_scores = {}
        for _, row in content_ranked.iterrows():
            content_scores[row["mode"]] = row["content_score"]

        hybrid_scores = self._combine_scores(
            available_modes, collab_scores, content_scores
        )

        hybrid_scores = self._contextual_rerank(
            hybrid_scores, raw_options, hour, temperature
        )

        diverse_modes = self._apply_diversity(hybrid_scores, top_k)

        recommendations = []
        for mode in diverse_modes:
            mode_features = raw_options[raw_options["mode"] == mode].iloc[0]
            weights = self.content_ranker.derive_weights(resident)
            explanation = self._generate_explanation(
                mode, mode_features, resident, weights
            )

            fastest_mode = raw_options.loc[
                raw_options["travel_time_min"].idxmin(), "mode"
            ]
            carbon_saving = self._compute_carbon_saving(
                mode, fastest_mode, raw_options
            )

            recommendations.append({
                "mode": mode,
                "score": round(hybrid_scores[mode], 4),
                "explanation": explanation,
                "carbon_saving_g": carbon_saving,
                "travel_time_min": float(mode_features["travel_time_min"]),
                "cost_sar": float(mode_features["cost_sar"]),
                "comfort_score": float(mode_features["comfort_score"]),
                "carbon_footprint_g": float(mode_features["carbon_footprint_g"]),
            })

        return recommendations

    def _combine_scores(self, modes, collab_scores, content_scores):
        hybrid = {}

        content_vals = list(content_scores.values())
        max_content = max(content_vals) if content_vals else 1.0
        collab_vals = list(collab_scores.values())
        max_collab = max(collab_vals) if collab_vals else 1.0

        for mode in modes:
            c_score = content_scores.get(mode, 0) / max(max_content, 1e-6)
            f_score = collab_scores.get(mode, 0) / max(max_collab, 1e-6)
            hybrid[mode] = (
                self.content_weight * c_score
                + self.collab_weight * f_score
            )

        return hybrid

    def _contextual_rerank(self, scores, trip_options, hour, temperature):
        adjusted = dict(scores)

        for _, row in trip_options.iterrows():
            mode = row["mode"]

            if temperature > 40 and mode in ["walking", "cycling"]:
                adjusted[mode] *= 0.5

            if temperature < 25 and mode in ["walking", "cycling", "e_scooter"]:
                adjusted[mode] *= 1.3

            if 7 <= hour <= 9 or 17 <= hour <= 19:
                if mode in ["high_speed_rail", "hyperloop"]:
                    adjusted[mode] *= 0.85
                if mode == "autonomous_pod":
                    adjusted[mode] *= 1.1

            if 22 <= hour or hour < 6:
                if mode in ["autonomous_pod"]:
                    adjusted[mode] *= 1.2
                if mode in ["water_taxi", "e_scooter"]:
                    adjusted[mode] *= 0.6

        return adjusted

    def _apply_diversity(self, scores, top_k):
        mode_categories = {
            "autonomous_pod": "road",
            "high_speed_rail": "rail",
            "hyperloop": "rail",
            "walking": "active",
            "cycling": "active",
            "e_scooter": "micro",
            "vertical_transit": "vertical",
            "water_taxi": "water",
        }

        sorted_modes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = []
        used_categories = set()

        for mode, score in sorted_modes:
            if len(selected) >= top_k:
                break
            category = mode_categories.get(mode, "other")
            if category not in used_categories:
                selected.append(mode)
                used_categories.add(category)

        if len(selected) < top_k:
            for mode, score in sorted_modes:
                if mode not in selected:
                    selected.append(mode)
                if len(selected) >= top_k:
                    break

        return selected

    def _generate_explanation(self, mode, features, resident, weights):
        reasons = []

        carbon = features.get("carbon_footprint_g", 0)
        sustainability = resident.get("sustainability_priority", 3)
        if sustainability >= 4 and carbon == 0:
            reasons.append("you prefer low-carbon options and this mode produces zero emissions")

        comfort = features.get("comfort_score", 3)
        comfort_pref = resident.get("comfort_preference", 3)
        if comfort_pref >= 4 and comfort >= 4.0:
            reasons.append(f"this offers high comfort (score: {comfort}/5)")

        cost = features.get("cost_sar", 0)
        budget = resident.get("budget_sensitivity", 3)
        if budget >= 4 and cost < 5:
            reasons.append(f"it is budget-friendly at {cost:.1f} SAR")

        travel_time = features.get("travel_time_min", 0)
        if travel_time < 15:
            reasons.append(f"fast travel time of {travel_time:.0f} minutes")

        if not reasons:
            reasons.append("it provides the best balance of speed, comfort, and sustainability")

        display_mode = mode.replace("_", " ").title()
        explanation = f"Recommended {display_mode} because " + " and ".join(reasons[:2])
        return explanation

    def _compute_carbon_saving(self, recommended_mode, fastest_mode, trip_options):
        rec_row = trip_options[trip_options["mode"] == recommended_mode]
        fast_row = trip_options[trip_options["mode"] == fastest_mode]

        if rec_row.empty or fast_row.empty:
            return 0.0

        rec_carbon = rec_row.iloc[0]["carbon_footprint_g"]
        fast_carbon = fast_row.iloc[0]["carbon_footprint_g"]
        return round(float(fast_carbon - rec_carbon), 1)

    def run_ab_test(self, residents, trips, test_residents, test_trips, n_tests=200):
        """Simulate A/B test comparing recommendation strategies.

        Args:
            residents: DataFrame with all resident profiles.
            trips: DataFrame with all trip records.
            test_residents: DataFrame with test resident profiles.
            test_trips: DataFrame with test trip records.
            n_tests: Number of test recommendations to generate.

        Returns:
            Dict with results for each strategy.
        """
        print("Running A/B test simulation...")
        strategies = {
            "hybrid": {"collab_weight": 0.4, "content_weight": 0.6},
            "content_only": {"collab_weight": 0.0, "content_weight": 1.0},
            "collaborative_only": {"collab_weight": 1.0, "content_weight": 0.0},
        }

        results = {}
        sample_residents = test_residents.sample(
            n=min(n_tests, len(test_residents)), random_state=42
        )

        for strategy_name, weights in strategies.items():
            print(f"  Testing strategy: {strategy_name}")
            self.collab_weight = weights["collab_weight"]
            self.content_weight = weights["content_weight"]

            hits = 0
            total = 0
            carbon_savings = []

            for _, resident in sample_residents.iterrows():
                origin = resident["home_zone"]
                dest = resident["work_zone"]
                if origin == dest:
                    continue

                recs = self.recommend(resident["resident_id"], origin, dest)
                if not recs:
                    continue

                actual_trips = test_trips[
                    test_trips["resident_id"] == resident["resident_id"]
                ]
                if actual_trips.empty:
                    continue

                actual_modes = set(actual_trips["mode_chosen"].unique())
                rec_modes = [r["mode"] for r in recs]

                hits += len(set(rec_modes) & actual_modes)
                total += len(rec_modes)
                carbon_savings.extend([r["carbon_saving_g"] for r in recs])

            precision = hits / max(total, 1)
            avg_carbon = np.mean(carbon_savings) if carbon_savings else 0

            results[strategy_name] = {
                "precision": round(precision, 4),
                "avg_carbon_saving_g": round(avg_carbon, 1),
                "n_recommendations": total,
            }

        self.collab_weight = 0.4
        self.content_weight = 0.6

        return results

    def save(self, output_dir="models"):
        """Save the recommender components to disk."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.collab_filter, path / "collab_filter.pkl")
        joblib.dump(self.feature_builder, path / "feature_builder.pkl")
        joblib.dump(self, path / "hybrid_recommender.pkl")
        print(f"Recommender saved to {path}")

    @staticmethod
    def load(output_dir="models"):
        """Load a saved recommender from disk."""
        path = Path(output_dir)
        recommender = joblib.load(path / "hybrid_recommender.pkl")
        print(f"Recommender loaded from {path}")
        return recommender

