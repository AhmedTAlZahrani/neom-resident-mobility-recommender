import json
from pathlib import Path

import numpy as np
import pandas as pd


class RecommendationEvaluator:

    def __init__(self):
        self._results = {}

    def precision_at_k(self, recommended, relevant, k=3):
        """Compute Precision@K for a single user.

        Args:
            recommended: List of recommended mode names.
            relevant: Set of relevant (actually used) mode names.
            k: Number of top recommendations to consider.

        Returns:
            Float precision score.
        """
        top_k = recommended[:k]
        if not top_k:
            return 0.0
        hits = sum(1 for m in top_k if m in relevant)
        return hits / len(top_k)

    def recall_at_k(self, recommended, relevant, k=3):
        """Compute Recall@K for a single user.

        Args:
            recommended: List of recommended mode names.
            relevant: Set of relevant (actually used) mode names.
            k: Number of top recommendations to consider.

        Returns:
            Float recall score.
        """
        top_k = recommended[:k]
        if not relevant:
            return 0.0
        hits = sum(1 for m in top_k if m in relevant)
        return hits / len(relevant)

    def ndcg_at_k(self, recommended, relevant, k=3):
        """Compute Normalized Discounted Cumulative Gain at K.

        Args:
            recommended: List of recommended mode names.
            relevant: Set of relevant (actually used) mode names.
            k: Number of top recommendations to consider.

        Returns:
            Float NDCG score.
        """
        top_k = recommended[:k]
        dcg = 0.0
        for i, mode in enumerate(top_k):
            if mode in relevant:
                dcg += 1.0 / np.log2(i + 2)

        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

        if idcg == 0:
            return 0.0
        return dcg / idcg

    def diversity_score(self, recommendations_list):
        """Compute diversity across a set of recommendation lists.

        Measures how varied the recommended modes are across all users.
        Higher diversity means less repetition.

        Args:
            recommendations_list: List of recommendation lists (each a list of mode names).

        Returns:
            Float diversity score between 0 and 1.
        """
        if not recommendations_list:
            return 0.0

        all_pairs = []
        for recs in recommendations_list:
            for i in range(len(recs)):
                for j in range(i + 1, len(recs)):
                    all_pairs.append(recs[i] != recs[j])

        if not all_pairs:
            return 0.0
        return round(sum(all_pairs) / len(all_pairs), 4)

    def coverage(self, recommendations_list, all_modes):
        """Compute coverage: fraction of modes recommended across all users.

        Args:
            recommendations_list: List of recommendation lists.
            all_modes: Set of all possible transport modes.

        Returns:
            Float coverage score between 0 and 1.
        """
        recommended_modes = set()
        for recs in recommendations_list:
            recommended_modes.update(recs)

        if not all_modes:
            return 0.0
        return round(len(recommended_modes) / len(all_modes), 4)

    def carbon_savings_vs_baseline(self, recommendations, trip_options_list):
        """Compute average carbon savings compared to fastest-route baseline.

        Args:
            recommendations: List of recommendation dicts with carbon_saving_g.
            trip_options_list: List of trip option DataFrames (unused, kept for API).

        Returns:
            Dict with mean and total carbon savings.
        """
        savings = [r.get("carbon_saving_g", 0) for r in recommendations]

        if not savings:
            return {"mean_saving_g": 0.0, "total_saving_g": 0.0, "pct_saving": 0.0}

        return {
            "mean_saving_g": round(np.mean(savings), 1),
            "total_saving_g": round(np.sum(savings), 1),
            "pct_saving": round(
                np.mean([s for s in savings if s != 0]) / max(1, np.mean(savings) + 100) * 100,
                1,
            ),
        }

    def satisfaction_correlation(self, predicted_scores, actual_ratings):
        """Compute correlation between predicted scores and actual ratings.

        Args:
            predicted_scores: List of predicted recommendation scores.
            actual_ratings: List of actual satisfaction ratings.

        Returns:
            Float Pearson correlation coefficient.
        """
        if len(predicted_scores) < 2 or len(actual_ratings) < 2:
            return 0.0

        predicted = np.array(predicted_scores[:len(actual_ratings)])
        actual = np.array(actual_ratings[:len(predicted_scores)])

        if predicted.std() == 0 or actual.std() == 0:
            return 0.0

        correlation = np.corrcoef(predicted, actual)[0, 1]
        return round(float(correlation), 4)

    def evaluate_recommender(self, recommender, test_residents, test_trips, k=3):
        """Run full evaluation of a recommender on test data.

        Args:
            recommender: Fitted HybridRecommender instance.
            test_residents: DataFrame with test resident profiles.
            test_trips: DataFrame with test trip records.
            k: Number of recommendations to evaluate.

        Returns:
            Dict with all evaluation metrics.
        """
        print("Evaluating recommender performance...")

        precisions = []
        recalls = []
        ndcgs = []
        all_recs_modes = []
        all_recs = []

        sample = test_residents.sample(n=min(300, len(test_residents)), random_state=42)

        for _, resident in sample.iterrows():
            origin = resident["home_zone"]
            dest = resident["work_zone"]
            if origin == dest:
                continue

            recs = recommender.recommend(resident["resident_id"], origin, dest)
            if not recs:
                continue

            rec_modes = [r["mode"] for r in recs]
            actual_trips = test_trips[
                test_trips["resident_id"] == resident["resident_id"]
            ]
            if actual_trips.empty:
                continue

            relevant = set(actual_trips["mode_chosen"].unique())

            precisions.append(self.precision_at_k(rec_modes, relevant, k))
            recalls.append(self.recall_at_k(rec_modes, relevant, k))
            ndcgs.append(self.ndcg_at_k(rec_modes, relevant, k))
            all_recs_modes.append(rec_modes)
            all_recs.extend(recs)

        all_modes = set(test_trips["mode_chosen"].unique())

        metrics = {
            f"precision@{k}": round(np.mean(precisions), 4) if precisions else 0.0,
            f"recall@{k}": round(np.mean(recalls), 4) if recalls else 0.0,
            f"ndcg@{k}": round(np.mean(ndcgs), 4) if ndcgs else 0.0,
            "diversity": self.diversity_score(all_recs_modes),
            "coverage": self.coverage(all_recs_modes, all_modes),
            "carbon_savings": self.carbon_savings_vs_baseline(all_recs, []),
            "n_evaluated": len(precisions),
        }

        print(f"  Precision@{k}: {metrics[f'precision@{k}']:.4f}")
        print(f"  Recall@{k}: {metrics[f'recall@{k}']:.4f}")
        print(f"  NDCG@{k}: {metrics[f'ndcg@{k}']:.4f}")
        print(f"  Diversity: {metrics['diversity']:.4f}")
        print(f"  Coverage: {metrics['coverage']:.4f}")

        self._results = metrics
        return metrics

    def compare_strategies(self, ab_results):
        """Format A/B test results for comparison.

        Args:
            ab_results: Dict with results from HybridRecommender.run_ab_test.

        Returns:
            DataFrame comparing strategies.
        """
        rows = []
        for strategy, metrics in ab_results.items():
            row = {"strategy": strategy}
            row.update(metrics)
            rows.append(row)

        comparison = pd.DataFrame(rows)
        print("\nStrategy Comparison:")
        print(comparison.to_string(index=False))
        return comparison

    def save_results(self, path="output/evaluation_results.json"):
        """Save evaluation results to a JSON file.

        Args:
            path: Output file path.
        """
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w") as f:
            json.dump(self._results, f, indent=2, default=str)
        print(f"Evaluation results saved to {output}")
