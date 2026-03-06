import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


# collaborative filter v1 - replaced with hybrid approach
# def simple_cf(user_id, items):
#     ...


class CollaborativeFilter:

    def __init__(self, n_neighbors=20):
        self.n_neighbors = n_neighbors
        self._user_mode_matrix = None
        self._user_similarity = None
        self._mode_similarity = None
        self._user_index = None
        self._mode_index = None
        self._mode_popularity = None
        self._global_mean = 0

    def fit(self, trips):
        """Build user-mode interaction matrix and compute similarities."""
        print("Building user-mode interaction matrix...")

        interaction = trips.groupby(
            ["resident_id", "mode_chosen"]
        )["satisfaction_rating"].mean().reset_index()

        self._user_index = {uid: i for i, uid in enumerate(
            interaction["resident_id"].unique()
        )}
        self._mode_index = {mode: i for i, mode in enumerate(
            interaction["mode_chosen"].unique()
        )}

        n_users = len(self._user_index)
        n_modes = len(self._mode_index)

        row_indices = [self._user_index[r] for r in interaction["resident_id"]]
        col_indices = [self._mode_index[m] for m in interaction["mode_chosen"]]
        values = interaction["satisfaction_rating"].values

        self._user_mode_matrix = sparse.csr_matrix(
            (values, (row_indices, col_indices)),
            shape=(n_users, n_modes),
        )

        self._global_mean = values.mean()
        print(f"  Matrix shape: {n_users} users x {n_modes} modes")
        print(f"  Sparsity: {1 - len(values) / (n_users * n_modes):.2%}")

        self._compute_user_similarity()
        self._compute_mode_similarity()
        self._compute_popularity(trips)

        print("Collaborative filter fitted successfully")
        return self

    def _compute_user_similarity(self):
        """Compute pairwise cosine similarity between users."""
        print("  Computing user-user similarity...")
        dense = self._user_mode_matrix.toarray()
        self._user_similarity = cosine_similarity(dense)
        np.fill_diagonal(self._user_similarity, 0)

    def _compute_mode_similarity(self):
        """Compute pairwise cosine similarity between modes."""
        print("  Computing mode-mode similarity...")
        dense = self._user_mode_matrix.toarray().T
        self._mode_similarity = cosine_similarity(dense)
        np.fill_diagonal(self._mode_similarity, 0)

    def _compute_popularity(self, trips):
        counts = trips["mode_chosen"].value_counts(normalize=True)
        ratings = trips.groupby("mode_chosen")["satisfaction_rating"].mean()

        self._mode_popularity = {}
        for mode in counts.index:
            freq_score = counts[mode]
            rating_score = ratings[mode] / 5.0
            self._mode_popularity[mode] = round(
                0.4 * freq_score + 0.6 * rating_score, 4
            )

    def _get_neighbors(self, user_idx):
        similarities = self._user_similarity[user_idx]
        neighbor_indices = np.argsort(similarities)[::-1][:self.n_neighbors]
        return neighbor_indices

    def predict_user_user(self, resident_id, available_modes):
        """Predict mode scores using user-user collaborative filtering."""
        if resident_id not in self._user_index:
            return self._cold_start_scores(available_modes)

        user_idx = self._user_index[resident_id]
        neighbors = self._get_neighbors(user_idx)
        neighbor_sims = self._user_similarity[user_idx, neighbors]

        scores = {}
        for mode in available_modes:
            if mode not in self._mode_index:
                scores[mode] = self._global_mean / 5.0
                continue

            mode_idx = self._mode_index[mode]
            neighbor_ratings = self._user_mode_matrix[neighbors, mode_idx].toarray().flatten()

            mask = neighbor_ratings > 0
            if mask.sum() == 0:
                scores[mode] = self._global_mean / 5.0
                continue

            weighted_sum = np.sum(neighbor_sims[mask] * neighbor_ratings[mask])
            sim_sum = np.sum(np.abs(neighbor_sims[mask]))

            if sim_sum > 0:
                scores[mode] = weighted_sum / sim_sum / 5.0
            else:
                scores[mode] = self._global_mean / 5.0

        return scores

    def predict_item_item(self, resident_id, available_modes):
        """Predict mode scores using item-item collaborative filtering."""
        if resident_id not in self._user_index:
            return self._cold_start_scores(available_modes)

        user_idx = self._user_index[resident_id]
        user_ratings = self._user_mode_matrix[user_idx].toarray().flatten()

        scores = {}
        for mode in available_modes:
            if mode not in self._mode_index:
                scores[mode] = self._global_mean / 5.0
                continue

            mode_idx = self._mode_index[mode]
            mode_sims = self._mode_similarity[mode_idx]

            rated_mask = user_ratings > 0
            if rated_mask.sum() == 0:
                scores[mode] = self._global_mean / 5.0
                continue

            weighted_sum = np.sum(mode_sims[rated_mask] * user_ratings[rated_mask])
            sim_sum = np.sum(np.abs(mode_sims[rated_mask]))

            if sim_sum > 0:
                scores[mode] = weighted_sum / sim_sum / 5.0
            else:
                scores[mode] = self._global_mean / 5.0

        return scores

    def predict_combined(self, resident_id, available_modes, user_weight=0.6):
        """Combine user-user and item-item predictions."""
        user_scores = self.predict_user_user(resident_id, available_modes)
        item_scores = self.predict_item_item(resident_id, available_modes)

        combined = {}
        for mode in available_modes:
            combined[mode] = (
                user_weight * user_scores.get(mode, 0)
                + (1 - user_weight) * item_scores.get(mode, 0)
            )

        return combined

    def _cold_start_scores(self, available_modes):
        scores = {}
        for mode in available_modes:
            scores[mode] = self._mode_popularity.get(mode, 0.5)
        return scores

    def get_user_profile(self, resident_id):
        """Get a resident's mode usage profile."""
        if resident_id not in self._user_index:
            return None

        user_idx = self._user_index[resident_id]
        ratings = self._user_mode_matrix[user_idx].toarray().flatten()

        reverse_mode = {v: k for k, v in self._mode_index.items()}
        profile = {}
        for idx, rating in enumerate(ratings):
            if rating > 0:
                profile[reverse_mode[idx]] = round(float(rating), 2)

        return profile

    def get_similar_users(self, resident_id, top_k=5):
        """Find the most similar residents."""
        if resident_id not in self._user_index:
            return []

        user_idx = self._user_index[resident_id]
        similarities = self._user_similarity[user_idx]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        reverse_user = {v: k for k, v in self._user_index.items()}
        return [
            (reverse_user[idx], round(float(similarities[idx]), 4))
            for idx in top_indices
        ]

