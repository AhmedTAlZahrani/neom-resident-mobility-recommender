import numpy as np
import pandas as pd
import pytest

from src.collaborative_filter import CollaborativeFilter
from src.hybrid_recommender import HybridRecommender


def _make_trips(n_users=5, modes=None, rng_seed=42):
    """Small synthetic trip dataset."""
    rng = np.random.RandomState(rng_seed)
    modes = modes or ["autonomous_pod", "high_speed_rail", "cycling", "walking"]
    rows = []
    for uid in range(n_users):
        for mode in rng.choice(modes, size=rng.randint(2, len(modes) + 1), replace=False):
            rows.append({
                "resident_id": f"R{uid:03d}",
                "mode_chosen": mode,
                "satisfaction_rating": round(rng.uniform(1, 5), 1),
            })
    return pd.DataFrame(rows)


def _make_residents(n=5, zones=None):
    """Tiny residents table."""
    zones = zones or ["the_line_north", "the_line_central", "the_line_south"]
    rows = []
    for i in range(n):
        rows.append({
            "resident_id": f"R{i:03d}",
            "home_zone": zones[i % len(zones)],
            "work_zone": zones[(i + 1) % len(zones)],
            "sustainability_priority": 3,
            "comfort_preference": 3,
            "budget_sensitivity": 3,
            "fitness_level": 3,
            "has_mobility_limitation": False,
            "age": 30,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def small_trips():
    return _make_trips()


@pytest.fixture
def fitted_cf(small_trips):
    cf = CollaborativeFilter(n_neighbors=3)
    cf.fit(small_trips)
    return cf


# --- CollaborativeFilter tests ---

class TestCollaborativeFilter:

    def test_fit_builds_matrix(self, fitted_cf):
        """Matrix dimensions match users and modes."""
        mat = fitted_cf._user_mode_matrix
        assert mat.shape[0] > 0
        assert mat.shape[1] > 0

    def test_user_similarity_shape(self, fitted_cf):
        n = fitted_cf._user_mode_matrix.shape[0]
        assert fitted_cf._user_similarity.shape == (n, n)

    def test_user_similarity_diagonal_zero(self, fitted_cf):
        diag = np.diag(fitted_cf._user_similarity)
        np.testing.assert_array_equal(diag, 0)

    def test_mode_similarity_shape(self, fitted_cf):
        n = fitted_cf._user_mode_matrix.shape[1]
        assert fitted_cf._mode_similarity.shape == (n, n)

    def test_predict_user_user_returns_all_modes(self, fitted_cf):
        modes = ["autonomous_pod", "cycling", "walking"]
        scores = fitted_cf.predict_user_user("R000", modes)
        assert set(scores.keys()) == set(modes)

    def test_predict_user_user_scores_positive(self, fitted_cf):
        modes = ["autonomous_pod", "cycling"]
        scores = fitted_cf.predict_user_user("R000", modes)
        for v in scores.values():
            assert v >= 0

    def test_predict_item_item_returns_all_modes(self, fitted_cf):
        modes = ["autonomous_pod", "high_speed_rail"]
        scores = fitted_cf.predict_item_item("R001", modes)
        assert set(scores.keys()) == set(modes)

    def test_predict_combined_blends(self, fitted_cf):
        """Combined should be a weighted mix."""
        modes = ["autonomous_pod", "cycling"]
        combined = fitted_cf.predict_combined("R000", modes, user_weight=0.5)
        uu = fitted_cf.predict_user_user("R000", modes)
        ii = fitted_cf.predict_item_item("R000", modes)
        for m in modes:
            expected = 0.5 * uu[m] + 0.5 * ii[m]
            assert abs(combined[m] - expected) < 1e-9

    def test_cold_start_unknown_user(self, fitted_cf):
        modes = ["autonomous_pod", "walking"]
        scores = fitted_cf.predict_user_user("UNKNOWN_USER", modes)
        assert len(scores) == 2
        for v in scores.values():
            assert v > 0

    def test_cold_start_unknown_mode(self, fitted_cf):
        scores = fitted_cf.predict_user_user("R000", ["fake_mode"])
        assert "fake_mode" in scores

    def test_get_user_profile_known(self, fitted_cf):
        profile = fitted_cf.get_user_profile("R000")
        assert profile is not None
        assert len(profile) > 0
        for rating in profile.values():
            assert 1.0 <= rating <= 5.0

    def test_get_user_profile_unknown(self, fitted_cf):
        assert fitted_cf.get_user_profile("NO_SUCH_USER") is None

    def test_get_similar_users(self, fitted_cf):
        similar = fitted_cf.get_similar_users("R000", top_k=2)
        assert len(similar) <= 2
        for uid, sim in similar:
            assert isinstance(uid, str)
            assert isinstance(sim, float)

    def test_get_similar_users_unknown(self, fitted_cf):
        assert fitted_cf.get_similar_users("GHOST") == []

    def test_mode_popularity_computed(self, fitted_cf):
        assert len(fitted_cf._mode_popularity) > 0
        for v in fitted_cf._mode_popularity.values():
            assert 0 <= v <= 1


# --- HybridRecommender combine/rerank tests ---

class TestHybridScoreMerge:

    def test_combine_scores_weights(self):
        rec = HybridRecommender(collab_weight=0.3, content_weight=0.7)
        modes = ["a", "b"]
        collab = {"a": 0.5, "b": 1.0}
        content = {"a": 1.0, "b": 0.5}
        merged = rec._combine_scores(modes, collab, content)
        assert "a" in merged and "b" in merged
        # content_weight=0.7 so mode 'a' (content=1.0) should beat 'b' (content=0.5)
        assert merged["a"] > merged["b"]

    def test_combine_scores_equal_weights(self):
        rec = HybridRecommender(collab_weight=0.5, content_weight=0.5)
        collab = {"x": 0.8}
        content = {"x": 0.8}
        merged = rec._combine_scores(["x"], collab, content)
        # normalized: both are 1.0, so result should be 1.0
        assert abs(merged["x"] - 1.0) < 1e-6

    def test_contextual_rerank_hot_weather(self):
        rec = HybridRecommender()
        scores = {"walking": 0.8, "autonomous_pod": 0.7}
        options = pd.DataFrame([
            {"mode": "walking"}, {"mode": "autonomous_pod"},
        ])
        adjusted = rec._contextual_rerank(scores, options, hour=12, temperature=45)
        assert adjusted["walking"] < scores["walking"]

    def test_contextual_rerank_cool_weather(self):
        rec = HybridRecommender()
        scores = {"cycling": 0.5, "autonomous_pod": 0.5}
        options = pd.DataFrame([
            {"mode": "cycling"}, {"mode": "autonomous_pod"},
        ])
        adjusted = rec._contextual_rerank(scores, options, hour=12, temperature=20)
        assert adjusted["cycling"] > scores["cycling"]

    def test_contextual_rerank_night(self):
        rec = HybridRecommender()
        scores = {"autonomous_pod": 0.5, "e_scooter": 0.5}
        options = pd.DataFrame([
            {"mode": "autonomous_pod"}, {"mode": "e_scooter"},
        ])
        adjusted = rec._contextual_rerank(scores, options, hour=23, temperature=30)
        assert adjusted["autonomous_pod"] > scores["autonomous_pod"]
        assert adjusted["e_scooter"] < scores["e_scooter"]

    def test_apply_diversity_picks_different_categories(self):
        rec = HybridRecommender()
        scores = {
            "high_speed_rail": 0.9,
            "hyperloop": 0.85,
            "autonomous_pod": 0.7,
            "walking": 0.6,
        }
        selected = rec._apply_diversity(scores, top_k=3)
        assert len(selected) == 3
        # should not pick both rail modes when top_k=3
        rail_count = sum(1 for m in selected if m in ("high_speed_rail", "hyperloop"))
        assert rail_count <= 1

    def test_apply_diversity_respects_top_k(self):
        rec = HybridRecommender()
        scores = {"autonomous_pod": 0.9, "walking": 0.5}
        selected = rec._apply_diversity(scores, top_k=5)
        assert len(selected) == 2
