from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel

app = FastAPI(
    title="NEOM Mobility Recommender API",
    description="Personalized multimodal trip recommendations for NEOM residents.",
    version="1.0.0",
)

RECOMMENDER_PATH = Path("models/hybrid_recommender.pkl")
RESIDENTS_PATH = Path("data/residents.csv")
TRIPS_PATH = Path("data/trips.csv")

_recommender = None
_residents_df = None
_trips_df = None


@app.on_event("startup")
def load_artifacts():
    """Load recommender model and data at startup."""
    global _recommender, _residents_df, _trips_df

    if RECOMMENDER_PATH.exists():
        _recommender = joblib.load(RECOMMENDER_PATH)
        print(f"Recommender loaded from {RECOMMENDER_PATH}")

    if RESIDENTS_PATH.exists():
        _residents_df = pd.read_csv(RESIDENTS_PATH)
        print(f"Loaded {len(_residents_df)} resident profiles")

    if TRIPS_PATH.exists():
        _trips_df = pd.read_csv(TRIPS_PATH)
        print(f"Loaded {len(_trips_df)} trip records")


def get_recommender():
    """Dependency that provides the recommender model."""
    if _recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _recommender


def get_data():
    """Dependency that provides residents and trips DataFrames."""
    if _residents_df is None or _trips_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    return _residents_df, _trips_df


class TripRequest(BaseModel):
    resident_id: str = "R00001"
    origin_zone: str = "the_line_north"
    dest_zone: str = "oxagon_port"
    time_of_day: int = 12
    temperature: float = 30.0


@app.get("/status")
def status():
    """Status check endpoint."""
    return {"status": "ok", "model_loaded": _recommender is not None}


@app.get("/model-info")
def model_info(recommender=Depends(get_recommender)):
    """Return model metadata."""
    return {
        "model_type": type(recommender).__name__,
        "collab_weight": recommender.collab_weight,
        "content_weight": recommender.content_weight,
        "n_residents": len(_residents_df) if _residents_df is not None else 0,
        "n_trips": len(_trips_df) if _trips_df is not None else 0,
    }


@app.post("/recommend")
def recommend(request: TripRequest, recommender=Depends(get_recommender)):
    """Get personalized trip recommendations for a resident."""
    recommendations = recommender.recommend(
        resident_id=request.resident_id,
        origin=request.origin_zone,
        dest=request.dest_zone,
        hour=request.time_of_day,
        temperature=request.temperature,
        top_k=3,
    )

    total_carbon_saving = sum(r["carbon_saving_g"] for r in recommendations)

    return {
        "resident_id": request.resident_id,
        "origin": request.origin_zone,
        "destination": request.dest_zone,
        "recommendations": recommendations,
        "total_carbon_saving_g": round(total_carbon_saving, 1),
    }


@app.get("/resident/{resident_id}/history")
def resident_history(resident_id: str, data=Depends(get_data)):
    """Get trip history for a specific resident."""
    residents_df, trips_df = data

    resident = residents_df[residents_df["resident_id"] == resident_id]
    if resident.empty:
        raise HTTPException(status_code=404, detail="Resident not found")

    history = trips_df[trips_df["resident_id"] == resident_id]

    mode_stats = history.groupby("mode_chosen").agg(
        trip_count=("mode_chosen", "count"),
        avg_satisfaction=("satisfaction_rating", "mean"),
        avg_distance=("trip_distance_km", "mean"),
    ).reset_index()

    return {
        "resident": resident.iloc[0].to_dict(),
        "total_trips": len(history),
        "mode_stats": mode_stats.round(2).to_dict(orient="records"),
        "recent_trips": history.sort_values("timestamp", ascending=False)
            .head(10)
            .to_dict(orient="records"),
    }
