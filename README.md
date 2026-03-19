# NEOM Resident Mobility Recommender

Hybrid recommendation system for NEOM's car-free smart city. Combines collaborative filtering with content-based ranking to suggest optimal transport modes based on resident preferences, weather, and real-time context.

## Install

```bash
pip install -r requirements.txt
```

## Generate Data & Train

```python
from src.data_generator import generate_residents, generate_trip_records, save_datasets
from src.hybrid_recommender import HybridRecommender

residents = generate_residents()
trips = generate_trip_records(residents, n_trips=50000)
save_datasets(residents, trips)
rec = HybridRecommender()
rec.fit(residents, trips)
rec.save()
```

## Run

```bash
uvicorn api.main:app --port 8000   # API
streamlit run app.py                # dashboard
```

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"resident_id": "R00001", "origin_zone": "the_line_north", "dest_zone": "oxagon_port"}'
```
