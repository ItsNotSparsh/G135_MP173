import json
import logging
import sys



from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

class JsonFormatter(logging.Formatter):
    def format(self, record):
        entry = {
            "time": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        entry.update(getattr(record, "fields", {}))
        return json.dumps(entry)


logger = logging.getLogger("mp173_api")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="MP173 Cold Storage Spoilage API")


class LotRequest(BaseModel):
    storage_days: float
    temp_set_c: float
    humidity_pct: float
    power_cuts_hours: float
    bag_weight_kg: float = Field(..., le=85)          # client rule 1: above 85 kg is impossible
    door_openings_per_day: float = Field(..., ge=0)   # client rule 2: below 0 is a typing mistake
    crop: str
    chamber: str



@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(lot: LotRequest):
    row = pd.DataFrame([lot.model_dump()])
    prediction = float(model.predict(row)[0])
    logger.info(
        "prediction",
        extra={"fields": {
            "crop": lot.crop,
            "chamber": lot.chamber,
            "storage_days": lot.storage_days,
            "prediction": round(prediction, 3),
        }},
    )    
    return {"spoilage_pct": round(prediction, 3)}
