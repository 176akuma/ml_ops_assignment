from pathlib import Path

import joblib
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, confloat
from prometheus_client import Counter, CONTENT_TYPE_LATEST, generate_latest

from src.training_utils import ensure_sqlite, log_prediction, setup_logger


with open("src/config/config.yaml") as f:
    config = yaml.safe_load(f)

model_path = Path(config["artifacts"]["models_dir"]) / config["artifacts"]["model_file"]
if not model_path.exists():
    raise FileNotFoundError(f"Model not found at {model_path}. Run training first.")

model = joblib.load(model_path)

app = FastAPI(title="California Housing API", version="1.0")

logger = setup_logger(Path("artifacts/logs/app.log"))
ensure_sqlite(Path("artifacts/db/preds.sqlite"))

PRED_COUNTER = Counter("predictions_total", "Number of predictions served")


class Features(BaseModel):
    MedInc: confloat(gt=0)
    HouseAge: confloat(ge=0)
    AveRooms: confloat(gt=0)
    AveBedrms: confloat(gt=0)
    Population: confloat(ge=0)
    AveOccup: confloat(gt=0)
    Latitude: confloat(ge=-90, le=90)
    Longitude: confloat(ge=-180, le=180)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(feats: Features):
    try:
        df = pd.DataFrame([feats.dict()])
        pred = float(model.predict(df)[0])
        logger.info(f"payload={feats.dict()} pred={pred:.6f}")
        log_prediction(Path("artifacts/db/preds.sqlite"), feats.dict(), pred)
        PRED_COUNTER.inc()
        return {"prediction": pred}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
