from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, Optional
import io, json, time

import numpy as np
import pandas as pd

from core.data import prepare_data_from_csv
from core.metrics import metrics_from_probs, details_from_preds
from core.registry import get_classical_runner, get_quantum_runner

app = FastAPI(title="QML Compare API", version="0.3.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ComparePayload(BaseModel):
    classicalModel: str
    quantumModel: str
    classicalParams: Dict[str, Any] = {}
    quantumParams: Dict[str, Any] = {}
    targetColumn: Optional[str] = None

@app.post("/api/compare")
async def compare_api(file: UploadFile = File(...), payload: str = Form(...)):
    print("/api/compare called")
    try:
        p = ComparePayload(**json.loads(payload))
    except Exception as e:
        return {"error": f"Invalid payload JSON: {e}"}

    csv_bytes = await file.read()
    try:
        (
            X_tr, X_te, y_tr, y_te,
            label_encoder, scaler,
            target_note, dataset_info
        ) = prepare_data_from_csv(csv_bytes, p.targetColumn)
    except Exception as e:
        return {"error": f"Data error: {e}"}

    classes = dataset_info["classes"]

    # ---- classical ----
    run_classical = get_classical_runner(p.classicalModel)
    try:
        t0 = time.perf_counter()
        proba_c, c_timings, c_extras = run_classical(X_tr, y_tr, X_te, p.classicalParams, classes)
        c_total = (time.perf_counter() - t0) * 1000.0
        c_metrics = metrics_from_probs(y_te, proba_c) | {"latency_ms": c_total}
        c_details = details_from_preds(y_te, proba_c, classes, timings=c_timings, extras=c_extras)
    except Exception as e:
        return {"error": f"Classical model '{p.classicalModel}' failed: {e}"}

    # ---- quantum ----
    run_quantum = get_quantum_runner(p.quantumModel)
    try:
        t0 = time.perf_counter()
        proba_q, q_timings, q_extras = run_quantum(X_tr, y_tr, X_te, p.quantumParams, classes)
        q_total = (time.perf_counter() - t0) * 1000.0
        q_metrics = metrics_from_probs(y_te, proba_q) | {"latency_ms": q_total}
        q_details = details_from_preds(y_te, proba_q, classes, timings=q_timings, extras=q_extras)
    except Exception as e:
        return {"error": f"Quantum model '{p.quantumModel}' failed: {e}"}

    # Diagnostics for charts (limit very large payloads)
    max_points = 5000
    y_true = y_te.tolist()
    diag = {
        "y_true": y_true[:max_points],
        "classical": {"proba": proba_c[:max_points].tolist()},
        "quantum":   {"proba": proba_q[:max_points].tolist()},
    }

    return {
        "summary": {
            "classicalModel": p.classicalModel,
            "quantumModel": p.quantumModel,
            "samples": dataset_info["n_samples"],
            "target": dataset_info["target"],
            "n_features": dataset_info["n_features"],
            "classes": classes,
            "class_counts": dataset_info["class_counts"],
        },
        "metrics": {"classical": c_metrics, "quantum": q_metrics},
        "details": {"classical": c_details, "quantum": q_details},
        "diagnostics": diag,
        "notes": target_note,
    }

@app.get("/api/health")
def health():
    return {"ok": True, "service": "qml-compare-api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
