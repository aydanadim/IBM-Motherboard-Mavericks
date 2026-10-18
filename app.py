from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import numpy as np
import psycopg2
import joblib
import os

app = FastAPI(title="z-Sentinel Ultra-Low Latency API", version="1.1")

MODEL_PATH = "fraud_model.joblib"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    raise RuntimeError("Model file not found!")

class TransactionPayload(BaseModel):
    step: int
    type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float

# Non-blocking background task for database logging
def log_transaction_to_db(tx: TransactionPayload, score: float, is_fraud: int):
    try:
        conn = psycopg2.connect(
            dbname="postgres", user="postgres", password="sentinelpass", host="localhost", port="5432"
        )
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO transactions
            (step, type, amount, name_orig, old_balance_orig, new_balance_orig, name_dest, old_balance_dest, new_balance_dest, is_fraud, prediction_score)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (tx.step, tx.type, tx.amount, 'API_CLIENT', tx.oldbalanceOrg, tx.newbalanceOrig, 'API_RECEIVER', tx.oldbalanceDest, tx.newbalanceDest, is_fraud, score)
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Background DB Log Error: {e}")

@app.post("/predict")
def predict_fraud(tx: TransactionPayload, background_tasks: BackgroundTasks):
    try:
        type_code = hash(tx.type) % 5
        features = np.array([[
            tx.step, type_code, tx.amount,
            tx.oldbalanceOrg, tx.newbalanceOrig,
            tx.oldbalanceDest, tx.newbalanceDest
        ]], dtype=np.float32)

        # Core Snap ML Inference (Lightning Fast)
        score = float(model.predict_proba(features)[0][1])
        is_fraud = int(score > 0.5)

        # Offload DB logging to background so it doesn't block response time
        background_tasks.add_task(log_transaction_to_db, tx, score, is_fraud)

        return {
            "status": "success",
            "inference_engine": "IBM Snap ML",
            "fraud_probability": round(score, 4),
            "fraud_detected": bool(is_fraud)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
