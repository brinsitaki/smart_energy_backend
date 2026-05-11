from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import firebase_admin
from firebase_admin import credentials, firestore

import pandas as pd
import numpy as np
import joblib

from datetime import datetime

import os
import uvicorn



app = FastAPI(
    title="Smart Energy API",
    version="1.0.0"
)


#############  CORS

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#############  FIREBASE

cred = credentials.Certificate("firebase-key.json")

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()

print("✅ Firestore connected")

# ==================================================
# LOAD MODEL
# ==================================================

model = joblib.load("energy_model.pkl")

print("✅ ML model loaded")


############  FEATURES

feature_cols = [
    "lag1",
    "lag2",
    "lag3",
    "lag4",
    "lag5",
    "lag6",
    "lag7",
    "rolling_mean_7",
    "day",
    "month",
    "weekday"
]


#############  STEG prices

def get_steg_price(monthly_kwh):

    if monthly_kwh <= 50:
        return 0.075

    elif monthly_kwh <= 100:
        return 0.108

    elif monthly_kwh <= 200:
        return 0.162

    elif monthly_kwh <= 300:
        return 0.198

    elif monthly_kwh <= 500:
        return 0.285

    else:
        return 0.350


############ get firestore data

def get_firestore_data():

    try:

        docs = db.collection("energy").stream()

        readings = []

        for doc in docs:

            data = doc.to_dict()

            data["id"] = doc.id

            readings.append(data)

        if len(readings) == 0:
            return None

        df = pd.DataFrame(readings)

        
        df = df.sort_values("id")

        if "power" not in df.columns:
            return None

        if "voltage" not in df.columns:
            df["voltage"] = 0

        if "current" not in df.columns:
            df["current"] = 0


        df["power"] = pd.to_numeric(
            df["power"],
            errors="coerce"
        )

        df["voltage"] = pd.to_numeric(
            df["voltage"],
            errors="coerce"
        )

        df["current"] = pd.to_numeric(
            df["current"],
            errors="coerce"
        )

        # remove invalid rows
        df = df.dropna(subset=["power"])

        if len(df) == 0:
            return None

        return df

    except Exception as e:

        print("❌ Firestore Error:", e)

        return None


############  get last values 

def get_last_readings():

    df = get_firestore_data()

    if df is None:
        return None

    values = df["power"].tail(7).values / 1000

    if len(values) == 0:
        return None

    if len(values) < 7:

        last_value = values[-1]

        while len(values) < 7:

            values = np.insert(
                values,
                0,
                last_value
            )

    return values

############  preduction function

def predict_next_month(last_values, days=30):

    future_dates = pd.date_range(
        start=datetime.now(),
        periods=days + 1,
        freq="D"
    )[1:]

    predictions = []

    for i in range(days):

        rolling_mean_7 = np.mean(
            last_values[-7:]
        )

        input_data = pd.DataFrame([[

            last_values[-1],
            last_values[-2],
            last_values[-3],
            last_values[-4],
            last_values[-5],
            last_values[-6],
            last_values[-7],
            rolling_mean_7,
            future_dates[i].day,
            future_dates[i].month,
            future_dates[i].weekday()

        ]], columns=feature_cols)

        try:

            pred = model.predict(input_data)[0]

        except Exception as e:

            return {
                "error": f"Prediction Error: {str(e)}"
            }

        predictions.append(float(pred))

        # update memory
        last_values = np.append(
            last_values,
            pred
        )


    ############ monthly calculation

    total_month_energy_kwh = 0

    for pred in predictions:

        total_month_energy_kwh += pred * 24

    price_per_kwh = get_steg_price(
        total_month_energy_kwh
    )

    total_month_cost = (
        total_month_energy_kwh *
        price_per_kwh
    )

    # Daily results

    daily_results = []

    for date, pred in zip(
        future_dates,
        predictions
    ):

        prediction_kw = float(pred)

        prediction_w = prediction_kw * 1000

        energy_kwh = prediction_kw * 24

        daily_cost = (
            energy_kwh *
            price_per_kwh
        )

        daily_results.append({

            "date": date.strftime("%Y-%m-%d"),

            "prediction_kw": round(
                prediction_kw,
                2
            ),

            "prediction_w": round(
                prediction_w,
                0
            ),

            "energy_kwh": round(
                energy_kwh,
                2
            ),

            "estimated_daily_cost_tnd": round(
                daily_cost,
                2
            )

        })

    return {

        "monthly_summary": {

            "predicted_total_energy_kwh": round(
                total_month_energy_kwh,
                2
            ),

            "steg_price_per_kwh_tnd": price_per_kwh,

            "estimated_total_bill_tnd": round(
                total_month_cost,
                2
            ),

            "average_daily_cost_tnd": round(
                total_month_cost / days,
                2
            )

        },

        "daily_predictions": daily_results

    }


#############  routes

@app.get("/")
def home():

    return {
        "message": "Smart Energy API Running"
    }


#############  health check

@app.get("/health")
def health():

    return {
        "status": "ok"
    }


#############  live data

@app.get("/live")
def live():

    df = get_firestore_data()

    if df is None:

        return {
            "error": "No Firestore data"
        }

    latest = df.iloc[-1]

    return {

        "power_w": float(
            latest["power"]
        ),

        "power_kw": round(
            float(latest["power"]) / 1000,
            3
        ),

        "voltage": float(
            latest["voltage"]
        ),

        "current": float(
            latest["current"]
        )

    }


#############  month preduction

@app.get("/predict")
def predict():

    last_values = get_last_readings()

    if last_values is None:

        return {
            "error": "No Firestore data"
        }

    return predict_next_month(last_values)


#############  main

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 8000)
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )