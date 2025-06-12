from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List
import joblib
import numpy as np

app = FastAPI()

# Wczytanie modeli
model_white = joblib.load("whitewine_rf.pkl")
model_red = joblib.load("redwine_rf.pkl")

# Wczytanie skalerów
scaler_white = joblib.load("scaler_white.pkl")
scaler_red = joblib.load("scaler_red.pkl")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None, "selected": "white"})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    wine_type: str = Form(...),
    features: List[float] = Form(...)
):
    X = np.array([features])

    # Wybór odpowiedniego modelu i skalera
    if wine_type == "white":
        X_scaled = scaler_white.transform(X)
        prediction_raw = model_white.predict(X_scaled)[0]
        prediction_map = {0: "Zła jakość", 1: "Średnia jakość", 2: "Dobra jakość"}
    else:
        X_scaled = scaler_red.transform(X)
        prediction_raw = model_red.predict(X_scaled)[0]
        prediction_map = {0: "Zła jakość", 1: "Dobra jakość"}

    prediction = prediction_map.get(int(prediction_raw), "Nieznana")

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": prediction,
            "selected": wine_type
        }
    )