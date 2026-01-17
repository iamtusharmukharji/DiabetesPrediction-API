from fastapi import FastAPI
import logging
from fastapi.responses import RedirectResponse
import schemas
import joblib
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format=" %(levelname)s : %(asctime)s -  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__) 
app = FastAPI(title="Diabetes Prediction API")

model = None

@app.on_event("startup")
async def load_model():
    global model
    model = joblib.load("../diabetes_model.pkl")
    logger.info("Model Loaded !")

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.post("/predict", tags=["Prediction"])
async def predict_diabetes(input_data: schemas.DiabetesInput):
    """Predict diabetes based on input features."""

    data = np.array([[input_data.age, input_data.bmi, input_data.hemoglobin, input_data.blood_glucose_level]])
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}