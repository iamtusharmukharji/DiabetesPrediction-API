from fastapi import FastAPI, Query
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
async def predict_diabetes(input_data: schemas.DiabetesInput, enable_probability: bool = Query(False, description="Return probability and risk level")):
    """Predict diabetes based on input features."""
    try:
        features = [
            input_data.age,
            input_data.bmi,
            input_data.hemoglobin,
            input_data.blood_glucose_level
            ]
        data = np.array([features])

        if enable_probability:
            prediction = model.predict_proba(data)
            # [[0.78506529 0.21493471]]
            print(prediction)
            prob = float(prediction[0][1])
            risk = "High"
            if prob >= 0.7:
                risk = "High"
            elif prob >= 0.4:
                risk = "Medium"
            else:
                risk = "Low"

            return {"success": True, "probability_%": round(prob*100, 1), "risk_level": risk}
        else:
            prediction = model.predict(data)

            return {"success": True, "prediction": int(prediction[0])}
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return {"success": False, "error": str(e)}