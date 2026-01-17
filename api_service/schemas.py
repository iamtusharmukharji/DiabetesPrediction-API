from pydantic import BaseModel

class DiabetesInput(BaseModel):
    age: int
    bmi: float
    hemoglobin: float
    blood_glucose_level: float