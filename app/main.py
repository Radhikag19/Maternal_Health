from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
import joblib
import pandas as pd


# Update these labels if your training label encoding changes.
RISK_LABELS = {
    0: "Low Risk",
    1: "High Risk",
}


try:
    model = joblib.load("app/model.pkl")
except FileNotFoundError:
    print("Error: Model file not found.")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class PatientParams(BaseModel):
    Age: int
    Systolic_BP: float = None
    Diastolic: float = None
    BS: float = None
    Body_Temp: float = None
    BMI: float
    Previous_Complications: int = None
    Preexisting_Diabetes: int = None
    Gestational_Diabetes: int = None
    Mental_Health: int = None
    Heart_Rate: float = None


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def predict(data: Union[PatientParams, List[PatientParams]]):
    if isinstance(data, PatientParams):
        data = [data]

    df = pd.DataFrame([item.model_dump() for item in data])
    df.columns = [col.replace('_', ' ').strip() for col in df.columns]
    missing_flags = df.isnull().astype(int)
    missing_flags.columns = [f"{col}_missing" for col in missing_flags.columns]
    missing_flags = missing_flags.drop(['Age_missing', 'Mental Health_missing','Gestational Diabetes_missing', 'Body Temp_missing'],axis=1)
    df = df.fillna(-999)
    X = pd.concat([df,missing_flags], axis=1)
    prediction = model.predict(X)
    prediction_list = prediction.tolist()
    prediction_labels = [RISK_LABELS.get(int(value), "Unknown") for value in prediction_list]

    return {
        "prediction": prediction_list,
        "prediction_label": prediction_labels,
    }
