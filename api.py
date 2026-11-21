
from fastapi import FastAPI  # إنشاء HTTP API
from pydantic import BaseModel  # التحقق من شكل البيانات القادمة
import joblib  # تحميل الموديل المحفوظ
import pandas as pd  # تجهيز بيانات العميل للموديل
from typing import List  # Defines a list of customer records.


# تحميل الموديل مرة واحدة عند تشغيل الـ API
artifact = joblib.load("models/churn_model.joblib")

model = artifact["model"]
threshold = artifact["threshold"]
feature_columns = artifact["feature_columns"]

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict whether a telecom customer is likely to churn.",
    version="1.0.0",
)


# تعريف شكل البيانات التي يجب أن يرسلها العميل
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

class BatchPredictionRequest(BaseModel):
    customers: List[CustomerData]

@app.get("/")
def home():
    return {"message": "Customer Churn API is running"}


@app.post("/predict")
def predict(customer: CustomerData):

    # تحويل بيانات الطلب إلى DataFrame بنفس ترتيب التدريب
    customer_df = pd.DataFrame([customer.model_dump()])
    customer_df = customer_df[feature_columns]

    # حساب احتمالية إلغاء الخدمة
    probability = model.predict_proba(customer_df)[0, 1]

    # اتخاذ القرار باستخدام Threshold المحفوظ
    prediction = int(probability >= threshold)

    return {
        "churn_probability": round(float(probability), 4),
        "prediction": "Churn" if prediction else "No Churn",
        "threshold": threshold,
    }

@app.post("/predict/batch")
def predict_batch(request: BatchPredictionRequest):
    customers = request.customers

    if not customers:
        return {"predictions": []}

    # Convert all customers into a single DataFrame.
    data = pd.DataFrame(
        [customer.model_dump() for customer in customers]
    )

    # Preserve the exact feature order used during training.
    data = data[feature_columns]

    # Predict probabilities for the entire batch.
    probabilities = model.predict_proba(data)[:, 1]

    # Apply the saved decision threshold.
    predictions = [
        {
            "churn_probability": round(float(probability), 6),
            "prediction": (
                "Churn" if probability >= threshold else "Active"
            ),
            "threshold": threshold,
        }
        for probability in probabilities
    ]

    return {"predictions": predictions}
