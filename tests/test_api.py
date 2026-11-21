
from fastapi.testclient import TestClient  # إرسال HTTP requests تجريبية للـ API
from api import app  # استيراد تطبيق FastAPI لاختباره


client = TestClient(app)


# بيانات عميل تجريبية صحيحة
VALID_CUSTOMER = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 3,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 75.0,
    "TotalCharges": 225.0,
}


def test_home_endpoint():
    """التأكد أن الـ API تعمل."""
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Customer Churn API is running"


def test_predict_valid_customer():
    """التأكد أن التنبؤ يعمل مع بيانات صحيحة."""
    response = client.post("/predict", json=VALID_CUSTOMER)

    assert response.status_code == 200

    result = response.json()

    assert "churn_probability" in result
    assert "prediction" in result
    assert "threshold" in result

    assert 0 <= result["churn_probability"] <= 1
    assert result["prediction"] in ["Churn", "No Churn"]


def test_predict_missing_field():
    """التأكد أن البيانات الناقصة يتم رفضها."""
    invalid_customer = VALID_CUSTOMER.copy()

    # حذف حقل مطلوب لاختبار التحقق من البيانات
    del invalid_customer["tenure"]

    response = client.post("/predict", json=invalid_customer)

    assert response.status_code == 422
