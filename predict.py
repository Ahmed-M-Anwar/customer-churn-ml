
import joblib  # تحميل الموديل المحفوظ
import pandas as pd  # تحويل بيانات العميل إلى DataFrame


MODEL_PATH = "models/churn_model.joblib"


def load_model():
    """تحميل الموديل وإعداداته من الملف."""
    return joblib.load(MODEL_PATH)


def predict_churn(customer_data):
    """التنبؤ باحتمالية إلغاء العميل للخدمة."""

    artifact = load_model()

    model = artifact["model"]
    threshold = artifact["threshold"]
    feature_columns = artifact["feature_columns"]

    # تجهيز بيانات العميل بنفس ترتيب Features وقت التدريب
    customer_df = pd.DataFrame([customer_data])
    customer_df = customer_df[feature_columns]

    # استخراج احتمالية انتماء العميل للفئة 1 (Churn = Yes)
    probability = model.predict_proba(customer_df)[0, 1]

    # تحويل الاحتمالية إلى قرار باستخدام Threshold المحفوظ
    prediction = int(probability >= threshold)

    return {
        "churn_probability": probability,
        "prediction": "Churn" if prediction == 1 else "No Churn",
        "threshold": threshold,
    }


def main():
    # بيانات تجريبية لعميل جديد
    customer = {
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

    result = predict_churn(customer)

    print("\n========== CUSTOMER CHURN PREDICTION ==========")
    print(f"Churn Probability: {result['churn_probability']:.2%}")
    print(f"Decision: {result['prediction']}")
    print(f"Threshold: {result['threshold']}")


if __name__ == "__main__":
    main()
