
# ============================================================
# 1. IMPORTS - استدعاء المكتبات
# ============================================================

import pandas as pd  # التعامل مع البيانات والجداول وعرض النتائج

from sklearn.model_selection import train_test_split  # تقسيم البيانات إلى Training وValidation وTest

from sklearn.compose import ColumnTransformer  # تطبيق معالجة مختلفة حسب نوع العمود
from sklearn.pipeline import Pipeline  # ربط خطوات المعالجة والموديل في مسار واحد
from sklearn.impute import SimpleImputer  # معالجة القيم المفقودة
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # تحويل النصوص لأرقام وتوحيد مقياس الأرقام

from sklearn.linear_model import LogisticRegression  # موديل التصنيف الذي يتوقع احتمالية إلغاء العميل

from sklearn.ensemble import RandomForestClassifier  # موديل يعتمد على مجموعة أشجار قرار لتحسين التصنيف

from sklearn.metrics import (  # مقاييس تقييم أداء الموديل
    accuracy_score,  # نسبة التوقعات الصحيحة
    classification_report,  # تقرير Precision وRecall وF1
    confusion_matrix,  # توزيع التوقعات الصحيحة والخاطئة
    roc_auc_score,  # قدرة الموديل على التمييز بين الفئتين
    precision_score,  # نسبة الإنذارات الصحيحة من إجمالي الإنذارات
    recall_score,  # نسبة العملاء المعرضين للإلغاء الذين اكتشفناهم
    f1_score,  # التوازن بين Precision وRecall
)

import joblib  # حفظ الموديل المدرب واستعادته لاحقًا
from pathlib import Path  # التعامل مع مسارات الملفات وإنشاء المجلدات

# ============================================================
# 2. CONFIGURATION - إعدادات المشروع
# ============================================================

DATASET_URL = (
    "https://raw.githubusercontent.com/"
    "IBM/telco-customer-churn-on-icp4d/master/data/"
    "Telco-Customer-Churn.csv"
)

RANDOM_STATE = 42  # تثبيت العشوائية للحصول على نفس التقسيم عند إعادة التشغيل

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]  # العتبات التي سنقارن بينها

MODEL_PATH = Path("models") / "churn_model.joblib"  # مكان حفظ الموديل النهائي

# ============================================================
# 3. DATA LOADING - تحميل البيانات
# ============================================================

def load_data():
    """تحميل البيانات الخام وتنظيف القيم الأساسية."""

    df = pd.read_csv(DATASET_URL)  # قراءة ملف CSV

    df["TotalCharges"] = pd.to_numeric(  # تحويل العمود من نص إلى رقم
        df["TotalCharges"],
        errors="coerce"  # القيم غير القابلة للتحويل تصبح NaN
    )

    df["TotalCharges"] = df["TotalCharges"].fillna(0)  # تعويض القيم الفارغة بصفر للعملاء ذوي tenure صفر

    return df  # إرجاع البيانات بعد التنظيف


# ============================================================
# 4. DATA SPLITTING - تقسيم البيانات
# ============================================================

def split_data(df):
    """تقسيم البيانات إلى Training وValidation وTest."""

    X = df.drop(  # إنشاء مجموعة الخصائص التي سيتعلم منها الموديل
        columns=["customerID", "Churn", "tenure_group"],  # استبعاد المعرف والنتيجة وعمود التحليل الاختياري
        errors="ignore"
    )

    y = df["Churn"].map(  # تحويل النتيجة إلى أرقام
        {"No": 0, "Yes": 1}
    )

    # أول تقسيم: 80% للتدريب والتحقق، و20% للاختبار النهائي
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # ثاني تقسيم: أخذ 20% من الـ 80% المتبقية للـ Validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train_full
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# 5. PREPROCESSING - تجهيز البيانات
# ============================================================

def build_preprocessor(X):
    """إنشاء خطوات تجهيز البيانات الرقمية والنصية."""

    numerical_features = X.select_dtypes(
        include="number"
    ).columns.tolist()  # تحديد الأعمدة الرقمية

    categorical_features = X.select_dtypes(
        exclude="number"
    ).columns.tolist()  # تحديد الأعمدة النصية والفئوية

    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),  # تعويض القيم الرقمية المفقودة بالوسيط
        ("scaler", StandardScaler()),  # توحيد مقاييس الأعمدة الرقمية
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),  # تعويض القيم النصية الأكثر تكرارًا
        ("encoder", OneHotEncoder(handle_unknown="ignore")),  # تحويل الفئات إلى أعمدة رقمية
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical_features),  # تطبيق المعالجة على الأعمدة الرقمية
            ("categorical", categorical_pipeline, categorical_features),  # تطبيق المعالجة على الأعمدة النصية
        ]
    )

    return preprocessor


# ============================================================
# 6. MODEL BUILDING - بناء الموديل
# ============================================================

def build_model(preprocessor):
    """ربط تجهيز البيانات بموديل Logistic Regression."""

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),  # تجهيز البيانات قبل التدريب والتوقع
        ("classifier", LogisticRegression(
            class_weight="balanced",  # مراعاة عدم توازن الفئات
            max_iter=1000,  # الحد الأقصى لمحاولات الوصول إلى أفضل معاملات
        )),
    ])

    return model


def build_random_forest(preprocessor):
    """بناء موديل Random Forest وربطه بمعالج البيانات."""

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),  # تجهيز البيانات قبل التدريب
        ("classifier", RandomForestClassifier(
            n_estimators=300,  # عدد أشجار القرار المستخدمة
            max_depth=None,  # السماح للأشجار بالنمو دون حد أقصى للعمق
            min_samples_split=5,  # أقل عدد عينات مطلوب لتقسيم العقدة
            min_samples_leaf=2,  # أقل عدد عينات مسموح به في الورقة النهائية
            class_weight="balanced",  # مراعاة عدم توازن الفئات
            random_state=RANDOM_STATE,  # تثبيت العشوائية
            n_jobs=-1,  # استخدام أنوية المعالج المتاحة
        )),
    ])

    return model

# ============================================================
# 7. MODEL TRAINING - تدريب الموديل
# ============================================================

def train_model(model, X_train, y_train):
    """تدريب الموديل على بيانات التدريب فقط."""

    model.fit(X_train, y_train)  # تعلم المعالجة والأنماط من بيانات التدريب

    return model


# ============================================================
# 8. THRESHOLD EVALUATION - مقارنة عتبات التوقع
# ============================================================

def evaluate_thresholds(model, X_val, y_val):
    """مقارنة أداء الموديل عند استخدام Thresholds مختلفة."""

    probabilities = model.predict_proba(X_val)[:, 1]  # استخراج احتمال إلغاء الاشتراك لكل عميل

    results = []  # قائمة لتجميع نتائج كل Threshold

    for threshold in THRESHOLDS:  # تجربة كل عتبة محددة في الإعدادات

        predictions = (probabilities >= threshold).astype(int)  # تحويل الاحتمالات إلى توقعات 0 أو 1

        tn, fp, fn, tp = confusion_matrix(
            y_val,
            predictions,
            labels=[0, 1]
        ).ravel()  # استخراج أعداد TN وFP وFN وTP

        results.append({
            "Threshold": threshold,  # العتبة المستخدمة
            "Precision": precision_score(y_val, predictions, zero_division=0),  # دقة الإنذارات
            "Recall": recall_score(y_val, predictions, zero_division=0),  # نسبة العملاء المكتشفين
            "F1": f1_score(y_val, predictions, zero_division=0),  # التوازن بين Precision وRecall
            "False Positives": fp,  # عملاء توقعنا إلغاءهم لكنهم لم يلغوا
            "Missed Churners": fn,  # عملاء ألغوا الخدمة ولم يكتشفهم الموديل
            "Predicted Churners": tp + fp,  # إجمالي العملاء الذين توقع الموديل إلغاءهم
        })

    results_df = pd.DataFrame(results)  # تحويل النتائج إلى جدول

    print("\n========== THRESHOLD COMPARISON ==========")
    print(results_df.round(3).to_string(index=False))  # عرض النتائج بثلاث خانات عشرية

    return results_df


# ============================================================
# 9. FINAL MODEL EVALUATION - التقييم النهائي
# ============================================================

def evaluate_model(model, X_test, y_test, threshold):
    """التقييم النهائي باستخدام Threshold تم اختياره مسبقًا."""

    probabilities = model.predict_proba(X_test)[:, 1]  # استخراج احتمالات الإلغاء

    predictions = (probabilities >= threshold).astype(int)  # تطبيق العتبة المختارة

    print("\n========== FINAL TEST EVALUATION ==========")
    print(f"\nSelected Threshold: {threshold}")

    print("\nAccuracy:")
    print(accuracy_score(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nROC-AUC:")
    print(roc_auc_score(y_test, probabilities))


def compare_models(models, X_train, y_train, X_val, y_val):
    """تدريب الموديلات ومقارنة أدائها على Validation."""

    results = []

    for model_name, model in models.items():

        model.fit(X_train, y_train)  # تدريب الموديل على Training

        predictions = model.predict(X_val)  # توقع الفئات على Validation

        probabilities = model.predict_proba(X_val)[:, 1]  # استخراج احتمالات الإلغاء

        results.append({
            "Model": model_name,
            "Accuracy": accuracy_score(y_val, predictions),
            "Precision": precision_score(y_val, predictions, zero_division=0),
            "Recall": recall_score(y_val, predictions, zero_division=0),
            "F1": f1_score(y_val, predictions, zero_division=0),
            "ROC-AUC": roc_auc_score(y_val, probabilities),
        })

    results_df = pd.DataFrame(results)

    print("\n========== MODEL COMPARISON ==========")
    print(results_df.round(3).to_string(index=False))

    return results_df


def save_model(model, threshold, feature_columns):
    """حفظ الموديل وإعداداته في ملف واحد."""

    artifact = {
        "model": model,  # الموديل المدرب
        "threshold": threshold,  # العتبة المستخدمة في التوقع
        "feature_columns": feature_columns,  # أسماء وترتيب الخصائص المطلوبة
        "model_name": "Logistic Regression",  # اسم الموديل
    }

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )  # إنشاء مجلد models إذا لم يكن موجودًا

    joblib.dump(artifact, MODEL_PATH)  # حفظ الموديل والإعدادات

    print(f"\nModel saved successfully: {MODEL_PATH}")

# ============================================================
# 10. APPLICATION ENTRY POINT - نقطة تشغيل البرنامج
# ============================================================

def main():
    """تنفيذ التدريب والتقييم وحفظ الموديل النهائي."""

    df = load_data()

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    print("Training shape:", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("Test shape:", X_test.shape)

    selected_threshold = 0.5  # العتبة التي اخترناها بعد المقارنة

    # المرحلة الأولى: تدريب الموديل وتقييمه على Test
    evaluation_model = build_model(build_preprocessor(X_train))

    evaluation_model = train_model(
        evaluation_model,
        X_train,
        y_train
    )

    evaluate_model(
        evaluation_model,
        X_test,
        y_test,
        selected_threshold
    )

    # المرحلة الثانية: تجهيز بيانات التدريب النهائية
    X_final = pd.concat([X_train, X_val])  # دمج Training وValidation
    y_final = pd.concat([y_train, y_val])  # دمج النتائج المقابلة لهما

    # بناء موديل جديد وتدريبه على 80% من البيانات
    final_model = build_model(build_preprocessor(X_final))

    final_model = train_model(
        final_model,
        X_final,
        y_final
    )

    # المرحلة الثالثة: حفظ الموديل النهائي
    save_model(
        final_model,
        selected_threshold,
        X_final.columns.tolist()
    )


if __name__ == "__main__":
    main()