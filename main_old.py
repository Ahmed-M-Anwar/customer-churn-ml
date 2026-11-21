import pandas as pd

url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

df = pd.read_csv(url)

# print(df.head())
# print(df.shape)
# print(df.info())
# print(df.describe())

# ############################################################################################


# print("TotalCharges data type:")
# print(df["TotalCharges"].dtype)

# print("\nUnique values sample:")
# print(df["TotalCharges"].unique()[:20])

# print("\nMissing values:")
# print(df.isnull().sum())

# print("\nChurn distribution:")
# print(df["Churn"].value_counts())


# ############################################################################################


# # Count empty or whitespace-only values
# empty_values = df["TotalCharges"].str.strip().eq("")

# print("Empty values:", empty_values.sum())

# # Display affected rows
# print(df.loc[empty_values, ["customerID", "tenure", "MonthlyCharges", "TotalCharges"]])

# ############################################################################################

# Convert TotalCharges from text to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Replace missing values with 0
df["TotalCharges"] = df["TotalCharges"].fillna(0)


# ############################################################################################

# print(df["TotalCharges"].dtype)
# print(df["TotalCharges"].isnull().sum())
# print(df.loc[df["tenure"] == 0, ["tenure", "MonthlyCharges", "TotalCharges"]])


############################################################################################

# print(
#     pd.crosstab(
#         df["Contract"],
#         df["Churn"]
#     )
# )

# print(
#     pd.crosstab(
#         df["Contract"],
#         df["Churn"],
#         normalize="index"
#     ) * 100
# )

#############################################################################################

# import matplotlib.pyplot as plt

# churn_rate = (
#     pd.crosstab(
#         df["Contract"],
#         df["Churn"],
#         normalize="index"
#     )["Yes"] * 100
# )

# churn_rate.plot(kind="bar", figsize=(8, 5))

# plt.title("Churn Rate by Contract Type")
# plt.xlabel("Contract Type")
# plt.ylabel("Churn Rate (%)")
# plt.xticks(rotation=0)
# plt.tight_layout()
# plt.show()


##############################################################################################

# df["tenure_group"] = pd.cut(
#     df["tenure"],
#     bins=[-1, 12, 24, 48, 72],
#     labels=["0-12", "13-24", "25-48", "49-72"]
# )

# tenure_churn = pd.crosstab(
#     df["tenure_group"],
#     df["Churn"],
#     normalize="index"
# ) * 100

# print(tenure_churn)

#############################################################################################

from sklearn.model_selection import train_test_split  # تقسيم البيانات إلى تدريب واختبار

# Separate features and target
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,                  # Features: خصائص العملاء التي سيتعلم منها الموديل
    y,                  # Target: النتيجة المطلوب توقعها (Churn)
    test_size=0.2,      # تخصيص 20% للاختبار و80% للتدريب
    random_state=42,    # تثبيت العشوائية للحصول على نفس التقسيم عند إعادة التشغيل
    stratify=y          # الحفاظ على نسب الفئات في مجموعتي التدريب والاختبار
)
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True))
