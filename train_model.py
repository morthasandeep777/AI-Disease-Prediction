import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("medical_dataset_1500_records_10_diseases.csv")

print(df.head())

# -----------------------------
# Separate Features and Target
# -----------------------------
X = df.drop("Disease", axis=1)
y = df["Disease"]

# -----------------------------
# Convert Disease Names to Numbers
# -----------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Build Model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Test Accuracy
# -----------------------------
prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("\nModel Accuracy :", round(accuracy*100,2), "%")

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(model, "disease_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("\nModel Saved Successfully!")