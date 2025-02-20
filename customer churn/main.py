import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Check for null values
df.drop(["customerID"], axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Encode categorical variables
le = LabelEncoder()
for column in df.select_dtypes(include=["object"]).columns:
    df[column] = le.fit_transform(df[column])

# Define Features and Target Variable
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Check class imbalance
print("Original Class Distribution:", Counter(y))

# Apply SMOTE (oversampling) & RandomUnderSampler (undersampling)
smote = SMOTE(sampling_strategy=0.6, random_state=42)  # Oversample minority class
undersample = RandomUnderSampler(sampling_strategy=0.8, random_state=42)  # Reduce majority class

X_resampled, y_resampled = smote.fit_resample(X, y)
X_resampled, y_resampled = undersample.fit_resample(X_resampled, y_resampled)

# Check new class distribution
print("Resampled Class Distribution:", Counter(y_resampled))

# Split into Train & Test Sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Feature Scaling (Apply only AFTER resampling & splitting)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict on Test Set
y_pred = model.predict(X_test_scaled)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("\n✅ Accuracy:", accuracy)
report = classification_report(y_test, y_pred)
print("\n📊 Classification Report:\n", report)

# Confusion Matrix Visualization
plt.figure(figsize=(5, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="coolwarm", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
