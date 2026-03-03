import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("anemia.csv")

# Features & Target
X = df.drop("Result", axis=1)
y = df["Result"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Save model + scaler
joblib.dump({
    "model": model,
    "scaler": scaler
}, "anemia_model.pkl")

print("Model saved as anemia_model.pkl")

# Save predictions
predictions = model.predict(X_test)
results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions
})
results.to_csv("anemia_model_results.csv", index=False)

print("Prediction file saved as anemia_model_results.csv")