import pickle
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from transformers import pipeline

# === Simulated hardcoded secret ===
API_KEY = "abc123-insecure-api-key"

# === Load dataset and train model ===
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

clf = RandomForestClassifier()
clf.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("[INFO] Model saved to model.pkl")

# === Insecure pickle load ===
model_path = input("Enter model path to load (e.g., model.pkl): ")
with open(model_path, "rb") as f:
    model = pickle.load(f)  # INSECURE

# === eval() of config ===
model_config = input("Enter model config dict (e.g., {'n_estimators': 10}): ")
try:
    config_dict = eval(model_config)  # INSECURE
    print("[DEBUG] Loaded config:", config_dict)
except Exception as e:
    print("[ERROR] eval failed:", e)

# === Log secret key ===
print(f"[DEBUG] API_KEY used: {API_KEY}")

# === Hugging Face Integration ===
sentiment = pipeline("sentiment-analysis")
result = sentiment("This project contains intentional security issues.")
print("[DEBUG] HuggingFace sentiment:", result)

# === Predict with model ===
sample = [[5.1, 3.5, 1.4, 0.2]]
print("Prediction:", model.predict(sample))
