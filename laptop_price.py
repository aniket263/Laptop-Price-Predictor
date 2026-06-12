import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("data.csv")

# Clean RAM column
df["Ram"] = df["Ram"].replace({
    "2GB": 2,
    "4GB": 4,
    "8GB": 8,
    "12GB": 12,
    "16GB": 16,
    "32GB": 32,
    "64GB": 64
})

# Clean ROM column
df["ROM"] = df["ROM"].replace({
    "32GB": 32,
    "64GB": 64,
    "128GB": 128,
    "256GB": 256,
    "512GB": 512,
    "1TB": 1024,
    "2TB": 2048
})

# Convert to numeric
df["Ram"] = pd.to_numeric(df["Ram"])
df["ROM"] = pd.to_numeric(df["ROM"])

# Remove unwanted columns
df = df.drop(columns=["Unnamed: 0.1", "Unnamed: 0"])

# Features and target
X = df.drop(columns=["price"])
y = df["price"]

# Categorical columns
categorical_features = [
    "brand",
    "name",
    "processor",
    "CPU",
    "Ram_type",
    "ROM_type",
    "GPU",
    "OS"
]

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder="passthrough"
)

# Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Score
score = r2_score(y_test, y_pred)
print("R2 Score:", score)

# Save model
joblib.dump(pipeline, "laptop_model.pkl")

print("Model Saved Successfully!")