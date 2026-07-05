import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Import the 4 classifiers required by Phase 3
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# 1. Load and prepare dataset
print("Loading dataset...")
df = pd.read_csv('data/Flood_prediction.csv')

column_mapping = {
    'TopographyDrainage': 'Annual_Rainfall',
    'DrainageSystems': 'Cloud_Visibility',
    'MonsoonIntensity': 'Seasonal_Rainfall',
    'ClimateChange': 'Avg_Temperature',
    'CoastalVulnerability': 'Humidity',
    'AgriculturalPractices': 'Catchment_Area',
    'Siltation': 'River_Level'
}
df = df.rename(columns=column_mapping)
df['Flood_Risk'] = (df['FloodProbability'] >= 0.5).astype(int)

features = ['Annual_Rainfall', 'Cloud_Visibility', 'Seasonal_Rainfall', 
            'Avg_Temperature', 'Humidity', 'Catchment_Area', 'River_Level']
X = df[features]
y = df['Flood_Risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Initialize the 4 classifiers
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=50, max_depth=10, n_jobs=-1),
    "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
}

# 3. Train and compare accuracy
print("\n--- TRAINING AND COMPARING 4 CLASSIFIERS ---")
best_accuracy = 0
best_model_name = ""
best_model = None

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, predictions)
    print(f"{name} Accuracy: {acc:.4f}")
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_model_name = name
        best_model = model

print(f"\nWinner: {best_model_name} with {best_accuracy:.4f} accuracy!")

# 4. Save the winning model (XGBoost) and scaler
os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/Flood_model.joblib')
joblib.dump(scaler, 'models/scaler.joblib')
print(f"Successfully serialized and saved '{best_model_name}' to models/ folder!")