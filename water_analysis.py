import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data_path = r"E:\dataset.csv"
df = pd.read_csv(data_path)

# Handle missing values (impute with median)
df.fillna(df.median(), inplace=True)

# Features and target
X = df.drop(columns=['Potability'])  # Features
y = df['Potability']  # Target

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# Feature Importance for Recommendations
feature_importance = pd.Series(model.feature_importances_, index=df.drop(columns=['Potability']).columns)
feature_importance = feature_importance.sort_values(ascending=False)
print("\nFeature Importance:\n", feature_importance)

# Generate recommendations based on feature importance
def generate_recommendations(row):
    recs = []
    if row['ph'] < 6.5 or row['ph'] > 8.5:
        recs.append("Adjust pH using chemicals like lime (for low pH) or acids (for high pH).")
    if row['Hardness'] > 150:
        recs.append("Use water softeners to reduce hardness.")
    if row['Solids'] > 500:
        recs.append("Consider reverse osmosis or filtration to remove excessive solids.")
    if row['Chloramines'] > 4:
        recs.append("Use activated carbon filters to reduce Chloramines.")
    if row['Sulfate'] > 250:
        recs.append("Reduce sulfate levels using distillation or ion exchange.")
    if row['Organic_carbon'] > 5:
        recs.append("Use biological treatment to reduce organic carbon content.")
    return recs if recs else ["Water quality is good; no major treatment needed."]

# Example recommendation for first row
sample_row = df.drop(columns=['Potability']).iloc[0]
recommendations = generate_recommendations(sample_row)
print("\nRecommendations:")
for rec in recommendations:
    print("-", rec)
