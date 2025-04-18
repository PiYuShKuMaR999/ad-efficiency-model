import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

# === Load Data ===
data = pd.read_csv("D:\\ad-efficiency-model\\synthetic_digital_ad_data_enriched.csv")

# === Derived Metrics ===
data['ctr'] = data['clicks'] / data['impressions']
data['conversion_rate'] = data['conversions'] / data['clicks']
data.fillna(0, inplace=True)  # Handle any NaNs caused by division

# === Efficiency Score Calculation ===
data['efficiency_score'] = (
    0.4 * data['roi'] +
    0.2 * data['ctr'] +
    0.15 * data['conversion_rate'] +
    0.15 * data['engagement_score'] -
    0.1 * data['digital_waste']
)

# === Target Variable ===
data['run_ad'] = (data['efficiency_score'] > 1.0).astype(int)

# === Label Encoding ===
le_region = LabelEncoder()
le_platform = LabelEncoder()
data['region_encoded'] = le_region.fit_transform(data['region'])
data['platform_encoded'] = le_platform.fit_transform(data['platform'])

# Save class labels for future use
region_classes = le_region.classes_.tolist()
platform_classes = le_platform.classes_.tolist()

# === Feature Selection ===
features = [
    'ad_spend', 'impressions', 'clicks', 'conversions',
    'engagement_score', 'roi', 'digital_waste',
    'region_encoded', 'platform_encoded'
]
X = data[features]
y = data['run_ad']

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Model Training ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Evaluation ===
print("\n=== Model Report ===")
print(classification_report(y_test, model.predict(X_test)))

# === Save Model & Encoders ===
joblib.dump(model, 'ad_efficiency_model.pkl')
joblib.dump(le_region, 'le_region.pkl')
joblib.dump(le_platform, 'le_platform.pkl')
joblib.dump(region_classes, 'region_classes.pkl')
joblib.dump(platform_classes, 'platform_classes.pkl')

# Optional: Feature names used in the model
print("\nModel trained on features:", model.feature_names_in_)
