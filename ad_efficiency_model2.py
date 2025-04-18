import numpy as np
import pandas as pd

import joblib

# Load model and encoders
model = joblib.load('ad_efficiency_model.pkl')
le_region = joblib.load('le_region.pkl')
le_platform = joblib.load('le_platform.pkl')
region_classes = joblib.load('region_classes.pkl')
platform_classes = joblib.load('platform_classes.pkl')

# Print expected feature names (debugging)
print(model.feature_names_in_)  
print("\n=== Ad Efficiency Checker ===\n")

# Region
print("Select a region:")
for i, r in enumerate(region_classes, 1):
    print(f"{i}. {r}")
region_idx = int(input("Enter region number: ")) - 1
region = region_classes[region_idx]

# Platform
print("\nSelect a platform:")
for i, p in enumerate(platform_classes, 1):
    print(f"{i}. {p}")
platform_idx = int(input("Enter platform number: ")) - 1
platform = platform_classes[platform_idx]

# Encode region and platform
region_encoded_value = le_region.transform([region])[0]
platform_encoded_value = le_platform.transform([platform])[0]



# Campaign data input
ad_spend = float(input("\nEnter ad spend ($): "))
impressions = int(input("Enter number of impressions: "))
clicks = int(input("Enter number of clicks: "))
conversions = int(input("Enter number of conversions: "))
engagement_score = float(input("Enter engagement score (0-1): "))
digital_waste = float(input("Enter digital waste (0-1): "))
roi = float(input("Enter ROI (e.g., 1.2 = 120%): "))

# Prepare input as a DataFrame with feature names
user_input = pd.DataFrame([{
    'region_encoded': region_encoded_value,
    'platform_encoded': platform_encoded_value,
    'ad_spend': ad_spend,
    'impressions': impressions,
    'clicks': clicks,
    'conversions': conversions,
    'engagement_score': engagement_score,
    'digital_waste': digital_waste,
    'roi': roi
}])

# Ensure feature order matches training
feature_order = [
    'ad_spend', 'impressions', 'clicks', 'conversions',
    'engagement_score', 'roi', 'digital_waste',
    'region_encoded', 'platform_encoded'
]
user_input = user_input[feature_order]


# Predict
prediction = model.predict(user_input)[0]
score = model.predict_proba(user_input)[0][1]

print("\n--- RESULT ---")
print(f"Predicted Efficiency Score: {round(score * 100, 2)}%")
if score * 100 >= 60.0:
    print("✅ Run the ad.")
else:
    print("❌ Do not run the ad.")
