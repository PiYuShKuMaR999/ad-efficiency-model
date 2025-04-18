# Ad Efficiency Model 🚀

This project predicts whether a digital ad campaign is worth running based on ROI, engagement, CTR, conversion rates, and digital waste. It uses a machine learning model trained on enriched synthetic ad campaign data.

---

## 📊 Project Overview

- **Data Source**: Synthetic enriched CSV data
- **Model Used**: Random Forest Classifier
- **Target Variable**: `run_ad` (1 = Run the ad, 0 = Do not run)
- **Main Goal**: Help marketers make ROI-based decisions on whether to launch a digital ad

---

## 🛠️ Features

- Calculates **CTR**, **Conversion Rate**, and an **Efficiency Score**
- Trains an ML model to predict whether an ad should be run
- Supports CLI-based user input for ad prediction
- Saves the model and encoders using `joblib` for reuse

---

## 📁 Project Structure

├── model_training.py # Trains the model and saves it 
├── predict_ad.py # CLI interface to make predictions 
├── synthetic_digital_ad_data_enriched.csv # Data file 
├── ad_efficiency_model.pkl # Trained model 
├── le_region.pkl # Region encoder 
├── le_platform.pkl # Platform encoder 
├── README.md # Project documentation


---

## 💡 How It Works

1. **Feature Engineering**:  
   - CTR = clicks / impressions  
   - Conversion Rate = conversions / clicks  
   - Efficiency Score = weighted formula combining ROI, CTR, conversions, engagement, and digital waste

2. **Target Generation**:  
   - If `efficiency_score > 1.0` → label as 1 (Run Ad)

3. **Model Training**:  
   - Uses `RandomForestClassifier` from `sklearn`

4. **Prediction Script**:  
   - User selects region/platform and inputs campaign stats  
   - Model returns recommendation and probability

---

## 🧪 Example Usage

```bash
$ python predict_ad.py

Select a region:
1. North America
2. Europe
...

Enter ad spend ($): 500
Enter number of impressions: 10000
Enter number of clicks: 450
...
✅ Run the ad.


🛠 Requirements
Python 3.7+
pandas
numpy
scikit-learn
joblib

Install with:
pip install -r requirements.txt


🔮 Future Improvements
Add a simple web dashboard (Streamlit or Flask)
Improve model tuning and cross-validation
Upload Power BI visualizations for performance



📬 Contact
Made with ❤️ by Piyush Kumar
📧 piyush990841@gmail.com







