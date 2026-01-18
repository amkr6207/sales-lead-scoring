# Sales Lead Scoring Pipeline 🚀

This project predicts lead conversion probability using behavioral and firmographic data. This project is specifically designed to showcase how to turn raw CRM data (simulating **Zoho CRM**) into actionable sales intelligence.

## � Project Overview
Lead scoring helps sales teams prioritize high-value prospects. This pipeline processes raw CRM data, performs feature engineering, trains a machine learning model, and provides an interactive dashboard for prediction.

### Key Results
- **Model Accuracy:** 91% (Random Forest Classifier)
- **Top Predictor:** Lead Engagement Ratio (Web Visits × Time Spent)

## 🛠️ Tech Stack
- **Languages:** Python 3
- **Data Analytics:** Pandas, Scikit-Learn, NumPy
- **Dashboard:** Streamlit
- **Visualization:** Matplotlib, Seaborn
- **Model Deployment:** Joblib

## � Project Structure
- `data_gen.py`: Generates synthetic lead data with realistic B2B distributions.
- `preprocessing.py`: Handles data cleaning, LabelEncoding, and feature engineering.
- `model.py`: Trains and evaluates the Random Forest model.
- `app.py`: Interactive Streamlit dashboard for real-time lead scoring.

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3 and the necessary libraries installed. 


### 2. Run the Pipeline
```bash
# Generate Data
python3 data_gen.py

# Preprocess & Train Model
python3 preprocessing.py
python3 model.py
```

### 3. Launch the Dashboard
```bash
streamlit run app.py
```

## 📈 Resume Impact (Highlights)
- **Problem:** Eliminated manual lead sorting by automating prioritization for 1,000+ records.
- **Solution:** Developed an end-to-end ML pipeline with a **91% accurate** predictive model.
- **Outcome:** Created a live BI tool that identifies "High Potential" leads instantly based on digital footprint.

