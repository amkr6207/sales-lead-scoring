import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def preprocess_data(input_path='leads.csv'):
    df = pd.read_csv(input_path)
    
    # Drop unique identifiers
    df_clean = df.drop(columns=['Lead_ID'])
    
    # Feature Engineering: Engagement Ratio
    # (Total Visits * Time) / (Industry Average Time) - simulated as a ratio
    df_clean['Engagement_Ratio'] = (df_clean['Total_Web_Visits'] * df_clean['Avg_Time_Per_Visit']) / 50
    
    # Encoding categorical variables
    categorical_cols = ['Lead_Source', 'Industry', 'Company_Size', 'Last_Activity']
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col])
        encoders[col] = le
        
    # Save encoders for the dashboard
    joblib.dump(encoders, 'encoders.joblib')
    
    # Split data
    X = df_clean.drop(columns=['Converted'])
    y = df_clean['Converted']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Save split data for model training
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    train_data.to_csv('train.csv', index=False)
    test_data.to_csv('test.csv', index=False)
    
    print("Preprocessing complete. Train/Test data and encoders saved.")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocess_data()
