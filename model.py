import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os

def train_model():
    # Load data
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    X_train = train_df.drop(columns=['Converted'])
    y_train = train_df['Converted']
    X_test = test_df.drop(columns=['Converted'])
    y_test = test_df['Converted']
    
    # Initialize and train model
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", report)
    
    # Feature Importance
    importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop Features:\n", importances.head(5))
    
    # Save model
    joblib.dump(model, 'lead_scoring_model.joblib')
    print("\nModel saved as lead_scoring_model.joblib")

if __name__ == "__main__":
    train_model()
