import joblib
import numpy as np
import pandas as pd

# === Load all models and tools ===
facial_model = joblib.load('../models/best_models/facial_recognition_model.pkl')
voice_model = joblib.load('../models/best_models/voiceprint_verification_model.pkl')
product_model = joblib.load('../models/best_models/product_recommendation_model.pkl')
scaler = joblib.load('../models/best_models/scaler.pkl')
encoder = joblib.load('../models/best_models/label_encoder.pkl')

def predict_face(features):
    feature_names = [f'feat_{i}' for i in range(96)] 
    df = pd.DataFrame(features, columns=feature_names)
    return facial_model.predict_proba(df)[0][1] * 100

def predict_voice(features):
    df = pd.DataFrame([features]) 
    return voice_model.predict_proba(df)[0][1] * 100

def predict_product():
    # Example input data with hardcoded values
    input_data = {
        'purchase_interest_score_mean': 7.5,
        'purchase_interest_score': 8.1,
        'purchase_amount': 49.99,
        'purchase_amount_mean': 45.20,
        'is_weekend': 0,
        'purchase_month': 6,
        'engagement_score': 7.2,
        'purchase_amount_max': 59.99,
        'purchase_day_of_month': 15,
        'customer_rating_mean': 4.3,
        'engagement_interest_interaction': 58.32,
        'purchase_quarter': 2,
        'purchase_day_of_week': 2,
        'customer_rating_max': 5.0,
        'purchase_amount_min': 32.50,
        'customer_rating_min': 3.5,
        'purchase_amount_std': 12.34,
        'customer_rating': 4.5,
        'engagement_score_std': 1.2,
        'purchase_amount_sum': 226.00,
        'amount_rating_interaction': 224.955,
        'sentiment_numeric': 2,
        'purchase_interest_score_std': 1.0,
        'engagement_score_mean': 6.8,
        'engagement_score_max': 8.5,
        'amount_per_engagement': 6.94
    }

    # Expected order of features
    expected_features = [
        'purchase_interest_score_mean', 'purchase_interest_score',
        'purchase_amount', 'purchase_amount_mean', 'is_weekend', 'purchase_month',
        'engagement_score', 'purchase_amount_max', 'purchase_day_of_month',
        'customer_rating_mean', 'engagement_interest_interaction',
        'purchase_quarter', 'purchase_day_of_week', 'customer_rating_max',
        'purchase_amount_min', 'customer_rating_min', 'purchase_amount_std',
        'customer_rating', 'engagement_score_std', 'purchase_amount_sum',
        'amount_rating_interaction', 'sentiment_numeric',
        'purchase_interest_score_std', 'engagement_score_mean',
        'engagement_score_max', 'amount_per_engagement'
    ]

    # Fill missing keys with 0
    for feat in expected_features:
        if feat not in input_data:
            input_data[feat] = 0

    # Create DataFrame with the exact column order
    X = pd.DataFrame([input_data], columns=expected_features)
    X_scaled = scaler.transform(X)

    pred = product_model.predict(X_scaled)
    return encoder.inverse_transform(pred)[0]
