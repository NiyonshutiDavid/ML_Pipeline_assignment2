# Multi-Modal System Testing Guide

This directory contains comprehensive testing scripts for the Multi-Modal Authentication and Product Recommendation System.

## Overview

The system implements the workflow shown in the system diagram:

1. **Start** → **Facial Recognition Model** (Authentication)
2. **Facial Recognition Success** → **Product Recommendation Model**
3. **Product Recommendation** → **Voice Validation Model** (Final verification)
4. **All Success** → **Display Predicted Product**
5. **Any Failure** → **Access Denied**

## Models Included

The three models: **Facial Recognition**, **Product Recommendation** and **Voice Validation**.

## Expected Model Files

After running the updated `Models.ipynb`, you should have:

```
best_models/
├── facial_recognition_model.pkl
├── facial_recognition_metadata.txt
├── voiceprint_verification_model.pkl
├── voiceprint_verification_metadata.txt
├── product_recommendation_model.pkl
├── scaler.pkl
├── label_encoder.pkl
└── model_metadata.txt
```

## System Workflow

### Step 1: Facial Recognition Authentication

- **Input**: Image features from `image_features.csv`
- **Process**: Facial recognition model identifies user
- **Success Criteria**: Confidence ≥ 50%
- **On Failure**: → **ACCESS DENIED**

### Step 2: Product Recommendation

- **Input**: Customer behavioral data from `cleaned_merged_dataset.csv`
- **Process**: ML model recommends product category
- **Success Criteria**: Model execution successful
- **Dependencies**: Requires Step 1 success

### Step 3: Voice Validation

- **Input**: Audio features from `audio_features.csv`
- **Process**: Voice verification model confirms user identity
- **Success Criteria**: Confidence ≥ 50%
- **On Failure**: → **ACCESS DENIED**

### Step 4: Final Result

- **All Success**: Display recommended product
- **Any Failure**: Access denied with specific reason

## Running the Tests

### Prerequisites

1. Ensure all models are trained by running `Models.ipynb`
2. Verify data files exist:
   - `image_features.csv`
   - `audio_features.csv`
   - `../data/cleaned_merged_dataset.csv`

## Troubleshooting

### Common Issues

1. **Model files not found**

   - Run `Models.ipynb` completely to generate all models
   - Check that `best_models/` directory exists

2. **Data files missing**

   - Ensure `image_features.csv` and `audio_features.csv` are in models directory
   - Verify `cleaned_merged_dataset.csv` is in `../data/` directory

3. **Low authentication confidence**
   - This is expected with dummy/test data
   - Use real customer data for production accuracy

## Security Notes

- **Authentication Thresholds**: Currently set to 50% confidence
- **Multi-Factor**: System requires both facial AND voice verification
- **Fail-Safe**: Any single failure results in complete access denial
- **Audit Trail**: All decisions are logged with confidence scores

