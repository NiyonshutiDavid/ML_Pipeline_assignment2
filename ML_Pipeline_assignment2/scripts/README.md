# Multi-Modal CLI App

This is a command-line application that performs facial recognition, product recommendation, and voice verification. It's built to show how multi-modal biometric verification works.

---

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

## How to run
From the scripts/ directory, run the CLI app:
```
python cli_app.py
```
or 
``` 
python3 cli_app.py
```
You'll be guided through 3 steps:

- Facial Recognition

- Product Recommendation

- Voice Verification

## Demo Instructions
You’ve got 4 sample files, two right files and two wrong files for both success and failure cases:

`Step 1`: Facial Recognition
Add the image path

`Step 2`: Product Recommendation
Automatically runs between face and voice steps, only if the Facial recognition have been passed. 

`Step 3`: Voice Verification
Type the path to your audio file

If Voice Verification is passed, the app will display the recommended product.
