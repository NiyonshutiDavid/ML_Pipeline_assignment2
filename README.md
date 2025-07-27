# ML_Pipeline_assignment2
# Multimodal Data Preprocessing Assignment

## 📅 Assignment: Formative 2 — Multimodal Data Preprocessing

This assignment simulates a secure product recommendation system using face and voice authentication. Our team will develop and demonstrate a complete ML pipeline that includes tabular, image, and audio data processing.

---

## 👥 Team Members

* David Niyonshuti
* Nicolle Marizani
* Tamanda Kaunda
* Annabelle Ineza
* Chance Karambizi

---

## 📁 Project Structure

```
├── data/
│   ├── customer_social_profiles.csv
│   ├── customer_transactions.csv
│   ├── merged_dataset.csv
│   ├── image_features.csv
│   └── audio_features.csv
├── images/
│   └── [member_name]/[image_files]
├── audio/
│   └── [member_name]/[audio_files]
├── notebooks/
│   └── data_processing.ipynb
├── scripts/
│   ├── merge_data.py
│   ├── image_processing.py
│   ├── audio_processing.py
│   ├── model_training.py
│   └── cli_app.py
├── team.md
└── README.md
```

---

## 🛠️ Project Setup Instructions

> 📝 Prerequisites: Python 3.8+, `venv` or `conda`, Jupyter Lab/Notebook

### 1. Clone the repository

```bash
git clone https://github.com/NiyonshutiDavid/ML_Pipeline_assignment2.git
```
### 2. Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
```
### 3. Install dependencies
```bash
pip install numpy pandas matplotlib librosa scikit-image pillow
```
---
### 🚀 Run the Application
Once everything is set up, you only need to run the CLI tool to simulate:
```bash
python cli_app.py
```
The CLI app will:
- Allow you to paste the path of images and audio to see if you're allowed to access recommendations 

‼️ You do not need to run the notebooks unless you want to explore or modify the feature extraction logic.

### 📚 Contributions

| Name             | Contributions                         |
| ---------------- | ------------------------------------- |
| David Niyonshuti | Image and Audio processing            |
| Nicolle Marizani | Data merge and cleaning for recommendations  |
| Annabelle Ineza  | Model training and evaluation        |
| Chance KARAMBIZI | CLI simulation                        |
| Tamanda Kaunda   | Model training and evaluation         |




