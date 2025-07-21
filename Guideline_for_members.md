# ML_Pipeline_assignment2
# TEAM.MD — Multimodal Data Preprocessing Assignment

## 📅 Assignment: Formative 2 — Multimodal Data Preprocessing

This assignment simulates a secure product recommendation system using face and voice authentication. Our team will develop and demonstrate a complete ML pipeline that includes tabular, image, and audio data processing.

---

## 👥 Team Members

* David Niyonshuti (@niyonshuti)
* \[Member 2 Name] (@github\_handle)
* \[Member 3 Name] (@github\_handle)
* \[Member 4 Name] (@github\_handle)

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

## 🔢 Step-by-Step Instructions

### 1. 📂 Data Merge (merge\_data.py)

```python
import pandas as pd

profiles = pd.read_csv('data/customer_social_profiles.csv')
transactions = pd.read_csv('data/customer_transactions.csv')

merged = pd.merge(profiles, transactions, on='customer_id')
merged.to_csv('data/merged_dataset.csv', index=False)
```

---

### 2. 📷 Image Data Processing (image\_processing.py)

#### Each team member must:

* Save 3 images (neutral, smiling, surprised) in `images/<your_name>/`

```bash
images/
├── david/
│   ├── neutral.jpg
│   ├── smiling.jpg
│   └── surprised.jpg
```

#### Example Code

```python
from PIL import Image
import os
import cv2
import numpy as np
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.Grayscale(),
    transforms.ToTensor()
])

def process_images(folder):
    features = []
    for file in os.listdir(folder):
        if file.endswith(".jpg"):
            path = os.path.join(folder, file)
            img = Image.open(path)
            img_t = transform(img)
            histogram = np.histogram(np.array(img.convert('L')).flatten(), bins=256)[0]
            features.append(histogram)
    return np.array(features)

features = process_images("images/david")
np.savetxt("data/image_features.csv", features, delimiter=",")
```

---

### 3. 🎤 Audio Processing (audio\_processing.py)

#### Each member records 2 samples: “Yes, approve” and “Confirm transaction”

```bash
audio/
├── david/
│   ├── approve.wav
│   └── confirm.wav
```

#### Example Code

```python
import librosa
import os
import numpy as np

def extract_features(path):
    y, sr = librosa.load(path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    energy = np.mean(librosa.feature.rms(y=y))
    return np.hstack([np.mean(mfcc, axis=1), np.mean(rolloff), energy])

def process_audio(folder):
    features = []
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            path = os.path.join(folder, file)
            feats = extract_features(path)
            features.append(feats)
    return np.array(features)

features = process_audio("audio/david")
np.savetxt("data/audio_features.csv", features, delimiter=",")
```

---

### 4. 📊 Model Training (model\_training.py)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

X = pd.read_csv("data/merged_dataset.csv").drop(columns=["target"])
y = pd.read_csv("data/merged_dataset.csv")["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
model = RandomForestClassifier()
model.fit(X_train, y_train)
preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print("F1 Score:", f1_score(y_test, preds, average='weighted'))
```

---

### 5. 🚀 CLI App Simulation (cli\_app.py)

```python
print("Welcome to the secure prediction system")
face_verified = input("Do you pass face recognition? (y/n): ")

if face_verified.lower() != 'y':
    print("Access Denied: Face not recognized")
    exit()

voice_verified = input("Do you pass voice verification? (y/n): ")

if voice_verified.lower() != 'y':
    print("Access Denied: Voice not verified")
    exit()

print("Access Granted. Displaying product prediction...")
# Call model.predict() here
```

---

### 📚 Contributions

| Name             | Contributions                         |
| ---------------- | ------------------------------------- |
| David Niyonshuti | Data merging, CLI simulation          |
| Member 2         | Image processing & feature extraction |
| Member 3         | Audio processing & augmentation       |
| Member 4         | Model training and evaluation         |

---

### 📥 Final Deliverables Checklist

* [x] Merged dataset
* [x] image\_features.csv
* [x] audio\_features.csv
* [x] All three model scripts
* [x] CLI app script
* [x] Jupyter notebook
* [x] Video demo
* [x] Report PDF
* [x] GitHub repo


