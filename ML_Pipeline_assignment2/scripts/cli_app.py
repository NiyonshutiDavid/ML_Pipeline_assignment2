import os
import time
from PIL import Image
import matplotlib.pyplot as plt
import librosa
from image_processing import process_image
from audio_processing import extract_features as extract_audio
from prediction_helpers import predict_face, predict_voice, predict_product

def loader(message="Processing", dots=3, delay=0.5):
    print(message, end="", flush=True)
    for _ in range(dots):
        time.sleep(delay)
        print(".", end="", flush=True)
    print()

def run():
    print("\n===== WELCOME TO THE MULTI-MODAL CLI =====\n")

    # === STEP 1: Facial Auth ===
    while True:
        print("\n----- STEP 1: Facial Recognition -----")
        img_path = input("Enter path to image: ").strip()
        if img_path == "":
            print("ACCESS DENIED: No image provided.")
            continue

        if not os.path.exists(img_path):
            print("Invalid path: File not found.")
            continue

        try:
            loader("Scanning face")
            img_feats = process_image(img_path)
        except Exception as e:
            print("Invalid image: Please provide a valid image file.")
            print("Error:", e)
            continue

        face_conf = predict_face(img_feats)

        if face_conf < 50:
            print(f"ACCESS DENIED: Face not recognized ({face_conf:.2f}%)")
            continue

        print(f"Facial Recognition Passed ({face_conf:.2f}%)")
        break

    # === STEP 2: Product Recommendation ===
    print("\n----- STEP 2: Product Recommendation -----")
    loader("Generating recommendation")
    product = predict_product()
    print(f"Product recommendation ready")

    # === STEP 3: Voice Verification ===
    print("\n----- STEP 3: Voice Verification -----")
    while True:
        audio_path = input("Enter path to audio: ").strip()
        if not os.path.exists(audio_path):
            print("Audio file not found. Try again.")
            continue

        loader("Verifying voice")
        try:
            y, sr = librosa.load(audio_path, sr=None)
            break  # exit loop if successful
        except Exception as e:
            print(f"[!] Failed to load audio: {type(e).__name__} - {str(e) or 'No details provided'}")
            continue

    audio_feats = extract_audio(y, sr)
    voice_conf = predict_voice(audio_feats)

    if voice_conf < 50:
        print("ACCESS DENIED: Voice not verified.")
        return

    print(f"Voice Verified ({voice_conf:.2f}%)")
    print("\n===== ACCESS GRANTED =====")
    print(f"Recommended Product: {product}")

if __name__ == "__main__":
    run()
