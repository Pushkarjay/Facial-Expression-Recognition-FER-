# Emotion Detection using Spiking Neural Network (SNN)

## Overview
This project trains a **Spiking Neural Network (SNN)** using the FER dataset to detect facial emotions in real time with OpenCV.

## Run Steps
1. Place dataset inside `data/raw/train/emotion` and `data/raw/test/emotion`.
2. Install dependencies:
   pip install -r requirements.txt
3. Train model: 
   python train_snn.py
4. Test accuracy:
   python test_snn.py
5. Run real-time detection:
   python realtime_detection.py

Press Q to quit live webcam.

