# Facial Expression Recognition (FER) - SNN Model

## Overview
This project implements a **Spiking Neural Network (SNN)** combined with CNN for real-time **Facial Expression Recognition**. It detects and classifies human emotions from facial expressions using a hybrid CNN-SNN architecture trained on the FER2013 dataset.

The model recognizes 7 emotion categories:
- 😠 **Angry**
- 😒 **Disgusted**
- 😰 **Fearful**
- 😊 **Happy**
- 😐 **Neutral**
- 😢 **Sad**
- 😲 **Surprised**

## Features
✨ **Real-time Detection** - Live emotion detection via webcam
🧠 **SNN-Based Architecture** - Energy-efficient spiking neural networks
📊 **Ensemble Prediction** - Multiple models for improved accuracy
🎨 **Color-Coded Risk Assessment** - Visual feedback with risk levels
⚡ **GPU Support** - CUDA acceleration for faster processing

## Project Structure
```
Facial Expression Recognition (FER)/
├── data/
│   └── raw/
│       ├── train/          # Training images (7 emotion folders)
│       └── test/           # Test images (7 emotion folders)
├── models/
│   ├── snn_model_1.pth    # Ensemble model 1
│   ├── snn_model_2.pth    # Ensemble model 2
│   └── snn_model_3.pth    # Ensemble model 3
├── train_snn.py           # Training script
├── test_snn.py            # Testing & validation script
├── realtime_detection.py  # Real-time webcam detection
├── utils.py               # CNN-SNN model architecture
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Requirements
- Python 3.8+
- PyTorch with CUDA support (optional but recommended)
- OpenCV
- Spiking Neural Network Torch (SNNTorch)

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd "Facial Expression Recognition (FER)"
```

### 2. Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### Dependencies:
- `torch` - PyTorch framework
- `torchvision` - Computer vision utilities
- `torchaudio` - Audio processing
- `snntorch` - Spiking Neural Networks
- `opencv-python` - Face detection & video capture
- `numpy` - Numerical computing
- `matplotlib` - Visualization
- `tqdm` - Progress bars

## Usage

### Option 1: Run Real-time Detection (Recommended)
```bash
python realtime_detection.py
```
**Controls:**
- Press `Q` to quit the detection window
- Webcam opens automatically and displays:
  - Detected faces with bounding boxes
  - Predicted emotion labels
  - Risk level assessment
  - Confidence scores

### Option 2: Train the Model
```bash
python train_snn.py
```
**Note:** Ensure training data is placed in `data/raw/train/` with subdirectories for each emotion.

### Option 3: Test Model Accuracy
```bash
python test_snn.py
```
Tests the trained models on test dataset and displays accuracy metrics.

## Color-Coded Risk Levels
The system provides visual risk assessment:
| Emotion | Risk Level | Color |
|---------|-----------|-------|
| 😢 Sad | High Risk | 🔴 Red |
| 😠 Angry | Medium Risk | 🟠 Orange |
| 😰 Fearful | Medium Risk | 🟠 Orange |
| 😐 Neutral | Low Risk | 🟢 Green |
| 😊 Happy | Very Low Risk | 🟢 Green |
| 😲 Surprised | Low Risk | 🟢 Green |
| 😒 Disgusted | Low Risk | 🟢 Green |

## Model Architecture
- **CNN Component:** Extracts spatial features from facial images
- **SNN Component:** Processes temporal dynamics using spiking neurons
- **Ensemble:** Combines predictions from 3 trained models for robust classification

## Performance
- Input size: 48×48 grayscale images
- Number of classes: 7 emotions
- Processing: GPU-accelerated (CUDA) or CPU fallback
- Real-time capability: 30+ FPS on modern GPUs

## Troubleshooting

### Webcam Not Detected
- Ensure your camera is properly connected
- Check if other applications are using the camera
- Update webcam drivers

### Low Detection Accuracy
- Ensure good lighting conditions
- Face should be clearly visible
- Try adjusting camera distance (30-60 cm optimal)

### Package Installation Issues
```bash
# Update pip
pip install --upgrade pip

# Reinstall requirements with verbose output
pip install -r requirements.txt -v
```

## Future Improvements
🔧 3D face pose correction
🔧 Multiple face detection
🔧 Emotion transition tracking
🔧 Real-time emotion statistics
🔧 Export results to CSV/JSON
🔧 Mobile deployment support

## License
MIT License - Feel free to use this project for research and educational purposes.

## Author
Developed for Facial Expression Recognition research project.

## References
- [SNNTorch Documentation](https://snntorch.readthedocs.io/)
- [PyTorch Official](https://pytorch.org/)
- [FER2013 Dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)

## Contact & Support
For issues, questions, or contributions, please open an issue on the GitHub repository.

