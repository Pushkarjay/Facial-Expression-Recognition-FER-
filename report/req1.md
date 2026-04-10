# System Requirements & Design - Facial Expression Recognition (FER)

---

## 1. Project Overview & Objectives

### Project Title:
**Facial Expression Recognition System using Spiking Neural Networks (SNN) and Convolutional Neural Networks (CNN)**

### Objectives:
- Develop an energy-efficient real-time facial expression recognition system
- Implement hybrid CNN-SNN architecture for temporal emotion processing
- Create ensemble-based prediction system for improved robustness
- Enable real-time processing from webcam video streams
- Classify emotions into 7 categories with risk assessment

### Target Users:
- Mental health professionals
- Human-computer interaction researchers
- Customer satisfaction monitoring systems
- Educational institutions

---

## 2. Functional Requirements

### FR1: Data Input & Preprocessing
- **Requirement**: Accept facial images or video streams as input
- **Implementation**:
  - Webcam video stream capture using OpenCV
  - Automatic face detection using Haar Cascade classifier
  - Image resizing to 48×48 pixels (standardized format)
  - Grayscale conversion for model compatibility
  - Real-time preprocessing pipeline

### FR2: Emotion Classification
- **Requirement**: Classify detected faces into 7 emotion categories
- **Emotion Categories**:
  1. Angry 😠
  2. Disgusted 😒
  3. Fearful 😰
  4. Happy 😊
  5. Neutral 😐
  6. Sad 😢
  7. Surprised 😲
- **Output**: Predicted emotion with confidence score

### FR3: Ensemble Prediction
- **Requirement**: Use multiple models for robust predictions
- **Implementation**:
  - 3 trained SNN-CNN hybrid models
  - Majority voting mechanism
  - Confidence threshold filtering
  - Weighted averaging of predictions

### FR4: Risk Assessment
- **Requirement**: Calculate and display risk levels based on emotion
- **Risk Levels**:
  - 🔴 **HIGH RISK**: Sad emotion (requires immediate attention)
  - 🟠 **MEDIUM RISK**: Angry, Fearful emotions
  - 🟢 **LOW RISK**: Happy, Neutral, Disgusted, Surprised
- **Output**: Color-coded visual feedback

### FR5: Real-time Detection
- **Requirement**: Process video frames in real-time
- **Performance Metrics**:
  - Minimum 15 FPS for smooth operation
  - Inference latency < 100ms per frame
  - GPU acceleration support

### FR6: Model Training
- **Requirement**: Train models on FER2013 dataset
- **Implementation**:
  - Data loading from directory structure
  - Training/validation/test split
  - Loss computation and backpropagation
  - Model checkpoint saving
  - Ensemble model creation (3 models)

### FR7: Testing & Validation
- **Requirement**: Comprehensive model validation
- **Metrics**:
  - Overall accuracy
  - Per-class precision, recall, F1-score
  - Confusion matrix
  - Cross-validation results

---

## 3. Non-Functional Requirements

### NFR1: Performance
- **Inference Time**: < 100ms per frame (GPU)
- **Frame Rate**: Minimum 15 FPS for real-time processing
- **Memory Usage**: < 2GB RAM during operation
- **Model Size**: < 50MB per model

### NFR2: Scalability
- **Multi-face Detection**: Support detection of multiple faces in single frame
- **Batch Processing**: Capability for offline batch prediction
- **Model Extension**: Easy addition of new emotion categories

### NFR3: Reliability
- **Model Accuracy**: Target minimum 75% overall accuracy
- **Robustness**: Handle partial occlusion, varying lighting conditions
- **Error Handling**: Graceful handling of edge cases

### NFR4: Usability
- **User Interface**: Simple, intuitive real-time display
- **Controls**: Keyboard shortcuts (Q to quit)
- **Feedback**: Real-time visualization of predictions and risk levels

### NFR5: Portability
- **Platform Support**: Windows, Linux, macOS
- **GPU Support**: CUDA for NVIDIA GPUs
- **CPU Fallback**: Functional on CPU-only systems (slower)

### NFR6: Maintainability
- **Code Quality**: Well-structured, documented code
- **Modular Design**: Separate training, testing, and inference modules
- **Version Control**: Git-based version management

---

## 4. System Architecture

### Architecture Diagram:
```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT LAYER                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Video Stream (Webcam) OR Static Images              │   │
│  └─────────────────────────┬──────────────────────────┘    │
└────────────────────────────┼──────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                 PREPROCESSING LAYER                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ • Face Detection (Haar Cascade)                      │   │
│  │ • Image Resize (48×48)                              │   │
│  │ • Grayscale Conversion                              │   │
│  │ • Normalization                                     │   │
│  └─────────────────────────┬──────────────────────────┘    │
└────────────────────────────┼──────────────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │     FEATURE EXTRACTION (CNN)           │
        │  ┌────────────────────────────────┐    │
        │  │ Conv Layer 1 (32 filters)      │    │
        │  │ MaxPool (2×2)                  │    │
        │  │ Conv Layer 2 (64 filters)      │    │
        │  │ MaxPool (2×2)                  │    │
        │  │ Output: 64×12×12 features      │    │
        │  └─────────────────┬──────────────┘    │
        │                    │                   │
        │     ┌──────────────┴──────────────┐    │
        │     │                             │    │
        └─────┼──────────────┬──────────────┘    │
              │              │                   │
              ▼              ▼                   │
    ┌──────────────────┬──────────────────┐    │
    │    SNN MODEL 1   │   SNN MODEL 2    │    │
    │  ┌────────────┐  │  ┌────────────┐  │    │
    │  │ Temporal   │  │  │ Temporal   │  │    │
    │  │ Processing │  │  │ Processing │  │    │
    │  │ (25 steps) │  │  │ (25 steps) │  │    │
    │  └─────┬──────┘  │  └─────┬──────┘  │    │
    │        │         │        │         │    │
    │        └─────────┼────────┘         │    │
    │                  │                  │    │
    └──────────────────┼──────────────────┘    │
                       │                        │
                       ▼                        │
    ┌──────────────────────────────────────┐   │
    │       SNN MODEL 3                     │   │
    │    ┌──────────────────────────────┐   │   │
    │    │   Temporal Processing        │   │   │
    │    │   (25 timesteps)            │   │   │
    │    └──────────────┬───────────────┘   │   │
    │                   │                   │   │
    └───────────────────┼───────────────────┘   │
                        │                        │
                        ▼                        │
         ┌──────────────────────────────┐       │
         │   ENSEMBLE VOTING LAYER      │       │
         │  • Majority Voting           │       │
         │  • Confidence Threshold      │       │
         │  • Final Prediction          │       │
         └──────────┬───────────────────┘       │
                    │                           │
                    ▼                           │
         ┌──────────────────────────────┐       │
         │  EMOTION CLASSIFICATION      │       │
         │  Output: Emotion + Score     │       │
         └──────────┬───────────────────┘       │
                    │                           │
                    ▼                           │
         ┌──────────────────────────────┐       │
         │  RISK ASSESSMENT             │       │
         │  🔴🟠🟢 Risk Level Mapping   │       │
         └──────────┬───────────────────┘       │
                    │                           │
                    ▼                           │
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT LAYER                             │
│  • Display emotion with bounding box                        │
│  • Show confidence score                                    │
│  • Display risk level (color-coded)                         │
│  • Real-time FPS and latency metrics                        │
└─────────────────────────────────────────────────────────────┘
```

### Component Architecture:

#### 1. **Data Pipeline**
```python
Raw Image/Video Stream
    ↓
[Face Detection] → Haar Cascade Classifier
    ↓
[Preprocessing] → Resize, Grayscale, Normalize
    ↓
[48×48 Grayscale Image] → Ready for inference
```

#### 2. **Model Architecture**
```python
Input (1, 48, 48)
    ↓
Conv2d(1, 32, 3, padding=1)
ReLU → MaxPool2d(2)  → (32, 24, 24)
    ↓
Conv2d(32, 64, 3, padding=1)
ReLU → MaxPool2d(2)  → (64, 12, 12)
    ↓
Flatten → Linear(9216, 128)
    ↓
SNN Layer (Temporal processing, T=25)
    ↓
Output (7 classes)  → Softmax → Log probabilities
```

#### 3. **Inference Pipeline**
```python
Video Frame
    ↓
[Preprocess]
    ↓
[Model 1 Inference] → Prediction 1
[Model 2 Inference] → Prediction 2
[Model 3 Inference] → Prediction 3
    ↓
[Ensemble Voting] → Final Emotion
    ↓
[Risk Mapping] → Risk Level & Color
    ↓
[Visualization] → Display Output
```

---

## 5. Technical Stack

### Programming Language:
- **Python 3.8+**: Primary development language

### Deep Learning Frameworks:
- **PyTorch**: Neural network implementation
- **SNNTorch**: Spiking neural network operations
- **torchvision**: Pre-trained models and transformations

### Computer Vision:
- **OpenCV 4.x**: Face detection and video processing
- **NumPy**: Numerical computations
- **Pillow**: Image processing

### Data Science & Metrics:
- **scikit-learn**: Model evaluation metrics, confusion matrix
- **Pandas**: Data handling and analysis
- **Matplotlib & Seaborn**: Visualization

### Development Tools:
- **Git**: Version control
- **Pytest**: Testing framework
- **Jupyter Notebook**: Experimentation and documentation

### Hardware Acceleration:
- **CUDA 11.0+**: GPU support (NVIDIA)
- **cuDNN**: Deep learning acceleration

---

## 6. Data Flow & Processing

### Training Data Flow:
```
data/raw/train/
├── angry/ → Load images
├── disgusted/ → Load images
├── fearful/ → Load images
├── happy/ → Load images
├── neutral/ → Load images
├── sad/ → Load images
└── surprised/ → Load images
    ↓
[Data Augmentation]
    ↓
[Train/Val Split (80/20)]
    ↓
[Model Training]
    ↓
[Model 1, Model 2, Model 3] saved
```

### Testing Data Flow:
```
data/raw/test/
├── [Emotion folders]
    ↓
[Preprocessing]
    ↓
[Model Inference]
    ↓
[Metrics Computation]
    ↓
[Performance Report]
```

### Real-time Inference Flow:
```
Webcam Stream
    ↓
[Frame Capture]
    ↓
[Face Detection]
    ↓
[Preprocessing]
    ↓
[Ensemble Inference]
    ↓
[Risk Assessment]
    ↓
[Visualization & Display]
```

---

## 7. Model Specifications

### CNN Feature Extractor:
- **Input**: 48×48 grayscale image
- **Conv Layer 1**: 32 filters, 3×3 kernel, padding=1
- **MaxPool 1**: 2×2, stride 2
- **Conv Layer 2**: 64 filters, 3×3 kernel, padding=1
- **MaxPool 2**: 2×2, stride 2
- **Output Features**: 64×12×12 = 9,216 features

### SNN Temporal Layer:
- **Type**: Leaky Integrate-and-Fire neurons
- **Time steps**: T = 25
- **Input Size**: 9,216 (flattened CNN features)
- **Hidden Size**: 128
- **Output Size**: 7 (emotion classes)
- **Beta (decay)**: 0.9 (membrane potential decay)
- **Threshold**: 1.0

### Ensemble Configuration:
- **Number of Models**: 3 identical SNN-CNN hybrids
- **Aggregation**: Majority voting
- **Confidence Threshold**: > 0.6
- **Output**: Class with highest vote count

---

## 8. Performance Requirements & Benchmarks

### Target Metrics:
| Metric | Target | Current |
|--------|--------|---------|
| Overall Accuracy | >75% | TBD |
| Inference Speed | <100ms | TBD |
| FPS (GPU) | >15fps | TBD |
| Memory Usage | <2GB | TBD |
| Per-class F1-score | >0.70 | TBD |

### Hardware Requirements:
- **Minimum (CPU only)**:
  - 4GB RAM
  - Quad-core processor
  - 200MB disk space
  
- **Recommended (GPU)**:
  - 8GB RAM
  - NVIDIA GPU with CUDA support
  - 500MB disk space

---

## 9. Deployment & Integration

### Real-time Detection Deployment:
```python
# Single command deployment
python realtime_detection.py
```

### Model Serving Options:
1. **Direct PyTorch Loading**: Load .pth files directly
2. **ONNX Export**: Convert for broader compatibility
3. **API Deployment**: Wrap in FastAPI for web service
4. **Docker Containerization**: Package for cloud deployment

### Integration Points:
- Webcam or IP camera streams
- Desktop application
- Web service API
- Mobile application (via API)

---

## 10. Testing & Validation Strategy

### Unit Testing:
- Model forward pass validation
- Data preprocessing correctness
- Utility function testing

### Integration Testing:
- End-to-end pipeline validation
- Model ensemble voting accuracy
- Real-time processing latency

### Validation Testing:
- Test set evaluation
- Cross-validation procedures
- Confusion matrix analysis
- Per-class metrics (precision, recall, F1)

### Performance Testing:
- Inference speed benchmarking
- Memory usage profiling
- GPU utilization analysis
- Scalability testing

---

## 11. Security & Privacy Considerations

### Data Handling:
- Video streams processed in real-time (no storage)
- Training data stored securely
- No personal data retention

### Model Security:
- Model files protected
- Release only validated versions
- Regular security audits

---

## 12. Future Enhancements

### Phase 2 Features:
- Multi-face tracking and independent emotion detection
- Emotion intensity measurement
- Temporal emotion tracking (emotion state over time)
- Advanced CNN architectures (ResNet, VGG)
- Attention mechanisms for facial landmark focus
- Mobile app integration

### Phase 3 Features:
- Real-time emotion statistics dashboard
- Group emotion analysis
- Continuous learning capabilities
- Transfer learning for new domains

---

**Document Version**: 1.0  
**Last Updated**: April 10, 2026  
**Status**: Final  
**Approved By**: Project Team
