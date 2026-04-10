# System Design - Facial Expression Recognition (FER)

---

## 1. High-Level System Overview

### System Context Diagram:
```
┌─────────────────┐
│   User/Client   │
└────────┬────────┘
         │
    ┌────▼────────────────────────┐
    │ FER System Interface         │
    │  • Real-time Detection       │
    │  • Visualization Display     │
    │  • Control Commands          │
    └────┬───────────────────────┘
         │
    ┌────▼───────────────────────────────────┐
    │  Core Processing Engine               │
    │  ┌──────────────────────────────────┐  │
    │  │  • Face Detection & Tracking     │  │
    │  │  • Image Preprocessing           │  │
    │  │  • Ensemble Model Inference      │  │
    │  │  • Risk Assessment & Mapping     │  │
    │  └──────────────────────────────────┘  │
    └────┬───────────────────────────────────┘
         │
    ┌────▼──────────────────┐
    │  Hardware Resources   │
    │  • GPU (CUDA)         │
    │  • CPU                │
    │  • Webcam             │
    │  • RAM/Storage        │
    └───────────────────────┘
```

---

## 2. Layered Architecture

### Layer 1: Presentation Layer
**Purpose**: User interface and visualization

**Components**:
- Real-time video display
- Bounding box visualization
- Emotion labels and confidence scores
- Risk level indicators (color-coded)
- FPS and latency metrics
- Control interface (keyboard shortcuts)

**Technologies**:
- OpenCV (cv2.imshow)
- Matplotlib
- NumPy arrays for pixel manipulation

### Layer 2: Application Logic Layer
**Purpose**: Orchestrate system workflow

**Components**:
```python
main_pipeline():
    ├── initialize_camera()
    ├── load_models()
    ├── while(camera.isOpened()):
    │   ├── capture_frame()
    │   ├── detect_faces()
    │   ├── preprocess_faces()
    │   ├── run_inference()
    │   ├── aggregate_predictions()
    │   ├── assess_risk()
    │   └── visualize_results()
    └── cleanup_resources()
```

**Key Files**:
- `realtime_detection.py` - Main application controller
- `train_snn.py` - Training controller

### Layer 3: Model/Business Logic Layer
**Purpose**: Core ML model operations

**Components**:
- **Preprocessing Module**: Image normalization, augmentation
- **Inference Module**: Model forward pass, batch processing
- **Ensemble Module**: Voting mechanism, confidence aggregation
- **Risk Assessment Module**: Emotion to risk mapping

**Key Files**:
- `utils.py` - Model definitions and utilities
- Trained models: `snn_model_[1-3].pth`

### Layer 4: Data Access Layer
**Purpose**: Data handling and I/O

**Components**:
- **Image Loading**: From disk or camera stream
- **Dataset Management**: Organization of train/test data
- **Model Persistence**: Loading/saving .pth files
- **Configuration Management**: Hyperparameters, paths

**Key Files**:
- `requirements.txt` - Dependencies
- `data/` - Dataset organization

---

## 3. Component Architecture

### Component Diagram:

```
┌──────────────────────────────────────────────────────────────┐
│                    FER System Components                     │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐         ┌─────────────────┐             │
│  │ Video Input  │────────▶│ Face Detector   │             │
│  │ (Webcam)     │         │ (Haar Cascade)  │             │
│  └──────────────┘         └────────┬────────┘             │
│                                    │                      │
│                            ┌───────▼────────┐             │
│                            │ Preprocessor   │             │
│                            │ • Resize       │             │
│                            │ • Normalize    │             │
│                            │ • Grayscale    │             │
│                            └────────┬───────┘             │
│                                     │                     │
│              ┌──────────────────────┼──────────────────┐  │
│              │                      │                  │  │
│       ┌──────▼────┐         ┌──────▼────┐      ┌──────▼────┐
│       │  Model 1  │         │  Model 2  │      │  Model 3  │
│       │ (SNN-CNN) │         │ (SNN-CNN) │      │ (SNN-CNN) │
│       └──────┬────┘         └──────┬────┘      └──────┬────┘
│              │                     │                 │
│              └─────────┬───────────┴─────────────────┘
│                        │
│              ┌─────────▼──────────┐
│              │ Ensemble Voter     │
│              │ • Voting Logic     │
│              │ • Confidence Calc  │
│              └─────────┬──────────┘
│                        │
│              ┌─────────▼──────────────┐
│              │ Risk Assessor          │
│              │ Emotion→Risk Mapping   │
│              └─────────┬──────────────┘
│                        │
│              ┌─────────▼──────────────┐
│              │ Visualizer             │
│              │ • Drawing              │
│              │ • Display              │
│              └───────────────────────┘
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. Data Flow Architecture

### Training Pipeline Data Flow:

```
Data Preparation Phase:
  data/raw/train/
  ├── angry/     → File list
  ├── disgusted/ → File list
  └── ... (7 emotions)
         │
         ▼
  ┌─────────────────────┐
  │ ImageFolder Dataset │
  │ Load images &       │
  │ Apply transforms    │
  └─────────────────────┘
         │
         ├─────────────────────┐
         │                     │
         ▼                     ▼
  ┌──────────────┐     ┌──────────────┐
  │ Training Set │     │ Validation   │
  │ (80%)        │     │ Set (20%)    │
  └──────────────┘     └──────────────┘
         │                     │
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │ DataLoader          │
         │ (Batches, Shuffle)  │
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────────┐
         │ Training Loop           │
         │ • Forward Pass          │
         │ • Loss Computation      │
         │ • Backpropagation       │
         │ • Weight Updates        │
         └──────────┬──────────────┘
                    │
         ┌──────────▼──────────────┐
         │ Model Checkpoints       │
         │ • snn_model_1.pth       │
         │ • snn_model_2.pth       │
         │ • snn_model_3.pth       │
         └────────────────────────┘
```

### Inference Pipeline Data Flow:

```
Real-time Detection:
  
  Webcam Stream
         │
         ▼
  ┌──────────────────┐
  │ Frame Capture    │
  │ (30 FPS default) │
  └──────────────────┘
         │
         ▼
  ┌──────────────────────┐
  │ Face Detection       │
  │ (Haar Cascade)       │
  │ → Face rectangles    │
  └──────────────────────┘
         │
         ▼
  ┌─────────────────────────┐
  │ Extract Face Regions    │
  │ & Preprocess            │
  └─────────────────────────┘
         │
    ┌────┴────┬────┐
    │          │    │
    ▼          ▼    ▼
  Model1    Model2  Model3
    │          │    │
    └────┬─────┴────┘
         │
         ▼
  ┌──────────────────┐
  │ Ensemble Voting  │
  │ Max Vote         │
  └──────────────────┘
         │
         ▼
  ┌──────────────────┐
  │ Risk Mapping     │
  │ Emotion→Risk     │
  └──────────────────┘
         │
         ▼
  ┌──────────────────┐
  │ Visualization    │
  │ Display Output   │
  └──────────────────┘
```

---

## 5. Module Design

### Module 1: Data Preprocessing (`utils.py`)

**Responsibilities**:
- Image loading and conversion
- Normalization and standardization
- Data augmentation
- Batch processing

**Key Functions**:
```python
preprocess_image(image_path):
    - Load image from disk
    - Convert to grayscale
    - Resize to 48×48
    - Normalize pixel values [0, 1]
    - Return tensor

augment_image(image):
    - Apply random rotations
    - Apply horizontal flips
    - Adjust brightness/contrast
    - Return augmented image

create_dataloader(dataset_path):
    - Create ImageFolder dataset
    - Apply transforms
    - Create DataLoader with batching
    - Return DataLoader
```

### Module 2: Model Architecture (`utils.py`)

**Responsibilities**:
- Define CNN feature extractor
- Define SNN temporal neuron
- Compose hybrid architecture

**Architecture Pseudo-code**:
```python
class SNNModel(nn.Module):
    def __init__(self):
        CNN Feature Extractor:
            - Conv(1, 32) → ReLU → MaxPool
            - Conv(32, 64) → ReLU → MaxPool
            - Flatten → 9216 features
        
        SNN Layer:
            - LIF Neuron (25 timesteps)
            - Input: 9216
            - Hidden: 128
            - Output: 7
    
    def forward(self, x):
        - CNN features = conv_layers(x)
        - For each timestep (T=25):
            - SNN output = lif_neuron(features)
        - Return final classification
```

### Module 3: Training (`train_snn.py`)

**Responsibilities**:
- Load and prepare data
- Initialize models
- Training loop with loss computation
- Model checkpointing
- Ensemble creation

**Training Algorithm**:
```
Initialize 3 models
For each epoch:
    For each batch in training data:
        - Forward pass through model
        - Compute cross-entropy loss
        - Backward pass (backpropagation)
        - Update weights with optimizer
    
    For each batch in validation data:
        - Forward pass
        - Compute validation accuracy
    
    If validation accuracy improves:
        - Save model checkpoint
        
After all epochs:
    - Save final three models
    - Create ensemble configuration
```

### Module 4: Real-time Detection (`realtime_detection.py`)

**Responsibilities**:
- Webcam stream handling
- Real-time inference
- Vision output rendering
- Performance monitoring

**Real-time Loop**:
```python
while camera.isOpened():
    1. Capture frame from camera
    2. Detect faces using Haar Cascade
    3. For each detected face:
        a. Crop face region
        b. Preprocess (resize, normalize)
        c. Run ensemble inference
        d. Get emotion prediction
        e. Map to risk level
    4. Draw on frame:
        a. Bounding boxes
        b. Emotion labels
        c. Confidence scores
        d. Risk indicators
    5. Display frame with OpenCV
    6. Check for exit command (Q key)
    7. Update FPS counter
```

### Module 5: Testing & Validation (`test_snn.py`)

**Responsibilities**:
- Model evaluation on test set
- Metrics computation
- Confusion matrix generation
- Performance reporting

**Testing Procedure**:
```python
For each model:
    For each test image:
        - Preprocess image
        - Run inference
        - Record prediction
    
    Compute metrics:
        - Overall accuracy
        - Per-class precision
        - Per-class recall
        - Per-class F1-score
        - Confusion matrix
    
    For ensemble:
        - Test voting mechanism
        - Compute ensemble accuracy
        - Compare with individual models
    
    Generate report:
        - Print metrics
        - Save confusion matrix visualization
        - Save performance summary
```

---

## 6. Class Diagram

```
┌──────────────────────────────────────┐
│          SNNModel                    │
├──────────────────────────────────────┤
│ Attributes:                          │
│ - conv_layers: Sequential            │
│ - lif_neuron: LIF                   │
│ - fc_layer: Linear                   │
├──────────────────────────────────────┤
│ Methods:                             │
│ + forward(x): Tensor                 │
│ + initialize_weights(): void         │
│ + reset_membrane(): void             │
└──────────────────────────────────────┘

┌──────────────────────────────────────┐
│     EnsemblePredictor                │
├──────────────────────────────────────┤
│ Attributes:                          │
│ - models: List[SNNModel]             │
│ - voting_method: str                 │
├──────────────────────────────────────┤
│ Methods:                             │
│ + predict(x): Tensor                 │
│ + majority_voting(predictions): int  │
│ + compute_confidence(votes): float   │
│ + load_models(paths): void           │
└──────────────────────────────────────┘

┌──────────────────────────────────────┐
│      FaceDetector                    │
├──────────────────────────────────────┤
│ Attributes:                          │
│ - cascade_classifier: CascadeClassif │
├──────────────────────────────────────┤
│ Methods:                             │
│ + detect_faces(frame): List[Rect]    │
│ + extract_face(frame, rect): Image   │
│ + preprocess_face(image): Tensor     │
└──────────────────────────────────────┘
```

---

## 7. Sequence Diagrams

### Real-time Detection Sequence:

```
User         System      Camera    Detector    Model    Visualizer
 │              │            │         │         │          │
 │ Start         │            │         │         │          │
 ├─────────────▶ │            │         │         │          │
 │          Initialize        │         │         │          │
 │              │  Open ────▶ │         │         │          │
 │              │ Loop        │         │         │          │
 │              │    Capture  │         │         │          │
 │              │  ◀─────────┤         │         │          │
 │              │  Frame      │         │         │          │
 │              │    Detect Faces
 │              │  ──────────▶│         │         │          │
 │              │  Rectangles │         │         │          │
 │              │  ◀──────────┤         │         │          │
 │              │  Process Faces
 │              │    Preprocess
 │              │  Inference  │         │         │          │
 │              │  ──────────────────▶│         │          │
 │              │  Prediction │         │         │          │
 │              │  ◀──────────────────┤         │          │
 │              │  Risk Map & Visualize
 │              │  ───────────────────────────▶│
 │              │  Display    │         │         │          │
 │              │  ◀─────────────────────────────┤
 │ Display      │            │         │         │          │
 │◀─────────────┤            │         │         │          │
 │              │            │         │         │          │
 │ Press Q      │            │         │         │          │
 ├─────────────▶│            │         │         │          │
 │              │ Cleanup    │         │         │          │
 │              │ ───────────▶Close   │         │          │
 │              │            │         │         │          │
```

---

## 8. State Machine

### System States:

```
┌─────────────────────────────────────┐
│       System State Machine          │
├─────────────────────────────────────┤
│                                     │
│  ┌───────────┐                     │
│  │   INIT    │◀──────────┐         │
│  └─────┬─────┘           │         │
│        │                 │         │
│    Initialize            │         │
│    Resources             │         │
│        │                 │         │
│        ▼                 │         │
│  ┌─────────────┐         │         │
│  │   READY     │         │         │
│  └─────┬───────┘         │         │
│        │                 │         │
│    Load Models            │         │
│    & Camera               │         │
│        │                 │         │
│        ▼                 │         │
│  ┌──────────────┐        │         │
│  │  RUNNING     │        │         │
│  └─────┬────────┘        │         │
│        │                 │         │
│   Process Frames         │         │
│   Detect Emotions        │         │
│   Update Display         │         │
│        │                 │         │
│    Q Key Pressed         │         │
│        │                 │         │
│        ▼                 │         │
│  ┌────────────────┐      │         │
│  │  SHUTTING_DOWN │      │         │
│  └────────┬───────┘      │         │
│           │              │         │
│   Release Resources      │         │
│           │              │         │
│           └──────────────┘         │
│                                     │
└─────────────────────────────────────┘
```

---

## 9. Error Handling & Recovery

### Error Scenarios:

| Error Type | Scenario | Handling Strategy |
|-----------|----------|-------------------|
| **No Camera** | Camera not found | Display error, exit gracefully |
| **Model Not Found** | .pth file missing | Check path, provide error message |
| **Inference Failure** | Model forward pass error | Log error, use fallback model |
| **Memory Error** | Out of memory | Reduce batch size, clear cache |
| **Face Not Detected** | No face in frame | Display "No face detected" |
| **Invalid Prediction** | Confidence too low | Show "Uncertain" instead |
| **GPU Out of Memory** | CUDA memory exceeded | Fall back to CPU |

---

## 10. Performance Considerations

### Optimization Strategies:

1. **Model Optimization**:
   - Quantization: Convert FP32 to FP16
   - Pruning: Remove insignificant weights
   - Knowledge distillation: Smaller student model

2. **Inference Optimization**:
   - Batch processing multiple faces
   - GPU caching for repeated computation
   - Result caching for identical frames

3. **I/O Optimization**:
   - Asynchronous model loading
   - Frame buffer for pipeline parallelism
   - Non-blocking visualization

---

## 11. Scalability Design

### Horizontal Scaling:
- Multiple inference servers
- Load balancing across GPUs
- Distributed model inference

### Vertical Scaling:
- Larger GPU memory
- Multi-GPU processing
- TPU support (future)

---

## 12. Deployment Architecture

### Deployment Options:

```
Option 1: Standalone Desktop
  Python → FER System → Webcam Output

Option 2: Web Service
  Client ──▶ FastAPI Server ──▶ FER System
                │
                └──▶ GPU Instance

Option 3: Docker Container
  Docker Image containing:
    - Python Runtime
    - PyTorch
    - Trained Models
    - Application Code

Option 4: Cloud Deployment
  Client ──▶ Cloud API Gateway ──▶ Cloud GPU Instance
                                    └──▶ FER System
```

---

**Document Version**: 1.0  
**Last Updated**: April 10, 2026  
**Architecture Lead**: Bhavya Singh  
**Status**: Final
