# Coding Standards & Guidelines - FER Project

---

## 1. Python Coding Conventions

### 1.1 Code Style

**Standard**: PEP 8 - Enhanced version

#### Naming Conventions:

```python
# Classes: PascalCase
class SNNModel:
    pass

class EnsemblePredictor:
    pass

# Functions/Methods: snake_case
def preprocess_image(image_path):
    pass

def detect_faces(frame):
    pass

# Constants: UPPER_SNAKE_CASE
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
NUM_TIMESTEPS = 25
MODEL_PATH = './models/'

# Private/Internal: _leading_underscore
def _compute_loss(predictions, targets):
    pass

class _InternalHelper:
    pass

# Protected/Subclass: __double_underscore (name mangling)
self.__private_var = 42
```

#### Indentation & Spacing:

```python
# ✅ CORRECT: 4 spaces, no tabs
def train_model(data, epochs=100):
    for epoch in range(epochs):
        for batch in data:
            loss = compute_loss(batch)
            loss.backward()

# ❌ WRONG: Mixed tabs and spaces
def train_model(data, epochs=100):
	for epoch in range(epochs):
        for batch in data:
            loss = compute_loss(batch)
```

#### Line Length:

```python
# ✅ CORRECT: Max 88 characters (Black formatter standard)
prediction = ensemble_model.predict(face_image)
confidence = get_confidence_score(prediction)

# ❌ WRONG: Exceeds 88 characters
very_long_prediction_variable = ensemble_model.predict_with_full_configuration(face_image, all_parameters=True)

# SOLUTION: Break into multiple lines
very_long_prediction_variable = ensemble_model.predict_with_full_configuration(
    face_image,
    all_parameters=True
)
```

### 1.2 Documentation Standards

#### Module-Level Docstrings:

```python
"""
Facial Expression Recognition - Real-time Detection Module

This module handles real-time emotion detection from webcam streams.
It integrates face detection, preprocessing, and ensemble prediction.

Functions:
    main: Entry point for real-time detection
    load_models: Load trained SNN models
    detect_emotions: Detect emotions in webcam stream

Dependencies:
    - torch, torchvision
    - cv2 (OpenCV)
    - numpy

Author: Bhavya Singh
Last Updated: 2026-04-10
"""
```

#### Function/Method Docstrings:

```python
def preprocess_image(image_path, target_size=(48, 48)):
    """
    Preprocess image for model inference.
    
    Converts image to grayscale, resizes to target dimensions,
    and normalizes pixel values to [0, 1] range.
    
    Args:
        image_path (str): Path to input image file
        target_size (tuple): Target image dimensions (height, width)
        
    Returns:
        torch.Tensor: Normalized image tensor of shape (1, 48, 48)
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded or processed
        
    Example:
        >>> img_tensor = preprocess_image('face.jpg')
        >>> print(img_tensor.shape)
        torch.Size([1, 48, 48])
    """
    pass
```

#### Class Docstrings:

```python
class SNNModel(torch.nn.Module):
    """
    Spiking Neural Network for Facial Expression Recognition.
    
    Combines CNN feature extraction with temporal SNN processing
    for robust emotion classification across 7 emotion categories.
    
    Attributes:
        num_classes (int): Number of emotion categories (default: 7)
        num_timesteps (int): Number of SNN time steps (default: 25)
        hidden_size (int): Size of hidden layer (default: 128)
        
    Example:
        >>> model = SNNModel(num_timesteps=25)
        >>> image = torch.randn(1, 1, 48, 48)
        >>> output = model(image)
        >>> print(output.shape)
        torch.Size([1, 7])
    """
    
    def __init__(self, num_classes=7, num_timesteps=25):
        """Initialize SNNModel."""
        pass
```

#### Inline Comments:

```python
# ✅ GOOD: Explain WHY, not WHAT
# Use ensemble voting for robustness against individual model errors
predictions = torch.stack([m(x) for m in models])
final_prediction = torch.mode(predictions, dim=0)[0]

# ❌ BAD: Obvious comments that clutter code
# Loop through models
for model in models:
    # Get prediction
    pred = model(x)
```

### 1.3 Import Organization

```python
# Standard library imports first
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Third-party imports (numpy, torch, etc.)
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import cv2
import snntorch as snn

# Local imports
from utils import SNNModel, preprocess_image
from config import EMOTIONS, MODEL_PATHS

# Organize imports within each group alphabetically
```

### 1.4 Type Hints

```python
# ✅ USE type hints for clarity
from typing import List, Tuple, Optional, Dict

def predict_emotion(image: np.ndarray) -> Tuple[str, float]:
    """
    Predict emotion from image.
    
    Args:
        image: Facial image as numpy array
        
    Returns:
        Tuple of (emotion_label, confidence_score)
    """
    pass

def load_models(model_paths: List[str]) -> List[nn.Module]:
    """Load multiple trained models."""
    pass

def assess_risk_level(emotion: str) -> Optional[str]:
    """Return risk level for given emotion (or None)."""
    pass

# For complex types
def process_batch(
    batch_data: Dict[str, torch.Tensor],
    model: nn.Module,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """Process batch with model on specified device."""
    pass
```

---

## 2. PyTorch & Deep Learning Standards

### 2.1 Model Definition

```python
class SNNModel(nn.Module):
    """Hybrid CNN-SNN model for emotion recognition."""
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 7,
        num_timesteps: int = 25,
        hidden_size: int = 128,
        threshold: float = 1.0,
        beta: float = 0.9
    ):
        super().__init__()
        
        # CNN Feature Extractor
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten_size = 64 * 12 * 12  # After pooling
        
        # SNN Temporal Layer
        self.fc1 = nn.Linear(self.flatten_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        # SNN Parameters
        self.num_timesteps = num_timesteps
        self.threshold = threshold
        self.beta = beta
        self.hidden_size = hidden_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through model.
        
        Args:
            x: Input tensor (batch_size, 1, 48, 48)
            
        Returns:
            Output logits (batch_size, 7)
        """
        # CNN Feature Extraction
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, self.flatten_size)
        
        # SNN Temporal Processing
        output = self._snn_forward(x)
        
        return output
    
    def _snn_forward(self, x: torch.Tensor) -> torch.Tensor:
        """SNN temporal processing with LIF neurons."""
        batch_size = x.shape[0]
        outputs = []
        
        # Initialize membrane potential
        mem = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        for t in range(self.num_timesteps):
            # Leaky integrate-and-fire dynamics
            cur = self.fc1(x)
            mem = self.beta * mem + cur
            spike = (mem > self.threshold).float()
            mem = mem * (1 - spike)
            outputs.append(spike)
        
        # Average spikes over time
        output = torch.mean(torch.stack(outputs), dim=0)
        logits = self.fc2(output)
        
        return logits
```

### 2.2 Training Loop

```python
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: Neural network model
        dataloader: Training data loader
        optimizer: Optimization algorithm
        criterion: Loss function
        device: Computation device (cuda/cpu)
        
    Returns:
        Average loss for epoch
    """
    model.train()  # Set to training mode
    total_loss = 0.0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        # Move data to device
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Progress logging
        if (batch_idx + 1) % 10 == 0:
            print(f'Batch {batch_idx + 1}: Loss = {loss.item():.4f}')
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[float, Dict]:
    """
    Evaluate model on dataset.
    
    Args:
        model: Neural network model
        dataloader: Evaluation data loader
        device: Computation device
        
    Returns:
        Tuple of (accuracy, metrics_dict)
    """
    model.eval()  # Set to evaluation mode
    correct = 0
    total = 0
    
    predictions_list = []
    labels_list = []
    
    with torch.no_grad():  # No gradient computation
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            predictions_list.extend(predicted.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    
    # Compute additional metrics
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_list, predictions_list, average='weighted'
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return accuracy, metrics
```

### 2.3 Model Checkpointing

```python
def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str
) -> None:
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.datetime.now().isoformat()
    }
    torch.save(checkpoint, filepath)
    print(f'Checkpoint saved: {filepath}')

def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    filepath: str
) -> Tuple[int, float]:
    """Load model checkpoint."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f'Checkpoint loaded from {filepath}')
    return epoch, loss
```

---

## 3. OpenCV & Computer Vision Standards

### 3.1 Face Detection

```python
def detect_faces(
    frame: np.ndarray,
    cascade_classifier: cv2.CascadeClassifier,
    scale_factor: float = 1.3,
    min_neighbors: int = 5
) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces in frame using Haar Cascade.
    
    Args:
        frame: Input image frame
        cascade_classifier: Haar Cascade classifier
        scale_factor: Image pyramid scale factor
        min_neighbors: Minimum detections for face
        
    Returns:
        List of face rectangles (x, y, w, h)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade_classifier.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(48, 48)
    )
    return faces

def extract_and_preprocess_face(
    frame: np.ndarray,
    face_rect: Tuple[int, int, int, int],
    target_size: Tuple[int, int] = (48, 48)
) -> np.ndarray:
    """
    Extract face region and preprocess.
    
    Args:
        frame: Input image frame
        face_rect: Face rectangle (x, y, w, h)
        target_size: Target output size
        
    Returns:
        Preprocessed face image
    """
    x, y, w, h = face_rect
    face = frame[y:y+h, x:x+w]
    
    # Convert to grayscale
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    
    # Resize
    face_resized = cv2.resize(face_gray, target_size)
    
    # Normalize to [0, 1]
    face_normalized = face_resized / 255.0
    
    return face_normalized
```

### 3.2 Visualization

```python
def draw_prediction(
    frame: np.ndarray,
    face_rect: Tuple[int, int, int, int],
    emotion: str,
    confidence: float,
    risk_level: str,
    risk_color: Tuple[int, int, int]
) -> np.ndarray:
    """
    Draw prediction on frame.
    
    Args:
        frame: Input frame to draw on
        face_rect: Face rectangle (x, y, w, h)
        emotion: Predicted emotion label
        confidence: Confidence score
        risk_level: Risk level text
        risk_color: RGB color for bounding box
        
    Returns:
        Annotated frame
    """
    x, y, w, h = face_rect
    
    # Draw bounding box
    cv2.rectangle(frame, (x, y), (x+w, y+h), risk_color, 2)
    
    # Draw emotion label
    label = f'{emotion.capitalize()} ({confidence:.2f})'
    cv2.putText(
        frame,
        label,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        risk_color,
        2
    )
    
    # Draw risk level
    cv2.putText(
        frame,
        risk_level,
        (x, y + h + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        risk_color,
        2
    )
    
    return frame
```

---

## 4. Project Structure Standards

```
FER/
├── models/                    # Trained model files
│   ├── snn_model_1.pth
│   ├── snn_model_2.pth
│   └── snn_model_3.pth
│
├── data/
│   └── raw/
│       ├── train/           # Training data
│       │   └── [7 emotion folders]
│       └── test/            # Test data
│           └── [7 emotion folders]
│
├── report/                   # Project documentation
│   ├── req1.md
│   ├── SYSTEM_DESIGN.md
│   ├── CODING_STANDARDS.md
│   ├── TESTING_STANDARDS.md
│   └── ROLES_AND_RESPONSIBILITIES.md
│
├── utils.py                 # Utilities and model definition
├── train_snn.py            # Training script
├── test_snn.py             # Testing script
├── realtime_detection.py   # Real-time detection
├── requirements.txt        # Dependencies
├── README.md              # Project overview
└── individual.md          # Team contributions
```

---

## 5. Code Review Checklist

- [ ] Code follows PEP 8 standards
- [ ] Functions have docstrings with Args/Returns
- [ ] Type hints used for function signatures
- [ ] No hardcoded values (use constants)
- [ ] Error handling present
- [ ] Logging for debugging
- [ ] No unused imports
- [ ] No print statements (use logging instead)
- [ ] Tests written and passing
- [ ] Performance benchmarked

---

## 6. Common Anti-patterns to Avoid

```python
# ❌ ANTI-PATTERN 1: Using print for debugging
print("Model loaded")
print("Epoch:", epoch, "Loss:", loss)

# ✅ CORRECT: Use logging
import logging
logger = logging.getLogger(__name__)
logger.info("Model loaded")
logger.info(f"Epoch: {epoch}, Loss: {loss}")

---

# ❌ ANTI-PATTERN 2: Bare except clause
try:
    model = torch.load("model.pth")
except:
    pass

# ✅ CORRECT: Specific exception handling
try:
    model = torch.load("model.pth")
except FileNotFoundError:
    logger.error("Model file not found")
    raise
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

---

# ❌ ANTI-PATTERN 3: Mutable default arguments
def load_models(paths=[], device=None):
    pass

# ✅ CORRECT: Use None as default
def load_models(paths: Optional[List[str]] = None, device: Optional[torch.device] = None):
    if paths is None:
        paths = []
    if device is None:
        device = torch.device('cpu')
    pass

---

# ❌ ANTI-PATTERN 4: CPU/GPU code not device-agnostic
model = model.cuda()  # Fails if no GPU
data = data.cuda()

# ✅ CORRECT: Device-agnostic code
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)
```

---

## 7. Performance Best Practices

```python
# ✅ Profile code before optimizing
import cProfile
profiler = cProfile.Profile()
profiler.enable()

# ... code to profile ...

profiler.disable()
profiler.print_stats()

---

# ✅ Use context managers for resource management
with torch.no_grad():
    predictions = model(data)

---

# ✅ Batch operations instead of loops
# ❌ SLOW:
for image in images:
    output = model(image.unsqueeze(0))

# ✅ FAST:
outputs = model(images)

---

# ✅ Move data to GPU before model
data = data.to(device)
model = model.to(device)
```

---

**Document Version**: 1.0  
**Last Updated**: April 10, 2026  
**Status**: Final  
**Compliance Standard**: PEP 8 Enhanced
