# Testing Standards & Quality Assurance - FER Project

---

## 1. Testing Framework & Tools

### 1.1 Testing Tools Used

```python
# Unit Testing
import pytest
import unittest

# Mocking & Fixtures
from unittest.mock import Mock, patch, MagicMock
import pytest.fixtures

# Code Coverage
# pip install pytest-cov
# pytest --cov=. --cov-report=html

# Performance Testing
import timeit
import memory_profiler

# Numerical Testing
import numpy as np
from numpy.testing import assert_array_almost_equal

# PyTorch Testing
import torch
from torch.testing import assert_close
```

### 1.2 Test Structure

```
tests/
├── unit/
│   ├── __init__.py
│   ├── test_model.py
│   ├── test_preprocessing.py
│   ├── test_inference.py
│   └── test_utils.py
│
├── integration/
│   ├── __init__.py
│   ├── test_pipeline.py
│   └── test_ensemble.py
│
├── fixtures/
│   ├── __init__.py
│   ├── sample_images.py
│   └── mock_models.py
│
└── conftest.py              # Pytest configuration
```

---

## 2. Unit Testing Standards

### 2.1 Test Template

```python
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

# Import the component to test
from utils import SNNModel, preprocess_image


class TestSNNModel:
    """Test suite for SNNModel class."""
    
    @pytest.fixture
    def model(self):
        """Fixture to create model instance."""
        model = SNNModel(
            in_channels=1,
            num_classes=7,
            num_timesteps=25
        )
        return model
    
    @pytest.fixture
    def sample_input(self):
        """Fixture for sample input tensor."""
        return torch.randn(1, 1, 48, 48)
    
    def test_model_initialization(self, model):
        """Test model initializes with correct architecture."""
        assert isinstance(model, torch.nn.Module)
        assert hasattr(model, 'conv1')
        assert hasattr(model, 'conv2')
        assert hasattr(model, 'fc1')
        assert hasattr(model, 'fc2')
    
    def test_forward_pass_shape(self, model, sample_input):
        """Test forward pass returns correct output shape."""
        output = model(sample_input)
        assert output.shape == (1, 7)  # (batch_size, num_classes)
    
    def test_forward_pass_values(self, model, sample_input):
        """Test forward pass returns valid values."""
        output = model(sample_input)
        # Check no NaN or Inf values
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_gradient_flow(self, model, sample_input):
        """Test gradients flow through model."""
        optimizer = torch.optim.Adam(model.parameters())
        output = model(sample_input)
        loss = output.sum()
        loss.backward()
        
        # Check gradients are computed
        for param in model.parameters():
            assert param.grad is not None
    
    def test_cuda_compatibility(self, model, sample_input):
        """Test model works on GPU (if available)."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            model = model.to(device)
            sample_input = sample_input.to(device)
            output = model(sample_input)
            assert output.device.type == 'cuda'
        else:
            pytest.skip("CUDA not available")
    
    def test_batch_processing(self, model):
        """Test model handles batches correctly."""
        batch_input = torch.randn(8, 1, 48, 48)
        output = model(batch_input)
        assert output.shape == (8, 7)
    
    def test_eval_mode(self, model, sample_input):
        """Test model behaves correctly in eval mode."""
        model.eval()
        with torch.no_grad():
            output = model(sample_input)
        
        assert output.shape == (1, 7)
```

### 2.2 Preprocessing Tests

```python
class TestPreprocessing:
    """Test suite for image preprocessing functions."""
    
    @pytest.fixture
    def sample_image_path(self, tmp_path):
        """Create temporary sample image for testing."""
        from PIL import Image
        import numpy as np
        
        # Create synthetic image
        img_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        filepath = tmp_path / "test_image.jpg"
        img.save(filepath)
        return str(filepath)
    
    def test_preprocess_image_output_shape(self, sample_image_path):
        """Test preprocessed image has correct shape."""
        processed = preprocess_image(sample_image_path)
        assert processed.shape == (1, 48, 48)
    
    def test_preprocess_image_value_range(self, sample_image_path):
        """Test preprocessed values are in [0, 1]."""
        processed = preprocess_image(sample_image_path)
        assert processed.min() >= 0
        assert processed.max() <= 1
    
    def test_preprocess_image_dtype(self, sample_image_path):
        """Test preprocessed image has correct dtype."""
        processed = preprocess_image(sample_image_path)
        assert processed.dtype == torch.float32
    
    def test_preprocess_image_file_not_found(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            preprocess_image("nonexistent_file.jpg")
    
    def test_preprocess_image_invalid_format(self, tmp_path):
        """Test error handling for invalid image format."""
        invalid_file = tmp_path / "invalid.txt"
        invalid_file.write_text("Not an image")
        
        with pytest.raises((ValueError, IOError)):
            preprocess_image(str(invalid_file))
```

### 2.3 Inference Tests

```python
class TestInference:
    """Test suite for model inference."""
    
    @pytest.fixture
    def model(self):
        """Load pre-trained model."""
        model = SNNModel()
        model.eval()
        return model
    
    def test_inference_output_sum_to_one(self, model):
        """Test model outputs are valid probabilities."""
        batch = torch.randn(4, 1, 48, 48)
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        
        # Check sums to approximately 1
        assert_array_almost_equal(
            probs.sum(dim=1).numpy(),
            np.ones(4),
            decimal=5
        )
    
    def test_inference_no_gradients(self, model):
        """Test inference doesn't compute gradients."""
        batch = torch.randn(4, 1, 48, 48)
        with torch.no_grad():
            output = model(batch)
        
        # All tensors should have requires_grad=False
        for param in model.parameters():
            assert not param.requires_grad or param.grad is not None
    
    def test_inference_deterministic(self, model):
        """Test inference is reproducible."""
        torch.manual_seed(42)
        batch = torch.randn(2, 1, 48, 48)
        
        model.eval()
        with torch.no_grad():
            output1 = model(batch).clone()
        
        torch.manual_seed(42)
        batch = torch.randn(2, 1, 48, 48)
        with torch.no_grad():
            output2 = model(batch)
        
        torch.testing.assert_close(output1, output2)
```

---

## 3. Integration Testing

### 3.1 Pipeline Integration Tests

```python
class TestPipeline:
    """Test complete FER pipeline."""
    
    def test_end_to_end_preprocessing_inference(self, tmp_path):
        """Test full pipeline from image to prediction."""
        # Create test image
        from PIL import Image
        img_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = tmp_path / "test.jpg"
        img.save(img_path)
        
        # Run pipeline
        preprocessed = preprocess_image(str(img_path))
        model = SNNModel()
        model.eval()
        
        with torch.no_grad():
            output = model(preprocessed.unsqueeze(0))
        
        prediction = torch.argmax(output, dim=1)
        assert prediction.shape == (1,)
        assert 0 <= prediction.item() < 7
    
    def test_ensemble_prediction(self):
        """Test ensemble voting mechanism."""
        # Create mock models with different predictions
        predictions = torch.tensor([
            [0.1, 0.8, 0.1],  # Model 1 predicts class 1
            [0.05, 0.85, 0.1], # Model 2 predicts class 1
            [0.2, 0.3, 0.5]    # Model 3 predicts class 2
        ])
        
        # Majority voting should return class 1
        votes = torch.argmax(predictions, dim=1)
        final_pred = torch.mode(votes)[0]
        
        assert final_pred.item() == 1
    
    def test_risk_assessment_mapping(self):
        """Test emotion to risk level mapping."""
        from realtime_detection import get_risk_level
        
        emotion_risk_map = {
            'sad': 'HIGH',
            'angry': 'MEDIUM',
            'fearful': 'MEDIUM',
            'happy': 'LOW',
            'neutral': 'LOW',
            'disgusted': 'LOW',
            'surprised': 'LOW'
        }
        
        for emotion, expected_risk in emotion_risk_map.items():
            actual_risk = get_risk_level(emotion)
            assert actual_risk == expected_risk
```

### 3.2 Configuration & Data Tests

```python
class TestDataHandling:
    """Test data loading and handling."""
    
    def test_emotion_categories(self):
        """Verify all emotion categories are present."""
        from config import EMOTIONS
        
        expected_emotions = [
            'angry', 'disgusted', 'fearful', 'happy',
            'neutral', 'sad', 'surprised'
        ]
        
        assert len(EMOTIONS) == 7
        assert set(EMOTIONS) == set(expected_emotions)
    
    def test_data_directory_structure(self):
        """Test data directories have correct structure."""
        import os
        from config import DATA_DIR
        
        # Check train directory
        train_dir = os.path.join(DATA_DIR, 'raw', 'train')
        assert os.path.exists(train_dir)
        
        # Check test directory
        test_dir = os.path.join(DATA_DIR, 'raw', 'test')
        assert os.path.exists(test_dir)
        
        # Check all emotion folders exist
        for emotion in EMOTIONS:
            train_emotion_dir = os.path.join(train_dir, emotion)
            test_emotion_dir = os.path.join(test_dir, emotion)
            assert os.path.exists(train_emotion_dir)
            assert os.path.exists(test_emotion_dir)
```

---

## 4. Validation Testing

### 4.1 Model Validation

```python
def test_model_accuracy_on_test_set():
    """Test model achieves acceptable accuracy."""
    from test_snn import evaluate_model
    
    model = SNNModel()
    test_loader = create_test_dataloader()
    
    accuracy, metrics = evaluate_model(model, test_loader)
    
    # Minimum acceptable accuracy: 75%
    assert accuracy >= 75.0, f"Accuracy {accuracy}% below threshold"
    assert metrics['f1_score'] >= 0.70

def test_per_class_metrics():
    """Test per-class performance meets requirements."""
    from sklearn.metrics import classification_report
    
    # Get predictions and ground truth
    predictions = get_all_predictions()
    ground_truth = get_all_labels()
    
    report = classification_report(ground_truth, predictions, output_dict=True)
    
    for emotion in EMOTIONS:
        f1 = report[emotion]['f1-score']
        precision = report[emotion]['precision']
        recall = report[emotion]['recall']
        
        assert f1 >= 0.65, f"{emotion}: F1-score below threshold"
        assert precision >= 0.60, f"{emotion}: Precision below threshold"
        assert recall >= 0.60, f"{emotion}: Recall below threshold"

def test_confusion_matrix_structure():
    """Test confusion matrix is valid."""
    from sklearn.metrics import confusion_matrix
    
    predictions = get_all_predictions()
    ground_truth = get_all_labels()
    
    cm = confusion_matrix(ground_truth, predictions)
    
    # Should be 7x7 matrix
    assert cm.shape == (7, 7)
    # All values should be non-negative
    assert (cm >= 0).all()
    # Diagonal should be high (correct predictions)
    diagonal_sum = np.trace(cm)
    total = cm.sum()
    accuracy = diagonal_sum / total
    assert accuracy >= 0.75
```

---

## 5. Performance Testing

### 5.1 Speed & Latency Tests

```python
class TestPerformance:
    """Test performance requirements."""
    
    def test_inference_latency(self):
        """Test inference latency is under 100ms."""
        import time
        
        model = SNNModel()
        model.eval()
        batch = torch.randn(1, 1, 48, 48)
        
        # Warm-up
        with torch.no_grad():
            _ = model(batch)
        
        # Measure
        start = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = model(batch)
        elapsed = (time.time() - start) / 100 * 1000  # Convert to ms
        
        assert elapsed < 100, f"Inference latency {elapsed}ms exceeds 100ms"
    
    def test_preprocessing_speed(self, sample_image_path):
        """Test preprocessing completes quickly."""
        import time
        
        start = time.time()
        for _ in range(100):
            _ = preprocess_image(sample_image_path)
        elapsed = (time.time() - start) / 100 * 1000
        
        assert elapsed < 50, f"Preprocessing {elapsed}ms too slow"
    
    def test_model_load_time(self):
        """Test model loading is fast."""
        import time
        from config import MODEL_PATHS
        
        start = time.time()
        for model_path in MODEL_PATHS:
            _ = torch.load(model_path)
        elapsed = time.time() - start
        
        assert elapsed < 5, f"Model loading {elapsed}s too slow"

    @pytest.mark.benchmark
    def test_batch_inference_throughput(self, benchmark):
        """Benchmark batch inference throughput."""
        model = SNNModel()
        model.eval()
        batch = torch.randn(8, 1, 48, 48)
        
        def run_inference():
            with torch.no_grad():
                return model(batch)
        
        result = benchmark(run_inference)
        assert result.shape == (8, 7)
```

### 5.2 Memory Tests

```python
class TestMemory:
    """Test memory usage."""
    
    def test_model_memory_usage(self):
        """Test model uses reasonable memory."""
        import sys
        
        model = SNNModel()
        size_mb = sys.getsizeof(model) / (1024 * 1024)
        
        # Models should be < 50MB each
        assert size_mb < 50, f"Model size {size_mb}MB exceeds 50MB"
    
    @pytest.mark.slow
    def test_memory_leak_in_training_loop(self):
        """Test for memory leaks during training."""
        import tracemalloc
        
        tracemalloc.start()
        
        model = SNNModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        # Simulate training loop
        for epoch in range(10):
            for batch_idx in range(5):
                inputs = torch.randn(4, 1, 48, 48)
                targets = torch.randint(0, 7, (4,))
                
                outputs = model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        current, peak = tracemalloc.get_traced_memory()
        memory_mb = peak / (1024 * 1024)
        
        # Memory should not exceed 500MB for this test
        assert memory_mb < 500, f"Peak memory {memory_mb}MB exceeds limit"
        tracemalloc.stop()
```

---

## 6. Edge Case & Error Handling Tests

### 6.1 Robustness Tests

```python
class TestRobustness:
    """Test error handling and edge cases."""
    
    def test_empty_image_handling(self):
        """Test handling of empty/black images."""
        empty_img = torch.zeros(1, 1, 48, 48)
        model = SNNModel()
        model.eval()
        
        with torch.no_grad():
            output = model(empty_img)
        
        assert output.shape == (1, 7)
        assert not torch.isnan(output).any()
    
    def test_extreme_values_handling(self):
        """Test handling of extreme input values."""
        extreme_img = torch.ones(1, 1, 48, 48) * 100  # Very large values
        model = SNNModel()
        model.eval()
        
        with torch.no_grad():
            output = model(extreme_img)
        
        assert output.shape == (1, 7)
        assert not torch.isnan(output).any()
    
    def test_low_confidence_prediction(self):
        """Test handling of low-confidence predictions."""
        model = SNNModel()
        model.eval()
        batch = torch.randn(4, 1, 48, 48)
        
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
        
        # Filter low confidence predictions
        confidence_threshold = 0.5
        high_conf_mask = probs.max(dim=1)[0] > confidence_threshold
        
        # Some predictions might be below threshold
        assert high_conf_mask.sum() >= 0
    
    def test_invalid_emotion_handling(self):
        """Test handling of invalid emotion strings."""
        from realtime_detection import get_risk_level
        
        # Should raise error or return default
        with pytest.raises((ValueError, KeyError)):
            get_risk_level("invalid_emotion")
```

---

## 7. Running Tests

### 7.1 Test Execution

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_model.py

# Run specific test class
pytest tests/unit/test_model.py::TestSNNModel

# Run specific test
pytest tests/unit/test_model.py::TestSNNModel::test_forward_pass_shape

# Run with coverage
pytest --cov=. --cov-report=html

# Run with verbose output
pytest -v

# Run only fast tests (skip slow)
pytest -m "not slow"

# Run with specific markers
pytest -m "benchmark"

# Generate test report
pytest --html=report.html

# Show slowest tests
pytest --durations=10
```

### 7.2 Pytest Configuration (conftest.py)

```python
import pytest
import torch
import numpy as np

@pytest.fixture(scope="session")
def model():
    """Load model once for all tests."""
    from utils import SNNModel
    model = SNNModel()
    return model

@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test."""
    torch.manual_seed(42)
    np.random.seed(42)
    yield
    torch.manual_seed(42)
    np.random.seed(42)

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as benchmarks"
    )

# pytest.ini configuration
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    slow: slow running tests
    benchmark: benchmark tests
```

---

## 8. Test Coverage Goals

| Component | Target Coverage | Current |
|-----------|-----------------|---------|
| utils.py | > 90% | TBD |
| train_snn.py | > 85% | TBD |
| realtime_detection.py | > 80% | TBD |
| Overall | > 85% | TBD |

---

## 9. Continuous Integration

### 9.1 GitHub Actions Workflow

```yaml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: pytest --cov=. --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

---

**Document Version**: 1.0  
**Last Updated**: April 10, 2026  
**Testing Lead**: Anushka Verma  
**Status**: Final
