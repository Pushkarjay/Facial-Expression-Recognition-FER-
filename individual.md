# Individual Contributions - Facial Expression Recognition (FER) Project

---

## Team Information

| Name | Roll No. | Contact |
|------|----------|---------|
| Pushkarjay Ajay | 22052328 | 8210164935 |
| Bhavya Singh | 2205120 | 7464062560 |
| Anushka Verma | 2205712 | 72588961346 |
| Kavya Dixit | 2205132 | +91 82877 40746 |

---

## Individual Contributions & Responsibilities

### 1. Pushkarjay Ajay (Roll No: 22052328)
**Role:** ML Architecture & Model Development Lead

#### Primary Contributions:
- **SNN Architecture Design**: Designed and implemented the hybrid CNN-SNN architecture combining feature extraction with temporal processing
- **Model Training Pipeline**: Developed the training script (`train_snn.py`) with:
  - Data loading and preprocessing
  - Model training with ensemble approach
  - Loss functions and optimization strategies
  - Hyperparameter tuning for 3 ensemble models
- **Ensemble Learning Implementation**: Created voting-based ensemble prediction system with 3 models for improved robustness
- **Performance Optimization**: Implemented CUDA support and GPU acceleration for faster training and inference
- **Model Serialization**: Implemented PyTorch model saving and loading mechanisms

#### Key Deliverables:
- ✅ `train_snn.py` - Complete training pipeline
- ✅ `utils.py` - SNN and CNN model architecture definitions
- ✅ 3 trained ensemble models (snn_model_1.pth, snn_model_2.pth, snn_model_3.pth)
- ✅ Model documentation and architecture diagrams
- ✅ Training logs and performance metrics

#### Technical Skills Demonstrated:
- PyTorch Framework
- Spiking Neural Networks (SNNTorch)
- CNN Architecture Design
- Ensemble Learning Methods
- GPU Computing (CUDA)
- Model Optimization Techniques

---

### 2. Bhavya Singh (Roll No: 2205120)
**Role:** System Design & Real-time Processing Lead

#### Primary Contributions:
- **System Architecture Design**: Designed the complete system architecture for real-time facial expression recognition
- **Real-time Detection Pipeline**: Developed `realtime_detection.py` with:
  - Webcam integration using OpenCV
  - Real-time face detection and preprocessing
  - Model inference on video streams
  - Risk assessment classification system
  - Visual feedback with color-coded risk levels
- **Face Detection Integration**: Implemented Haar Cascade face detection for preprocessing
- **Data Pipeline Architecture**: Designed the data flow from raw images to model predictions
- **Performance Monitoring**: Integrated FPS counter and latency measurement for real-time monitoring

#### Key Deliverables:
- ✅ `realtime_detection.py` - Complete real-time detection system
- ✅ System architecture documentation
- ✅ Data pipeline diagrams
- ✅ Real-time processing performance reports
- ✅ Integration testing between components

#### Technical Skills Demonstrated:
- OpenCV for Computer Vision
- Real-time Processing
- Face Detection Algorithms
- System Architecture Design
- PyTorch Inference Optimization
- Performance Analysis

---

### 3. Anushka Verma (Roll No: 2205712)
**Role:** Testing, Validation & Documentation Lead

#### Primary Contributions:
- **Comprehensive Testing Framework**: Developed `test_snn.py` with:
  - Model accuracy evaluation on test datasets
  - Per-class performance metrics (precision, recall, F1-score)
  - Confusion matrix generation
  - Cross-validation procedures
  - Edge case testing
- **Model Validation**: Implemented rigorous validation pipeline with:
  - Test set evaluation
  - Metrics computation (accuracy, precision, recall, F1-score)
  - Performance visualization
  - Robustness testing
- **Dataset Validation**: Verified data integrity and format consistency
- **Documentation**: Created comprehensive project documentation in `COMPLETE_PROJECT_UNDERSTANDING.md`
- **Quality Assurance**: Ensured code quality through testing and validation

#### Key Deliverables:
- ✅ `test_snn.py` - Comprehensive testing suite
- ✅ `COMPLETE_PROJECT_UNDERSTANDING.md` - Complete project documentation with explanations
- ✅ Test reports and performance metrics
- ✅ Validation procedures documentation
- ✅ Code quality assessment reports

#### Technical Skills Demonstrated:
- PyTorch Model Evaluation
- Statistical Analysis
- Scikit-learn Metrics
- Data Validation
- Testing Methodologies
- Technical Documentation

---

### 4. Kavya Dixit (Roll No: 2205132)
**Role:** Utilities, Data Management & Integration Lead

#### Primary Contributions:
- **Utility Functions**: Developed `utils.py` containing:
  - Common utility functions for image preprocessing
  - Data augmentation techniques
  - Helper functions for model inference
  - Configuration management utilities
- **Data Management**: 
  - Organized training and test datasets
  - Implemented data loading utilities
  - Created data augmentation pipeline for improved model robustness
- **Integration Testing**: Ensured seamless integration between:
  - Training pipeline and model components
  - Real-time detection and trained models
  - Test suite and model evaluation
- **Requirements Management**: Maintained `requirements.txt` with all dependencies
- **Project Documentation**: Updated `README.md` with installation and usage instructions
- **Environment Setup**: Documented environment configuration and dependency management

#### Key Deliverables:
- ✅ `utils.py` - Comprehensive utility functions
- ✅ `requirements.txt` - Complete dependency list
- ✅ `README.md` - User-friendly documentation
- ✅ Data organization and structure
- ✅ Integration documentation

#### Technical Skills Demonstrated:
- Python Development
- Data Preprocessing & Augmentation
- Environment Management
- Package Management
- Software Integration
- Technical Writing

---

## Collaboration & Team Contributions

### Cross-functional Collaboration:
- **Architecture Alignment**: All team members collaborated on overall system architecture design
- **Code Reviews**: Regular peer reviews to ensure code quality and consistency
- **Documentation**: Team-wide documentation efforts for clarity and knowledge transfer
- **Testing & Validation**: Continuous integration and testing throughout development
- **GitHub Coordination**: Regular commits and collaborative branch management

### Shared Responsibilities:
- Dataset organization (All)
- Testing and validation (All)
- Documentation improvements (All)
- GitHub repository management (All)

---

## Project Contribution Summary

| Contributor | Primary Role | Lines of Code | Key Components |
|-------------|--------------|---------------|-----------------|
| Pushkarjay Ajay | ML Architecture | ~800 | train_snn.py, utils.py (model defs) |
| Bhavya Singh | System Design | ~600 | realtime_detection.py, architecture |
| Anushka Verma | Testing & Validation | ~700 | test_snn.py, complete_understanding.md |
| Kavya Dixit | Integration & Utils | ~500 | utils.py (helpers), requirements.txt, README.md |

---

## Technical Stack Contributions

### By Role:
- **ML Framework**: PyTorch (Pushkarjay)
- **Computer Vision**: OpenCV (Bhavya, Kavya)
- **Neural Networks**: SNNTorch, torchvision (Pushkarjay)
- **Testing Framework**: Pytest, Scikit-learn (Anushka)
- **Data Processing**: NumPy, Pandas (Kavya)
- **Documentation**: Markdown, Technical Writing (Anushka, Kavya)

---

## Development Milestones Achieved

1. ✅ **Week 1-2**: Project setup, environment configuration, data organization
2. ✅ **Week 3-4**: Architecture design and system planning
3. ✅ **Week 5-6**: Core model development and training pipeline
4. ✅ **Week 7-8**: Real-time detection implementation
5. ✅ **Week 9-10**: Testing, validation, and optimization
6. ✅ **Week 11**: Documentation and final integration
7. ✅ **Week 12**: Final review and GitHub repository setup

---

## Future Enhancements & Potential Contributions

- **Pushkarjay**: Advanced SNN architectures, continuous learning implementation
- **Bhavya**: Multi-face detection, real-time performance optimization
- **Anushka**: Extended testing frameworks, automated performance regression testing
- **Kavya**: API development for model serving, containerization and deployment

---

## Contact & Support

For questions regarding specific components:
- **ML & Training Issues**: Contact Pushkarjay Ajay (8210164935)
- **Real-time Detection Issues**: Contact Bhavya Singh (7464062560)
- **Testing & Validation**: Contact Anushka Verma (72588961346)
- **Utilities & Setup Issues**: Contact Kavya Dixit (+91 82877 40746)

---

**Last Updated**: April 10, 2026  
**Project Status**: Completed  
**Repository**: GitHub - Facial Expression Recognition (FER)
