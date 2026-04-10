# Roles & Responsibilities - Facial Expression Recognition (FER) Project

---

## Overview

This document outlines the individual roles and responsibilities for the Facial Expression Recognition (FER) project team. Each team member has specific areas of ownership while maintaining collaborative relationships with other team members.

---

## Team Composition

| Name | Roll No. | Contact | Primary Role |
|------|----------|---------|--------------|
| Pushkarjay Ajay | 22052328 | 8210164935 | ML Architecture & Model Development |
| Bhavya Singh | 2205120 | 7464062560 | System Design & Real-time Processing |
| Anushka Verma | 2205712 | 72588961346 | Testing & Validation |
| Kavya Dixit | 2205132 | +91 82877 40746 | Utilities & Integration |

---

## Role Structure & Responsibilities

### 1. Pushkarjay Ajay - ML Architecture & Model Development Lead

#### Primary Responsibilities:
- Design and implement the hybrid CNN-SNN architecture
- Develop and optimize deep learning models for emotion recognition
- Conduct experiments with different hyperparameters and architectures
- Implement ensemble learning mechanisms
- Optimize model performance and inference speed
- Handle GPU acceleration and CUDA integration
- Troubleshoot deep learning-specific issues

#### Key Deliverables:
- **Model Architecture** (`utils.py` - model definitions)
  - CNN feature extractor implementation
  - SNN temporal processing layer
  - Ensemble voting mechanism
  
- **Training Pipeline** (`train_snn.py`)
  - Data loading and normalization
  - Training loop with loss computation
  - Model checkpointing and version control
  - Hyperparameter optimization
  - Ensemble model creation (3 models)
  
- **Trained Models** (model directory)
  - `snn_model_1.pth` - First ensemble model
  - `snn_model_2.pth` - Second ensemble model
  - `snn_model_3.pth` - Third ensemble model
  
- **Documentation**
  - Model architecture diagrams
  - Training procedures and logs
  - Performance benchmarks

#### Technical Skills & Expertise:
- PyTorch Deep Learning Framework
- Spiking Neural Networks (SNNTorch)
- CNN Architecture Design
- Ensemble Learning Methods
- Model Optimization Techniques
- CUDA & GPU Computing
- Hyperparameter Tuning

#### Success Metrics:
- Model Accuracy: ≥ 75% overall
- Inference Speed: < 100ms per image
- Training Stability: Smooth convergence
- Ensemble Robustness: Improved over individual models

#### Collaboration Points:
- Works with **Bhavya**: Integrates models into real-time system
- Works with **Anushka**: Provides models for testing and validation
- Works with **Kavya**: Ensures models work with utility functions

---

### 2. Bhavya Singh - System Design & Real-time Processing Lead

#### Primary Responsibilities:
- Design the overall system architecture and data flow
- Implement real-time emotion detection from webcam streams
- Develop face detection and preprocessing pipeline
- Optimize system for real-time performance
- Implement visualization and user feedback
- Handle video frame processing
- Ensure seamless integration of all components

#### Key Deliverables:
- **Real-time Detection System** (`realtime_detection.py`)
  - Webcam stream handling
  - Face detection with Haar Cascade
  - Real-time frame preprocessing
  - Model inference orchestration
  - Visualization with OpenCV
  - Risk assessment and color-coding
  - FPS and latency monitoring
  
- **System Architecture**
  - Architecture diagrams (context, component, data flow)
  - System design documentation
  - Integration specifications
  - Performance requirements definition
  
- **User Interface**
  - Bounding box visualization
  - Emotion label display
  - Confidence score rendering
  - Risk level indicators (color-coded)
  - Control interface (keyboard shortcuts)
  - Real-time metrics display

#### Technical Skills & Expertise:
- OpenCV for Computer Vision
- Real-time Processing & Stream Handling
- Face Detection Algorithms
- System Architecture Design
- PyTorch Model Integration
- Performance Optimization
- Video Processing

#### Success Metrics:
- Real-time FPS: ≥ 15 fps
- Face Detection Accuracy: ≥ 95%
- Latency: < 100ms per frame
- Robustness: Handles 1-5 faces simultaneously
- UI Responsiveness: Smooth visualization

#### Collaboration Points:
- Works with **Pushkarjay**: Loads and runs inference with trained models
- Works with **Anushka**: Provides real-time system for end-to-end testing
- Works with **Kavya**: Uses utility functions for preprocessing

---

### 3. Anushka Verma - Testing & Validation Lead

#### Primary Responsibilities:
- Develop comprehensive testing framework
- Create test cases for all components
- Conduct model validation and evaluation
- Measure and report performance metrics
- Ensure code quality and correctness
- Create testing documentation
- Perform edge case and robustness testing

#### Key Deliverables:
- **Testing Suite** (`test_snn.py`)
  - Unit tests for model components
  - Integration tests for pipeline
  - Model accuracy evaluation
  - Per-class performance metrics (precision, recall, F1-score)
  - Confusion matrix generation
  - Statistical validation tests
  
- **Test Coverage**
  - Preprocessing tests
  - Model inference tests
  - Ensemble voting tests
  - Risk assessment tests
  - Edge case handling tests
  
- **Performance Reports**
  - Accuracy metrics by emotion class
  - Confusion matrices and visualizations
  - Cross-validation results
  - Performance benchmarks
  - Quality assurance checklist
  
- **Documentation**
  - Testing standards and procedures
  - Test case documentation
  - Coverage reports
  - Validation results

#### Technical Skills & Expertise:
- PyTorch Model Evaluation
- Statistical Analysis & Metrics
- Scikit-learn Performance Metrics
- Data Validation Techniques
- Testing Methodologies (Unit, Integration)
- Quality Assurance
- Technical Documentation

#### Success Metrics:
- Test Coverage: ≥ 85%
- Model Accuracy: ≥ 75%
- Per-class F1-score: ≥ 0.70
- All critical tests passing
- Edge cases handled gracefully

#### Collaboration Points:
- Works with **Pushkarjay**: Tests trained models and validates results
- Works with **Bhavya**: End-to-end system testing
- Works with **Kavya**: Tests utility functions and data handling

---

### 4. Kavya Dixit - Utilities & Integration Lead

#### Primary Responsibilities:
- Develop utility functions and helper modules
- Manage data organization and loading
- Handle dependencies and environment setup
- Ensure seamless component integration
- Maintain project documentation and README
- Manage requirements and package dependencies
- Handle data augmentation and preprocessing helpers

#### Key Deliverables:
- **Utility Module** (`utils.py` - utility functions)
  - Image preprocessing functions
  - Data loading utilities
  - Data augmentation techniques
  - Helper functions for model inference
  - Configuration management
  - Device-agnostic code utilities
  
- **Data Management**
  - Organized training data structure
  - Test data organization
  - Data directory configurations
  - Dataset loading utilities
  
- **Dependencies & Environment** (`requirements.txt`)
  - All required Python packages
  - Specific package versions
  - Optional dependencies documentation
  - Installation instructions
  
- **Project Documentation** (`README.md`)
  - Project overview
  - Installation guide
  - Usage instructions
  - Project structure documentation
  - Troubleshooting guide
  - Contributing guidelines
  
- **Integration & Testing**
  - Component integration testing
  - End-to-end workflow validation
  - Cross-component compatibility
  - Environment compatibility

#### Technical Skills & Expertise:
- Python Development
- Data Preprocessing & Augmentation
- Environment & Dependency Management
- Package Management (pip)
- Integration Testing
- Software Documentation
- Data Structure Design

#### Success Metrics:
- All dependencies properly documented
- utilities module is reusable across components
- Data loading works seamlessly
- README is clear and complete
- Integration tests passing

#### Collaboration Points:
- Works with **Pushkarjay**: Provides utility functions for model training
- Works with **Bhavya**: Supports real-time preprocessing pipeline
- Works with **Anushka**: Provides utility functions for testing

---

## Shared Responsibilities

### 1. Code Quality & Standards
- **All Team Members** responsible for:
  - Following PEP 8 coding standards
  - Writing clear and documented code
  - Maintaining consistent code style
  - Reviewing peer code changes

### 2. Documentation
- **All Team Members** responsible for:
  - Creating docstrings for functions/classes
  - Maintaining component documentation
  - Updating README and guides
  - Documenting design decisions

### 3. Testing
- **All Team Members** responsible for:
  - Unit testing their own code
  - Reporting bugs and issues
  - Participating in code reviews
  - Ensuring their code is testable

### 4. Version Control
- **All Team Members** responsible for:
  - Regular commits with clear messages
  - Creating feature branches
  - Handling merge conflicts professionally
  - Reviewing pull requests

### 5. Communication
- **All Team Members** responsible for:
  - Regular status updates
  - Clear documentation of changes
  - Prompt response to questions
  - Collaborative problem-solving

---

## Development Phases & Milestones

### Phase 1: Setup & Planning (Weeks 1-2)
- **Lead**: Kavya Dixit
- **Activities**: Environment setup, data organization, architecture planning
- **Deliverable**: Development environment ready, project structure defined

### Phase 2: Core Development (Weeks 3-8)
- **Module 1 (Pushkarjay)**: Model development and training
- **Module 2 (Bhavya)**: Real-time detection system
- **Module 3 (Kavya)**: Utilities and integration
- **Deliverable**: All core modules completed and functional

### Phase 3: Testing & Validation (Weeks 9-10)
- **Lead**: Anushka Verma
- **Activities**: Comprehensive testing, performance validation, bug fixes
- **Deliverable**: All tests passing, performance metrics documented

### Phase 4: Documentation & Finalization (Weeks 11-12)
- **Lead**: Kavya Dixit
- **Activities**: Final documentation, report writing, repository preparation
- **Deliverable**: Complete documentation, ready for submission

---

## Communication Protocol

### Regular Meetings:
- **Weekly Sync**: Every Monday, 10:00 AM
  - Progress updates
  - Blocker identification
  - Coordination on dependencies

- **Technical Discussions**: As needed
  - Architecture decisions
  - Complex problem-solving
  - Integration issues

### Communication Channels:
- **Primary**: GitHub Issues and Pull Requests
- **Secondary**: Email (for urgent matters)
- **Tertiary**: Phone (for critical issues)

### Documentation Requirements:
- Commit messages must be clear and descriptive
- Pull requests must have comprehensive descriptions
- All code must be documented before merging
- Monthly progress reports required

---

## Conflict Resolution

### Process:
1. **Direct Communication**: Team members discuss and resolve directly
2. **Technical Lead Mediation**: If needed, escalate to Bhavya Singh
3. **Group Discussion**: Bring to team meeting for resolution
4. **External Help**: Seek assistance from mentor/supervisor if needed

---

## Performance Evaluation Criteria

### For Pushkarjay Ajay:
- Model accuracy and performance
- Code quality and documentation
- Timely completion of deliverables
- Technical depth and innovation

### For Bhavya Singh:
- Real-time system performance (FPS, latency)
- System stability and robustness
- Integration quality
- Architecture clarity

### For Anushka Verma:
- Test coverage percentage
- Test case quality
- Bug identification rate
- Documentation completeness

### For Kavya Dixit:
- Utility function efficiency and reusability
- Documentation quality
- Integration success
- Dependency management

---

## Escalation Path

1. **Component-Level Issues**: Owned by individual team member
2. **Integration Issues**: Coordinate between relevant team members
3. **Architectural Issues**: Escalate to Bhavya Singh (System Architecture Lead)
4. **Project-Level Issues**: Escalate to team lead (Kavya Dixit)

---

## Knowledge Sharing & Handover

### Documentation Areas:
- **Pushkarjay**: Model training procedures, hyperparameters, benchmarks
- **Bhavya**: System architecture, real-time optimization techniques
- **Anushka**: Testing procedures, validation methodologies, metrics
- **Kavya**: Setup instructions, dependency management, integration points

### Code Documentation:
- All functions must have docstrings
- Complex algorithms must have inline comments
- Design decisions documented in README or inline
- Performance-critical sections documented with benchmarks

---

## Success Criteria for Project Completion

✅ All team members have defined roles and responsibilities  
✅ Individual contributions are clear and measurable  
✅ Regular communication and collaboration maintained  
✅ Code quality and documentation standards met  
✅ Performance requirements achieved  
✅ Comprehensive testing completed  
✅ Final deliverables submitted on time  
✅ Knowledge successfully transferred for maintenance  

---

**Document Version**: 1.0  
**Last Updated**: April 10, 2026  
**Status**: Final  
**Project Lead**: Kavya Dixit  
**System Architecture Lead**: Bhavya Singh
