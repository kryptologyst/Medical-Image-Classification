# Medical Image Classification

**RESEARCH DEMO ONLY - NOT FOR CLINICAL USE**

This project demonstrates deep learning techniques for medical image classification using chest X-ray data. It showcases modern AI approaches for radiology support tools and clinical decision support research.

## ⚠️ IMPORTANT DISCLAIMER

**THIS SOFTWARE IS FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY**

- NOT approved for clinical diagnosis or treatment decisions
- NOT a medical device
- NOT validated for clinical use
- NOT intended to replace professional medical judgment

**Always consult qualified healthcare professionals for medical decisions.**

See [DISCLAIMER.md](DISCLAIMER.md) for complete terms.

## Features

- **Modern Architecture**: EfficientNet, Vision Transformer, ResNet variants
- **Medical Imaging Pipeline**: DICOM/NIfTI support, proper preprocessing
- **Clinical Metrics**: AUROC, sensitivity, specificity, calibration
- **Explainability**: Grad-CAM, uncertainty estimation
- **Interactive Demo**: Streamlit web interface
- **Research Ready**: Comprehensive evaluation, reproducible results

## Quick Start

### Installation

```bash
# Clone and setup
git clone https://github.com/kryptologyst/Medical-Image-Classification.git
cd Medical-Image-Classification

# Install dependencies
pip install -r requirements.txt

# Run demo
streamlit run demo/app.py
```

### Training

```bash
# Train with default config
python scripts/train.py --config configs/default.yaml

# Train with custom settings
python scripts/train.py --config configs/efficientnet.yaml --epochs 50
```

### Evaluation

```bash
# Evaluate trained model
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth

# Generate clinical report
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --report
```

## Dataset

This project uses synthetic medical imaging data for demonstration. For real research:

1. **ChestX-ray14**: Large-scale chest X-ray dataset
2. **COVIDx**: COVID-19 detection dataset  
3. **SIIM-ACR**: Pneumothorax detection challenge

### Data Format

```
data/
├── train/
│   ├── normal/
│   └── abnormal/
├── val/
│   ├── normal/
│   └── abnormal/
└── test/
    ├── normal/
    └── abnormal/
```

## Models

- **ResNet18/50**: Baseline CNN architectures
- **EfficientNet-B0/B3**: Efficient scaling for medical images
- **Vision Transformer**: Transformer-based classification
- **Custom CNN**: Optimized for medical imaging

## Evaluation Metrics

### Classification Metrics
- **AUROC**: Area under ROC curve
- **AUPRC**: Area under precision-recall curve
- **Sensitivity**: True positive rate
- **Specificity**: True negative rate
- **PPV/NPV**: Positive/negative predictive value

### Calibration
- **Brier Score**: Probability calibration
- **ECE**: Expected calibration error
- **Reliability Diagrams**: Visual calibration assessment

## Project Structure

```
├── src/                    # Source code
│   ├── models/            # Model definitions
│   ├── data/              # Data loading and preprocessing
│   ├── losses/            # Loss functions
│   ├── metrics/           # Evaluation metrics
│   ├── utils/             # Utilities
│   └── train/             # Training scripts
├── configs/               # Configuration files
├── scripts/               # Training and evaluation scripts
├── demo/                  # Streamlit demo application
├── tests/                 # Unit tests
├── notebooks/             # Jupyter notebooks
├── assets/                # Sample outputs and visualizations
└── checkpoints/           # Model checkpoints
```

## Configuration

Models and training can be configured via YAML files:

```yaml
# configs/default.yaml
model:
  name: "efficientnet_b0"
  pretrained: true
  num_classes: 2

data:
  batch_size: 32
  image_size: [224, 224]
  augmentation: true

training:
  epochs: 100
  learning_rate: 0.001
  weight_decay: 1e-4
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper tests
4. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{medical_image_classification,
  title={Medical Image Classification - Research Project},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Medical-Image-Classification}
}
```

## Acknowledgments

- MONAI for medical imaging utilities
- PyTorch for deep learning framework
- ChestX-ray14 dataset contributors
- Medical imaging research community
# Medical-Image-Classification
