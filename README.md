# Motor Imagery Classification using Deep Learning

A deep learning approach to classify left vs. right hand motor imagery from EEG brain-computer interface (BCI) data.

## Overview

This project uses a multi-layer perceptron (MLP) neural network to distinguish between left-hand and right-hand motor imagery from 68-channel EEG recordings. Motor imagery classification is a key component in BCI systems that enable direct brain-to-computer communication.

Research paper: https://jhss.scholasticahq.com/article/118171-evaluating-the-efficacy-of-motor-imagery-classifiers-linear-discriminant-analysis-and-a-multi-layer-perceptron-neural-network

## Dataset

The project uses EEG data files containing motor imagery recordings:
- **68 EEG channels** per subject
- **Two classes**: Left-hand imagery (class 0) and Right-hand imagery (class 1)
- **Data format**: MATLAB .mat files (s01.mat through s29.mat)

Each .mat file contains:
- `eeg[0][0][7]`: Left motor imagery data (68 channels × time samples)
- `eeg[0][0][8]`: Right motor imagery data (68 channels × time samples)

## Model Architecture

**Multi-Layer Perceptron Classifier:**
```
Input (68) → 2048 → 1024 → 512 → 256 → 128 → 64 → Output (2)
```

- **Activation**: ReLU
- **Output activation**: Softmax
- **Batch normalization** after each layer
- **Loss function**: Negative Log-Likelihood Loss
- **Optimizer**: Adam (learning rate: 1e-4)

## Installation
```bash
# Clone the repository
git clone https://github.com/TeyjK/motor-imagery-research.git
cd motor-imagery-research

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training
```bash
python train.py --subject 1 --epochs 10 --batch_size 1024 --learning_rate 0.0001
```

**Arguments:**
- `--subject`: Subject number (1-29)
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size for training (default: 1024)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--save_model`: Save trained model (default: True)

### Evaluation
```bash
python evaluate.py --subject 1 --model_path weights/model_subject1.pt
```

**Arguments:**
- `--subject`: Subject number to evaluate
- `--model_path`: Path to trained model weights