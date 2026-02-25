// filepath: /home/aayam/CNN/README.md
# CNN Image Classification

A comprehensive deep learning project implementing Convolutional Neural Networks (CNNs) for image classification tasks using PyTorch. This repository includes custom CNN architectures and transfer learning with ResNet18.

## 📋 Overview

This project demonstrates multiple approaches to image classification:
- **Custom CNNs**: Built from scratch for CIFAR-10 and Fashion MNIST datasets
- **Transfer Learning**: Fine-tuned ResNet18 for CIFAR-10 classification

## 📁 Project Structure

```
CNN/
├── src/
│   ├── CIFAR10.ipynb         # Custom CNN for CIFAR-10 (200 epochs)
│   ├── FashionMNIST.ipynb    # Custom CNN for Fashion MNIST (30 epochs, 90.69% accuracy)
│   ├── resnet18.ipynb        # Transfer learning with ResNet18 (15 epochs)
│   ├── best_model.pth        # Best model weights for Fashion MNIST
│   ├── good_model.pth        # Good model weights for ResNet18
│   └── data/
│       └── cifar-10-batches-py/  # CIFAR-10 dataset
├── data/
│   └── FashionMNIST/
│       └── raw/              # FashionMNIST dataset
├── best_model.pth            # Best CIFAR-10 model weights
├── environment.yml           # micromamba environment configuration
├── LICENSE                   # MIT License
└── README.md                 # This file
```

## 🎯 Datasets

### CIFAR-10
- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Images**: 60,000 total (50,000 train, 10,000 test)
- **Size**: 32×32 RGB images
- **Data Augmentation**: Random cropping, horizontal flipping

### Fashion MNIST
- **Classes**: 10 fashion item categories
- **Images**: 70,000 total (60,000 train, 10,000 test)
- **Size**: 28×28 grayscale images
- **Best Accuracy**: 90.69%

## 🏗️ Models

### Custom CNN Architecture

#### CIFAR-10 Model
```
Conv2d(3, 32) → BatchNorm → ReLU → (stride 1)
Conv2d(32, 64) → BatchNorm → ReLU → (stride 2)
Conv2d(64, 128) → BatchNorm → ReLU → (stride 1)
Conv2d(128, 256) → BatchNorm → ReLU → (stride 2)
Conv2d(256, 512) → BatchNorm → ReLU → (stride 1)
AdaptiveAvgPool2d → Flatten → Dropout(0.2) → Linear(512, 10)
```

#### Fashion MNIST Model
```
Conv2d(1, 16) → BatchNorm → ReLU → (stride 1)
Conv2d(16, 32) → BatchNorm → ReLU → (stride 2)
Conv2d(32, 64) → BatchNorm → ReLU → (stride 1)
Conv2d(64, 128) → BatchNorm → ReLU → (stride 2)
Conv2d(128, 256) → BatchNorm → ReLU → (stride 1)
AdaptiveAvgPool2d → Flatten → Dropout(0.2) → Linear(256, 10)
```

### Transfer Learning (ResNet18)
- **Base Model**: ResNet18 (pretrained on ImageNet)
- **Modifications**: Modified final fully connected layer for 10 classes
- **Training Strategy**: Progressive layer unfreezing
- **Input Size**: 224×224 RGB images

## 📊 Training Details

### CIFAR-10 (Custom CNN)
- **Epochs**: 200
- **Batch Size**: 128
- **Learning Rate**: Adaptive (ReduceLROnPlateau)
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Train/Val Split**: 40,000/10,000

### Fashion MNIST (Custom CNN)
- **Epochs**: 30
- **Batch Size**: 128
- **Learning Rate**: 0.001 with ReduceLROnPlateau scheduler
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Train/Val Split**: 50,000/10,000
- **Best Val Loss**: 0.2688
- **Test Accuracy**: 90.69%

### ResNet18 (Transfer Learning)
- **Epochs**: 15
- **Batch Size**: 128
- **Batch Sizes**: 128
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Training Strategy**: 
  - Progressive layer unfreezing
  - Fine-tuning classifier first, then gradual unfreezing of backbone layers
- **Data Augmentation**: Random crop, horizontal flip, resize to 224×224

## 🚀 Getting Started

### Prerequisites
- Python 3.11.9
- CUDA 12.1 (for GPU acceleration)
- conda or micromamba

### Installation

1. Clone the repository
```bash
git clone <repository-url>
cd CNN
```

2. Create the conda environment
```bash
conda env create -f environment.yml
conda activate mnist
```

3. Launch Jupyter Lab
```bash
jupyter lab
```

### Running the Notebooks

1. **CIFAR-10 Custom CNN**: Open `src/CIFAR10.ipynb`
2. **Fashion MNIST**: Open `src/FashionMNIST.ipynb`
3. **ResNet18 Transfer Learning**: Open `src/resnet18.ipynb`

## 🔧 Key Features

- **Data Normalization**: Custom normalization statistics for CIFAR-10
  - Mean: (0.4914, 0.4822, 0.4465)
  - Std: (0.2470, 0.2435, 0.2616)

- **Batch Normalization**: Applied after each convolution layer for stable training

- **Adaptive Learning Rate Scheduling**: ReduceLROnPlateau to avoid overfitting

- **Model Checkpointing**: Saves best models during training

- **GPU Acceleration**: Automatic CUDA device detection and usage

## 📈 Performance Metrics

### Fashion MNIST Highlights
- Training converges well with minimal overfitting
- Final validation loss: ~0.40
- Test accuracy: **90.69%**

### CIFAR-10 Highlights
- 200 epochs of training on custom architecture
- Progressive improvement in loss across all epochs
- Best model weights saved for inference

### ResNet18 Results
- Effective transfer learning with pre-trained weights
- Quick convergence due to ImageNet pre-training
- Validation loss plateau around 0.31

## 💡 Techniques Used

- **Convolutional Neural Networks (CNNs)**: Multi-layer convolution for feature extraction
- **Transfer Learning**: Leveraging pre-trained ResNet18 weights
- **Data Augmentation**: Random cropping and horizontal flipping
- **Batch Normalization**: Stabilizes training and improves convergence
- **Dropout**: Prevents overfitting with 0.2 dropout rate
- **Adaptive Average Pooling**: Global feature aggregation
- **Mixed Precision Training** (ResNet18): GradScaler for efficient GPU usage

## 🛠️ Dependencies

Key libraries:
- **PyTorch 2.5.1**: Deep learning framework
- **TorchVision 0.20.1**: Computer vision utilities
- **NumPy 2.4.2**: Numerical computing
- **Matplotlib 3.10.8**: Visualization
- **Jupyter Lab 4.5.3**: Development environment

See [environment.yml](environment.yml) for complete dependencies.

## 📝 Usage Examples

### Loading a Pre-trained Model

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
```

### Making Predictions

```python
model.eval()
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)
```

## 🔍 Model Training Loop

All notebooks follow a consistent training pattern:
1. **Forward Pass**: Compute predictions
2. **Loss Calculation**: CrossEntropyLoss
3. **Backward Pass**: Compute gradients
4. **Optimization**: Update weights with Adam optimizer
5. **Validation**: Evaluate on validation set
6. **Checkpointing**: Save best model based on validation loss

## 📚 References

- [PyTorch Documentation](https://pytorch.org/docs)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ✨ Highlights

- Clean, well-documented Jupyter notebooks
- Comprehensive CNN architectures with batch normalization
- Effective transfer learning implementation
- GPU acceleration support
- High accuracy on Fashion MNIST (90.69%)

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for any improvements.

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---

**Environment**: Python 3.11.9 with CUDA 12.1