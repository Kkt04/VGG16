# VGG16 Implementation and Inference Project

## Overview
This project demonstrates the implementation of the VGG16 architecture from scratch using TensorFlow/Keras and performs image classification using a pre-trained VGG16 model from PyTorch's torchvision library.

## Project Structure

### 1. VGG16 Architecture Implementation
The project includes a custom implementation of the VGG16 neural network architecture with the following specifications:

#### Network Architecture:
- **5 Convolutional Blocks** with increasing filter sizes (64, 128, 256, 512, 512)
- Each block contains:
  - 2-3 convolutional layers with 3×3 kernels and ReLU activation
  - Max pooling layers with 2×2 windows and stride 2
- **Fully Connected Classifier**:
  - Flatten layer
  - Two 4096-unit dense layers with ReLU activation and 50% dropout
  - Final softmax output layer for 1000-class classification

#### Model Specifications:
- **Total Parameters**: 138,357,544 (≈138.36 million)
- **Model Size**: 527.79 MB
- **Input Shape**: 224×224×3 (RGB images)
- **Output**: 1000 classes (ImageNet)

### 2. Pre-trained VGG16 Inference
The project also demonstrates using a pre-trained VGG16 model from PyTorch for image classification:

#### Features:
- Loads pre-trained VGG16 weights trained on ImageNet
- Implements proper image preprocessing pipeline:
  - Resize to 256×256
  - Center crop to 224×224
  - Normalization using ImageNet statistics
- Performs inference on custom images
- Provides human-readable class labels

#### Image Processing Pipeline:
```python
transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])
```

## Key Components

### Model Architecture Details

#### Convolutional Blocks:
- **Block 1**: 2×[64 filters, 3×3 conv] → MaxPool
- **Block 2**: 2×[128 filters, 3×3 conv] → MaxPool
- **Block 3**: 3×[256 filters, 3×3 conv] → MaxPool
- **Block 4**: 3×[512 filters, 3×3 conv] → MaxPool
- **Block 5**: 3×[512 filters, 3×3 conv] → MaxPool

#### Classifier:
- Flatten → Dense(4096) → Dropout(0.5)
- Dense(4096) → Dropout(0.5) → Dense(1000)

### Technical Notes

1. **Parameter Calculation**: 
   - Pooling layers have 0 parameters as they perform fixed operations
   - Most parameters are in the fully connected layers (≈102.7M in first dense layer)

2. **Implementation Differences**:
   - Custom implementation uses Keras Sequential API
   - Pre-trained model uses PyTorch with slightly different layer ordering
   - Both maintain the same architectural design principles

## Usage

### Building Custom VGG16:
```python
model = build_VGG16(input_shape=(224, 224, 3), num_classes=1000)
model.summary()
```

### Running Inference with Pre-trained Model:
```python
# Load pre-trained model
model = models.vgg16(pretrained=True)
model.eval()

# Preprocess image
input_tensor = preprocess(img).unsqueeze(0)

# Get prediction
with torch.no_grad():
    output = model(input_tensor)
    pred_class = output.argmax(1).item()
    label = labels[pred_class]
```

## Dependencies

- TensorFlow/Keras
- PyTorch
- torchvision
- PIL (Pillow)
- matplotlib
- numpy

## Applications

This implementation can be used for:
- Understanding CNN architectures
- Transfer learning
- Image classification tasks
- Feature extraction
- Model comparison studies

## Notes

- The custom implementation matches the original VGG16 paper specifications
- The pre-trained model achieves state-of-the-art performance on ImageNet
- Both implementations support transfer learning by modifying the final classification layer
- Memory usage is high due to large fully connected layers (characteristic of VGG architecture)
