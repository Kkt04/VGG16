# 🧠 VGG16 Image Classification

## 📌 Overview
This project implements the **VGG16 CNN architecture** using both **TensorFlow/Keras** and **PyTorch**.  
It demonstrates how to build the model, load pretrained weights, preprocess an image, and visualize predictions.

## ⚙️ Features
- VGG16 built from scratch (TensorFlow/Keras)  
- Pretrained VGG16 (PyTorch, ImageNet weights)  
- Image preprocessing and label prediction  
- Visualization of predicted class  

## 🧩 Architecture Summary
- 13 Convolutional layers (3×3 filters)  
- 5 MaxPooling layers  
- 3 Fully Connected layers  
- Softmax output layer  

## 🛠️ Tech Stack
Python, TensorFlow, PyTorch, Torchvision, Matplotlib, Pillow

## 🚀 How to Run
```bash
git clone https://github.com/yourusername/VGG16-Image-Classifier.git
cd VGG16-Image-Classifier
pip install -r requirements.txt
python vgg16_inference.py
