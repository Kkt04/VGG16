🧠 VGG16 Image Classification Project
📌 Overview

This project implements the VGG16 Convolutional Neural Network (CNN) architecture using both TensorFlow/Keras and PyTorch.
It demonstrates how deep learning models can be built, loaded, and used for image classification tasks.

The project includes:

VGG16 built from scratch using TensorFlow

Use of pretrained VGG16 (PyTorch) for inference

Image preprocessing and visualization of predictions

Foundation for extending into custom datasets and transfer learning

🎯 Objectives

Understand and implement the VGG16 architecture layer-by-layer

Learn how to use pretrained models for feature extraction and prediction

Explore image preprocessing pipelines for deep learning

Serve as a base project for transfer learning or fine-tuning

🧩 Architecture Overview

VGG16 is a deep convolutional neural network developed by Simonyan & Zisserman (2014).
It consists of 16 weight layers — convolutional, pooling, and fully connected layers.

Structure Summary:

Input: 224×224 RGB image

13 Convolutional Layers (3×3 filters)

5 MaxPooling Layers

3 Fully Connected Layers

Output: Softmax with 1000 classes (ImageNet)

⚙️ Tech Stack

Python 3.x

TensorFlow / Keras

PyTorch

Torchvision

Matplotlib

PIL (Pillow)

🧾 How It Works

VGG16 from Scratch (TensorFlow/Keras)
Builds the architecture manually using Conv2D, MaxPooling2D, and Dense layers.

Pretrained VGG16 (PyTorch)

Loads ImageNet pretrained weights

Preprocesses an image (resize, crop, normalize)

Performs inference to predict the class label

Displays the image with predicted category name

📸 Example Output
Predicted label: "Egyptian Cat"


🚀 How to Run the Code

Clone the repository

git clone https://github.com/yourusername/VGG16-Image-Classifier.git
cd VGG16-Image-Classifier


Install dependencies

pip install -r requirements.txt


Run the script

python vgg16_inference.py


Or open the notebook

jupyter notebook VGG16_Project.ipynb

📊 Future Improvements

Train on a custom dataset (e.g., CIFAR-10, Cats vs Dogs)

Add transfer learning and fine-tuning

Visualize Grad-CAM heatmaps for model interpretability

Deploy a Streamlit web app for live predictions

📚 References

Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition (arXiv:1409.1556)

TensorFlow VGG16 Documentation

PyTorch VGG16 Documentation
