# FashionMNIST CNN Image Classification using Keras

## Overview

This project implements a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify images from the FashionMNIST dataset.

After training the model on FashionMNIST, the trained CNN is tested on 10 real-world images captured using a smartphone.

## Dataset

FashionMNIST contains 10 classes:

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

## CNN Architecture

The model contains:

* Conv2D layers
* ReLU activation
* MaxPooling2D
* Flatten
* Dense layers
* Dropout
* Softmax output layer

### Architecture

```text
Input: 28 × 28 × 1
       ↓
Conv2D: 32 filters
       ↓
ReLU
       ↓
MaxPooling
       ↓
Conv2D: 64 filters
       ↓
ReLU
       ↓
MaxPooling
       ↓
Flatten
       ↓
Dense: 128
       ↓
ReLU
       ↓
Dropout
       ↓
Dense: 10
       ↓
Softmax
```

## Preprocessing

The standard FashionMNIST images are processed using:

* Resize to 28 × 28
* Grayscale
* Pixel scaling
* Normalization

The same preprocessing pipeline is applied to the custom smartphone images.

## Training

The model uses:

* Optimizer: Adam
* Learning Rate: 0.001
* Loss: Sparse Categorical Crossentropy
* Batch Size: 64
* Epochs: 5

## Real-World Testing

Ten custom smartphone photographs are stored in the `dataset/` directory.

The notebook automatically clones the GitHub repository and loads the images without requiring manual file uploads.

## Results

The project includes:

* Training Loss vs Epoch
* Validation Loss vs Epoch
* Training Accuracy vs Epoch
* Validation Accuracy vs Epoch
* Test Set Confusion Matrix
* Custom Prediction Gallery
* Prediction Confidence
* Error Analysis

## Repository Structure

```text
fashionmnist-cnn-keras/
│
├── dataset/
│   ├── tshirt.jpg
│   ├── trouser.jpg
│   ├── pullover.jpg
│   ├── dress.jpg
│   ├── coat.jpg
│   ├── sandal.jpg
│   ├── shirt.jpg
│   ├── sneaker.jpg
│   ├── bag.jpg
│   └── ankle_boot.jpg
│
├── model/
│   └── fashion_mnist_cnn.keras
│
├── 220134.ipynb
├── model.py
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

## Conclusion

This project demonstrates a complete image classification workflow using Keras. The model is trained on the standard FashionMNIST dataset and then evaluated on real-world smartphone photographs.

The difference between the standard test performance and real-world performance demonstrates the challenge of applying a model trained on controlled dataset images to real-world images with different backgrounds, lighting, viewpoints, and object appearances.
