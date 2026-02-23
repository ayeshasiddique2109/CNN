# CIFAR-10 Image Classification using CNN (PyTorch)

## Dataset Information

* **Dataset:** CIFAR-10 (Downloaded automatically via `torchvision.datasets`).
* **Images:** 60,000 total (50,000 Train / 10,000 Test).
* **Resolution:** 32x32 pixels, RGB color channels.
* **Classes:** 10 (Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck).

##  Key Transformations (Data Augmentation)

In `cnn_modified.py`, the script utilizes the `torchvision.transforms` library to apply stochastic transformations to the training set:

* **Flipping:** `RandomHorizontalFlip()` enables horizontal mirroring to simulate different perspectives.
* **Cropping:** `RandomCrop(32, padding=4)` randomly crops the image to improve spatial invariance.
* **Normalization:** Scales pixel values using a mean and standard deviation of 0.5 for all three (RGB) channels.
* **Tensor Conversion:** Converts raw images into PyTorch 4D Tensors (Batch, Channels, Height, Width).

## Model Architectures

### 1. Basic CNN (`cnn_basic.py`)

A foundational model used to establish baseline accuracy.

* **Conv Layers:** 2 Layers (32 and 64 filters).
* **Normalization:** `BatchNorm2d` used after each convolution.
* **Regularization:** `Dropout(0.4)` and L2 Weight Decay ().
* **Fully Connected:** 128 neurons in the hidden layer.

### 2. Modified CNN (`cnn_modified.py`)

An improved version focusing on deeper feature extraction and better generalization.

* **Conv Layers:** 3 Layers (32, 64, and 128 filters).
* **Improved Depth:** Added a third convolutional layer for complex feature detection.
* **Regularization:** Increased `Dropout(0.5)` to prevent overfitting.
* **Fully Connected:** Three layers (256 → 128 → 10) for refined classification.
