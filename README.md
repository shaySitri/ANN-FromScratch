# Artificial Neural Network (ANN) from Scratch 🧠💻

[![Python Script](https://img.shields.io/badge/Script-ann_from_scratch.py-blue)](https://github.com/shaySitri/ANN-FromScratch/blob/main/ann_from_scratch.py)

This repository implements an **Artificial Neural Network (ANN) from scratch** in Python. The project demonstrates the inner workings of a neural network by building forward and backward propagation, as well as optimization, without using high-level libraries like TensorFlow or PyTorch.

---

## 📄 Project Overview
The ANN model is applied to the **MNIST dataset**, which contains grayscale images of handwritten digits (0–9). The goal is to classify each image into one of the 10 digit classes. This project covers data preprocessing, network initialization, training, and evaluation of a custom-built neural network.

---

## 📊 Data Insights
- **Dataset**: MNIST, containing 28x28 grayscale images of digits (flattened into 784 input features).
- **Classes**: 10 (digits 0–9).
- **Preprocessing**:
  - **One-Hot Encoding**: Encodes labels as vectors.
  - **Normalization**: Scales pixel values to [0, 1] range.
  - **Data Split**: Training, validation, and test sets.

---

## 🧩 Model Architecture
- **Layers**:
  - Input Layer: 784 nodes (one per pixel).
  - Hidden Layers: 3 hidden layers with sizes `[20, 7, 5]`.
  - Output Layer: 10 nodes (one for each digit class).
- **Activation Functions**:
  - **ReLU**: For hidden layers.
  - **Softmax**: For output layer to calculate class probabilities.

---

## 🔄 Training Process
1. **Forward Propagation**:
   - Computes activations for each layer using ReLU and Softmax.
2. **Loss Calculation**:
   - Uses **Cross-Entropy Loss** to quantify the difference between predictions and true labels.
3. **Backward Propagation**:
   - Updates weights based on gradient descent to minimize the loss.
4. **Parameter Updates**:
   - Uses learning rate and calculated gradients to adjust weights.

### Hyperparameters
- **Batch Size**: Variable (experiments conducted with 32, 64, 128, and 256).
- **Learning Rate**: 0.009
- **Stopping Criterion**: Early stopping applied if no improvement after set validation checks.

---

## 📈 Experimental Results
- **Batch Size Experiments**: Models with larger batch sizes (e.g., 256) achieved smoother and faster convergence.
- **Regularization and Batch Normalization**:
  - **Batch Normalization**: Improved training stability but showed limited impact on validation accuracy.
  - **L2 Regularization**: Applied to prevent overfitting but required careful tuning to avoid training disruptions.

---

## 🛠️ Tools and Technologies Used
- **Python** 🐍
- **NumPy**: For matrix operations and efficient numerical calculations.
- **Matplotlib**: For visualization of training/validation loss and accuracy.

---

## 🔍 Key Findings
- **Training Stability**: Batch normalization provided more stable learning dynamics.
- **Impact of Regularization**: L2 regularization improved generalization but could hinder training if too strong.
- **Optimal Batch Size**: Larger batch sizes reduced training time and improved accuracy.
