# PyTorch‑Learning
This repository records my pytorch machine‑learning practice codes.

## exp01_linear_regression
Single‑variable linear regression implemented with PyTorch.

Features:
- Automatically select CUDA(GPU) or CPU for training
- Use DataLoader to generate mini‑batches
- Save checkpoint and load weight for inference
- Plot loss curve and fitting result

Ground truth parameter:
w = 2.5, b = 5.2
Gaussian noise was added to the dataset.

## exp02_mlp_binary_classify
Binary classification using multi‑layer perceptron(MLP).
- Construct 2‑D synthetic gaussian dataset
- Two hidden‑layer MLP with ReLU activation
- BCELoss + Adam optimizer
- Save checkpoint and separate inference script
- Visualize training loss and data distribution

## exp03_mnist_mlp
Hand‑written digit recognition on MNIST dataset using MLP.
- Flatten 28×28 grayscale images into 1‑D feature vector as network input
- Multi‑hidden‑layer MLP with ReLU activation for 10‑classes classification
- Cross‑Entropy loss + Adam optimizer
- Plot training loss and accuracy curve for both train and test set
- Save model checkpoint and implement separate inference script
- Observe slight over‑fitting phenomenon: test loss rises after several epochs

## exp04_cnn_mnist
Hand‑written digit recognition on MNIST using simple CNN.
- Build CNN with Conv2d, MaxPool2d and fully‑connected layers
- Compare performance between CNN and previous MLP experiment
- Plot training loss and accuracy curve
- Save checkpoint and separate inference script
- Study weight‑sharing and spatial feature extraction of convolution

## exp05_cnn_regularization
Regularized CNN on MNIST with Dropout and weight‑decay(L2 regularization).
- Add Dropout layer inside fully‑connected part
- Enable weight‑decay in Adam optimizer for L2 penalty
- Compare training curve with exp04 baseline
- Analyze how regularization mitigates over‑fitting
- Save checkpoint and independent inference script
