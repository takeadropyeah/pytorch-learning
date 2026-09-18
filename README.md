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
