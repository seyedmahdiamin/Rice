
# Rice Seed Classification
Deep learning model to categorize rice images according to their types.

This project implements a Convolutional Neural Network (CNN) model to classify images of rice seeds into different categories. The project uses TensorFlow to preprocess the data, build the model, and evaluate its performance.

## Table of Contents

- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)

## Project Overview

The goal of this project is to classify images of rice seeds into different categories. The model is trained on a dataset of rice seed images, which is split into training, validation, and test sets. The performance of the model is evaluated using a confusion matrix and accuracy metrics.

## Directory Structure

```
rice-classification/
│
├── data/                                 # Directory containing the dataset
├── artifacts/                            # Directory to store training artifacts such as logs and models
├── rice_classification.ipynb             # jupyter notebook to train the model
├── README.md                             # This file
```

## Dataset

The dataset consists of images of rice seeds stored in different subdirectories, each representing a different class. The images are resized to 128x128 pixels before being fed into the model.

- **Training Set**: 70% of the total images
- **Validation Set**: 15% of the total images
- **Test Set**: 15% of the total images

## Model Architecture

The model is built using the following layers:

1. **Input Layer**: Accepts input images of size 128x128x3 (RGB).
2. **Data Augmentation**: Includes random vertical and horizontal flipping, and contrast adjustment.
3. **Convolutional Layers**: Four convolutional layers with increasing filter sizes (32, 64, 64, 128), each followed by batch normalization, ReLU activation, and dropout.
4. **Global Max Pooling**: Reduces the dimensionality of the feature maps.
5. **Dense Layer**: A fully connected layer with softmax activation for classification.

## Training and Evaluation

- The model is compiled with the Adam optimizer and Sparse Categorical Crossentropy loss.
- The model is trained for 500 epochs with early stopping and model checkpointing based on validation accuracy.
- TensorBoard is used for monitoring the training process.

### Training Samples Visualization

The following figure shows a subset of the training samples:

![Training Samples](/assets/Training%20Samples.png)

### Confusion Matrix

The confusion matrix below shows the performance of the model on the test set, illustrating how well the model distinguishes between different rice seed categories:

![Confusion Matrix](/assets/Confusion%20Matrix.png)

## Results

- **Accuracy**: The model achieved an accuracy of `97.73%` on the test set.
- **Loss**: The final loss on the test set was `0.08946`.
