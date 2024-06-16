# Customer_Churn_Prediction

This repository contains a neural network model implemented in Keras to predict customer churn. Customer churn prediction is crucial for businesses to identify customers who are likely to leave, allowing proactive retention strategies.

## Table of Contents

- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Customer churn refers to the phenomenon where customers stop doing business with a company. Predicting churn can help businesses take preventive actions to retain customers, thereby reducing revenue loss. This repository provides a deep learning solution using a neural network built with Keras to predict customer churn based on available features.

## Model Architecture

The neural network model architecture is designed as follows:

- **Input Layer**: 11 neurons with ReLU activation function, corresponding to the 11 input features.
- **Batch Normalization**: Applied after each dense layer to normalize the activations.
- **Dropout**: Regularization technique with a dropout rate of 0.5 to prevent overfitting.
- **Hidden Layers**: Two hidden layers, each with 11 neurons and ReLU activation.
- **Output Layer**: 1 neuron with sigmoid activation for binary classification (churn or not churn).

The model is compiled using the Adam optimizer with a learning rate of 0.001 and binary crossentropy loss function, suitable for binary classification tasks.

## Dataset

The model is trained and evaluated using a dataset that includes features relevant to predicting customer churn. Ensure the dataset is appropriately preprocessed and split into training and testing sets before training the model.

## Usage

To use this model:
 
1. **Install Dependencies**: Ensure you have Keras, TensorFlow, and other required libraries installed (`pip install -r requirements.txt`).

2. **Prepare Data**: Replace placeholder data with your actual dataset for customer churn prediction.

3. **Train the Model**: Run the training script (`python train.py`) to train the model on your dataset.

4. **Evaluate the Model**: Evaluate the model's performance using metrics such as accuracy on your test dataset.

## Contributing

Contributions to improve the model's performance, add new features, or enhance documentation are welcome! Feel free to fork this repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
