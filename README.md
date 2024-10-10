# **MNIST Digit Classifier with Custom Neural Network**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Model Type: DistilBERT](https://img.shields.io/badge/Model-DistilBERT-green)](https://huggingface.co/distilbert-base-uncased)

## **Table of Contents**
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Model Performance](#model-performance)
- [License](#license)

---

## **Overview**

This project implements a digit classifier using a custom neural network built with NumPy and TensorFlow. The model is trained on the **MNIST** dataset, which consists of handwritten digits from 0 to 9.

## **Features**
- Custom neural network architecture with two hidden layers.
- Uses **sigmoid** and **ReLU** activation functions.
- Implements dropout for regularization to prevent overfitting.
- High accuracy on the MNIST test dataset.

---

## **Installation**

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/mnist-digit-classifier.git
cd mnist-digit-classifier
```
### **2. Install Dependencies**
```bash
pip install numpy tensorflow

```
### **3. Download Pretrained Models**
```bash
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt)

```



## **Usage**

### **1. Load and Preprocess the MNIST Dataset**
```bash
The script automatically loads the MNIST dataset and preprocesses the data."

```
## **2.Run the model**

### **To train the model and evaluate performance, run:**
```bash
python mnist_classifier.py

```

## **Model Training**


```bash
The training is conducted in the main script, with adjustable parameters:

Learning Rate: Set in the model function.
Iterations: Number of training iterations.
Dropout Probabilities: Control the dropout layers.
The model prints the cost every 100 iterations and evaluates accuracy on the test set at the end.




```

## **Model Performance**
```bash

Test Accuracy: The model achieves an accuracy of approximately 98% on the MNIST test dataset.
Metric	         Value
Test Accuracy	    98%

```



### **License**
```bash
This project is licensed under the MIT License. See the LICENSE file for more details.

```














