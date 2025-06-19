# CO₂ Emission Classification using a Custom Neural Network

This project implements a neural network from scratch using NumPy to classify vehicles based on their CO₂ emissions, using a real-world dataset from Canada. The model categorizes each vehicle into one of three classes: **Low**, **Medium**, or **High** emission.

## Motivation

This project was developed during my 4th semester as an Computer Science student in Germany. My goal was to deepen my understanding of machine learning by building a fully custom neural network from the ground up, without using any high-level ML libraries. It served as a practical introduction to neural networks, training dynamics, and model evaluation.

## Features

- Fully custom neural network (no deep learning libraries)
- Class weighting to handle class imbalance
- L2 regularization to reduce overfitting
- Early stopping to prevent unnecessary training
- Visualization of validation loss and confusion matrix
- Basic unit tests with `pytest`

## Project Structure
```
.
├── best_model.npz
├── confusion_matrix.png
├── data
│   ├── CO2 Emissions_Canada.csv
│   └── Data Description.csv
├── README.md
├── src
│   ├── __init__.py
│   ├── model.py
│   ├── predict.py
│   ├── preprocess.py
│   └── train.py
├── tests
│   ├── __init__.py
│   └── test_model.py
└── validation_loss_vs_epochs.png
```

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/co2_vehicles.git
cd co2_vehicles
```

### 2. Train the model
Run the training script to start training and it will save the best model:
```bash
python -m src.train
```

### 3. Run tests
```bash
pytest
```

## Model Performance

Final model evaluation on the test set:

| Class   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Low     | 0.97      | 0.97   | 0.97     | 365     |
| Medium  | 0.96      | 0.97   | 0.97     | 732     |
| High    | 0.98      | 0.96   | 0.97     | 380     |

**Overall Accuracy:** 96.75%  
**Macro Average F1 Score:** 0.97  
**Weighted Average F1 Score:** 0.97

## Dataset Source

The dataset used in this project is the **CO₂ Emission by Vehicles** dataset published on Kaggle.

- Dataset title: *CO2 Emission by Vehicles*
- Source: Kaggle, uploaded by Debajyoti Podder  
- Link: [https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles](https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles)

This dataset contains specifications and CO₂ emission values for various vehicles and is intended for research and educational use.
