# Data Classification Using AI

## Project Overview
This project is part of the **DecodeLabs Industrial Training Kit (Project 2)**. It focuses on the **Supervised Learning** phase of AI development. The goal is to master the fundamental pipeline of teaching a machine to recognize patterns in data and categorize new information based on what it has learned.

## Key Features
- **Data Handling**: Loads and processes the classic **Iris flower dataset**.
- **Data Splitting**: Demonstrates the essential step of dividing data into training and testing sets.
- **Supervised Learning**: Implements the **K-Nearest Neighbors (KNN)** classification algorithm.
- **Model Validation**: Calculates accuracy and provides a detailed classification report to evaluate performance.

## Pipeline Steps
1. **Load Dataset**: Import the Iris dataset and convert it into a manageable format (Pandas DataFrame).
2. **Understand Data**: Identify features (sepal/petal length/width) and target classes (species).
3. **Split Data**: Separate the data to ensure the model is tested on unseen information.
4. **Train Model**: Use the training set to teach the KNN algorithm to recognize species patterns.
5. **Test & Validate**: Use the testing set to make predictions and calculate the model's accuracy.

## Requirements
- Python 3.x
- pandas
- scikit-learn

## How to Run
First, ensure you have the required libraries installed:
```bash
pip install pandas scikit-learn
```
Then, run the classification script:
```bash
python3 classification_model.py
```

## Developed by
- **Hadia Akbar** (Project Task 2)
- Powered by **DecodeLabs**
