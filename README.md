# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. .Load the placement.csv dataset and select academic features (ssc_p, hsc_p, degree_p, etest_p, mba_p) and the target status.

2. Encode the target variable as Placed = 1 and Not Placed = 0, and standardize the input features.

3. .Initialize the weights and bias, and define the Sigmoid function to calculate placement probability.

4. .Calculate the prediction and gradient, then update the weights using Gradient Descent.

5. .Repeat the gradient calculation and weight update for the specified number of iterations.

6. .Predict the placement status using a 0.5 threshold and calculate the model accuracy.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: KIREN SIDDHARTH D
RegisterNumber: 212224250009

import pandas as pd
import numpy as np

# Load dataset
data = pd.read_csv(r"C:\Users\k\Downloads\Placement_Data.csv")

print("Dataset:")

print(data.iloc[:5, :8].to_string(index=False))


print(data.iloc[:5, [0, 8, 9, 10, 11, 12, 13, 14]].to_string(index=False))
# Select input features
X = data[["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]].values

# Convert target into binary values
y = data["status"].map({"Placed": 1, "Not Placed": 0}).values

# Standardize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Add bias term
X = np.c_[np.ones(X.shape[0]), X]

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Initialize weights
weights = np.zeros(X.shape[1])

# Parameters
learning_rate = 0.01
iterations = 1000

# Gradient Descent
for i in range(iterations):
    z = np.dot(X, weights)
    y_pred = sigmoid(z)

    gradient = np.dot(X.T, (y_pred - y)) / len(y)

    weights = weights - learning_rate * gradient

# Predictions
probabilities = sigmoid(np.dot(X, weights))
predictions = (probabilities >= 0.5).astype(int)

# Accuracy
accuracy = np.mean(predictions == y)

# Display results
print("\nFinal Weights:")
print(weights)

print("\nAccuracy:", accuracy * 100, "%")



*/
```

## Output:


<img width="603" height="289" alt="image" src="https://github.com/user-attachments/assets/23afa6ad-61d6-4b0b-9f34-b8cee60cc57c" />




## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

