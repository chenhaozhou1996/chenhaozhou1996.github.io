# Module 2: Linear Regression

## Overview
This module introduces linear regression, one of the fundamental algorithms in machine learning. Students will learn how to model relationships between variables, understand the mathematics behind ordinary least squares, and implement regression models from scratch and using scikit-learn.

## Learning Objectives
By the end of this module, students will be able to:
- Understand the mathematical foundations of linear regression
- Implement gradient descent from scratch
- Apply linear regression to real-world datasets
- Evaluate model performance using appropriate metrics
- Handle multiple features and polynomial regression

## Module Structure
1. **Theory**: Mathematical foundations and intuition
2. **Visualization**: Interactive demonstrations of regression concepts
3. **Code Lab**: Hands-on implementation
4. **Case Study**: Boston Housing Price Prediction

## Traditional Wisdom Connection
Linear regression mirrors the ancient principle of "finding the middle way" (中庸之道). Just as traditional wisdom seeks balance and optimal paths, linear regression finds the best-fitting line that minimizes overall error, achieving mathematical harmony in data relationships.

## Files in This Module
- `README.md` - This file
- `theory.html` - Interactive theory explanation
- `visualization.html` - Visual demonstrations
- `code/` - Implementation notebooks
  - `01_from_scratch.ipynb` - Build linear regression from scratch
  - `02_sklearn_implementation.ipynb` - Using scikit-learn
  - `03_advanced_techniques.ipynb` - Regularization and feature engineering
- `case_study/` - Practical application
  - `boston_housing.ipynb` - Complete case study
  - `data/` - Dataset files
- `exercises/` - Practice problems
- `solutions/` - Exercise solutions

## Prerequisites
- Basic calculus (derivatives)
- Matrix operations
- Python fundamentals
- NumPy basics

## Key Concepts
- Ordinary Least Squares (OLS)
- Gradient Descent
- Cost Function (MSE)
- Normal Equation
- R-squared and evaluation metrics
- Regularization (Ridge, Lasso)

## Quick Start
```python
# Simple linear regression example
from sklearn.linear_model import LinearRegression
import numpy as np

# Generate sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Train model
model = LinearRegression()
model.fit(X, y)

# Make predictions
predictions = model.predict([[6]])
print(f"Prediction for x=6: {predictions[0]}")
```

## Resources
- [Andrew Ng's Linear Regression Notes](https://cs229.stanford.edu/lectures-spring2022/main_notes.pdf)
- [ESL Chapter 3](https://hastie.su.domains/Papers/ESLII.pdf)
- [Visual Introduction to ML](http://www.r2d3.us/)
