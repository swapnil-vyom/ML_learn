Scikit-Learn LinearRegression Explained
Scikit-Learn's LinearRegression is a simple and efficient way to perform linear regression in Python. It fits a linear model to the data by minimizing the sum of squared residuals.

1. Importing the Required Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


2. Creating Sample Data
We generate a simple dataset where X is the independent variable and Y is the dependent variable.

# Generating sample data
np.random.seed(42)  # For reproducibility
X = np.random.rand(100, 1) * 10  # 100 data points between 0 and 10
y = 2.5 * X + 5 + np.random.randn(100, 1) * 2  # y = 2.5X + 5 + noise



3. Splitting the Data
We split the dataset into training and testing sets (80% train, 20% test).

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


4. Training the Model
Now, we create an instance of LinearRegression and fit it to the training data.

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)


5. Model Parameters
Once trained, we can check the learned parameters:

print(f"Intercept (β0): {model.intercept_[0]:.2f}")
print(f"Coefficient (β1): {model.coef_[0][0]:.2f}")
This will print:

Intercept (β0): 4.93
Coefficient (β1): 2.56
Which means our model has learned:

𝑌=4.93+2.56𝑋
Y=4.93+2.56X


6. Making Predictions
Now, let's predict on the test data:

y_pred = model.predict(X_test)


7. Evaluating the Model
We use Mean Squared Error (MSE) and R² Score to evaluate how well the model fits the data.
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")
MSE: Measures how far predictions are from actual values (lower is better).
R² Score: Measures how well the model explains the variance in data (closer to 1 is better).


8. Visualizing the Results
We can plot the original data points and the regression line.

plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression with Scikit-Learn')
plt.legend()
plt.show()


9. Making Predictions on New Data
Once trained, you can use the model to predict new values.
new_X = np.array([[3], [7], [12]])  # New input values
new_y_pred = model.predict(new_X)

print(f"Predicted values: {new_y_pred.flatten()}")
Summary of Scikit-Learn's LinearRegression
fit(X, y) → Trains the model on data.
predict(X) → Predicts outputs for new data.
coef_ & intercept_ → Retrieves learned parameters.
score(X, y) → Computes R² score.
mean_squared_error(y_true, y_pred) → Evaluates model performance.
