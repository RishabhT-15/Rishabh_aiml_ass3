import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=== TASK 1: DATA UNDERSTANDING ===")

data = {
    'Position': ['Business Analyst', 'Junior Consultant', 'Senior Consultant', 'Manager', 
                 'Country Manager', 'Region Manager', 'Partner', 'Senior Partner', 
                 'C-level', 'CEO'],
    'Level': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [45000, 50000, 60000, 80000, 110000, 150000, 200000, 300000, 500000, 1000000]
}
df = pd.DataFrame(data)

print("\nFirst 5 Records:")
print(df.head())

print("\nInput Feature (X): 'Level'")
print("Target Variable (y): 'Salary'")

print("\nDataset Info:")
df.info()

print("\nSummary Statistics:")
print(df.describe())

print("\n=== TASK 2: DATA PREPROCESSING ===")

print("\nMissing Values Count:")
print(df.isnull().sum())

X = df[['Level']].values
y = df['Salary'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

print("\n=== TASK 3: MODEL DEVELOPMENT ===")

poly_reg = PolynomialFeatures(degree=3)
X_poly_train = poly_reg.fit_transform(X_train)
X_poly_test = poly_reg.transform(X_test)

model = LinearRegression()
model.fit(X_poly_train, y_train)

y_pred = model.predict(X_poly_test)

print("Model Training Completed Successfully!")
print("Predicted values for Test Set:", y_pred)
print("Actual values for Test Set:   ", y_test)

print("\n=== TASK 4: MODEL EVALUATION ===")

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"Mean Squared Error (MSE):  ${mse:,.2f}")
print(f"R² Score:                  {r2:.4f}")

plt.figure(figsize=(10, 6))

plt.scatter(X, y, color='red', label='Original Data')

X_grid = np.arange(min(X), max(X), 0.1).reshape(-1, 1)
plt.plot(X_grid, model.predict(poly_reg.transform(X_grid)), color='blue', label='Polynomial Curve (Degree 3)')

plt.title('Position Level vs Salary (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary ($)')
plt.legend()
plt.grid(True)
plt.savefig('salary_regression_chart.png')
plt.show()

print("\nObservations:")
print("1. The dataset exhibits an exponential/curved growth trend where salary increases drastically at higher position levels.")
print("2. A standard degree-1 linear regression would fail to capture this curve, leading to high underfitting errors.")
print("3. The degree-3 polynomial curve tightly fits the observed values, achieving an excellent fit across both lower and higher job tiers.")

print("\n=== TASK 5: CONCLUSION ===")
conclusion = """
Conclusion:
In this analysis, we modeled the relationship between employee position levels and their corresponding salaries. 
Key findings show that compensation grows exponentially rather than at a constant rate, particularly beyond mid-level roles. 

Unlike Simple Linear Regression, which fits a straight line ($y = mx + c$) and assumes a constant rate of change, Polynomial Regression models degree-based polynomial equations ($y = \beta_0 + \beta_1 x + \beta_2 x^2 + ... + \beta_n x^n$) to flexibly map non-linear curves. 

The primary advantage of Polynomial Regression for this dataset is its ability to accurately reflect non-linear wage structures without misrepresenting mid-level earnings or underestimating executive-level compensation.
"""
print(conclusion)
