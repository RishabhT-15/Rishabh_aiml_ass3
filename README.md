# Executive Salary Estimation Using Polynomial Regression

## Objective
The primary aim of this project is to construct a **Polynomial Regression (Degree = 3)** model to predict an employee's salary based on their position level. By capturing the non-linear relationship inherent in corporate compensation structures, this model provides a far more accurate salary estimate than standard linear baselines.

## Dataset Link
- [Position Salaries Dataset on Kaggle](https://www.kaggle.com/datasets/akram24/position-salaries)

## Libraries Used
- **Pandas**: Used for structured data loading, dataset inspection, and feature slicing.
- **NumPy**: Applied for mathematical arrays, grid generation, and numerical transformations.
- **Scikit-learn**: Employed for feature polynomial transformation (`PolynomialFeatures`), data splitting (`train_test_split`), model fitting (`LinearRegression`), and performance evaluation metrics.
- **Matplotlib**: Utilized for generating visual scatter plots and continuous regression curve plots.

## Methodology
1. **Data Exploration & Assessment**: Loaded the 10-row dataset to examine structural details, variable types (`Level` as $X$, `Salary` as $y$), and initial distribution summary stats.
2. **Data Preprocessing & Splitting**: Verified zero missing values across all entries. Configured the feature matrix into a 2D format and split the data into an 80% training set and a 20% test set using a set seed (`random_state=42`) for reproducibility.
3. **Polynomial Feature Transformation & Modeling**: Enhanced the single feature space by applying a degree-3 polynomial transformation. Trained an ordinary least squares regression model on the expanded feature matrix and generated predictions for the test split.
4. **Performance Evaluation & Visualization**: Calculated regression metrics (MAE, MSE, $R^2$) on test predictions. Rendered a continuous non-linear curve over the raw data points to visually analyze model fit.

## Results
*Metric scores derived from model run:*
- **Mean Absolute Error (MAE)**: Measures the average magnitude of absolute prediction errors.
- **Mean Squared Error (MSE)**: Quantifies the average squared difference between true and predicted salaries.
- **$R^2$ Score**: Indicates the proportion of salary variance explained by position levels.

### Key Observations
1. **Curved Wage Acceleration**: Salaries follow a distinct non-linear path, expanding exponentially at senior executive tiers (levels 8 to 10) rather than advancing uniformly.
2. **Superior Alignment**: Converting position levels into degree-3 polynomial space enables the line to follow the steep curve without underfitting mid-range titles.
3. **Linear Model Inadequacy**: A standard 1st-degree linear fit would dramatically underestimate executive compensation while overestimating entry-to-mid level roles.

## Conclusion
This assignment demonstrated how Polynomial Regression addresses complex, non-linear relationships that traditional Simple Linear Regression fails to resolve. The key insight is that corporate pay scale increments do not follow fixed linear increments ($y = mx + c$); instead, they scale polynomial-wise ($y = \beta_0 + \beta_1 x + \beta_2 x^2 + ... + \beta_n x^n$).

The core difference lies in model flexibility: Linear Regression strictly assumes a constant rate of change, whereas Polynomial Regression bends to accommodate accelerating trajectories.

The main advantage of Polynomial Regression for this scenario is its capacity to smoothly track exponential growth trends without distorting mid-level predictions or severely underestimating C-suite compensation.
