import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# Load the data from a CSV file into a pandas DataFrame
df = pd.read_csv('q2_standardized.csv')

# Extract the input features and target variable from the DataFrame
X = df[['Urban pop', 'Gas Prices', 'Battery_Value', "Disposable income"]].values
y = df['Sales'].values

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# Create a Ridge regression object
ridge_regressor = Ridge(alpha=1.0)

# Fit the model to the training data
ridge_regressor.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = ridge_regressor.predict(X_test)

# Evaluate the model using mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Calculate R-squared and p-value
X_train_ols = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train_ols)
results = model.fit()
print("R-squared:", results.rsquared)
print("p-values:", results.pvalues)

# Plot the predicted values and the true values
plt.scatter(y_test, y_pred, color='blue')
plt.plot(y_test, y_test, color='red', linewidth=2)

# Set the plot title and axis labels
plt.title('Ridge Regression')
plt.xlabel('True Values')
plt.ylabel('Predictions')

# Print the coefficients and intercept
print('Coefficients: ', ridge_regressor.coef_)
print('Intercept: ', ridge_regressor.intercept_)

# Show the plot
plt.show()
