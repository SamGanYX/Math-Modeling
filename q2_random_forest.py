import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset into a pandas DataFrame
df = pd.read_csv('q2_standardized.csv')

# Split the dataset into input features (X) and target variable (y)
X = df.drop('Sales', axis=1)
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_regressor = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)

rf_regressor.fit(X_train, y_train)

y_pred = rf_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Calculate feature importances using permutation feature importance
result = permutation_importance(rf_regressor, X, y, n_repeats=10, random_state=0)

# Print the feature importances
for i in range(X.shape[1]):
    print(f"{X.columns[i]:<30}: {result.importances_mean[i]:.3f}")

plt.scatter(y_test, y_pred, color='blue')
plt.plot(y_test, y_test, color='red', linewidth=2)

plt.title('Random Forest Regression')
plt.xlabel('True Values')
plt.ylabel('Predictions')

plt.show()
