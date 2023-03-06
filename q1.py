import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate some example data
x_data = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
y_data = np.array([13.63119073, 24.06322445, 38.80716544, 58.6975764, 81.78714436, 99.59114858, 118.7860906, 126.1580611, 158.4278188, 189.7239199, 227.6965227, 288.4805058, 369, 423, 416, 750, 928])

# Transform the data using logarithms
x_log = np.log(x_data.reshape(-1, 1))
y_log = np.log(y_data)


# Fit a linear regression model to the transformed data
model = LinearRegression().fit(x_log, y_log)

# Calculate p-values for the fitted parameters
x_log = sm.add_constant(x_log) # add constant to the transformed data
model_sm = sm.OLS(y_log, x_log).fit() # fit OLS model using the transformed data
p_values = model_sm.pvalues[1:] # exclude the constant from the p-values

# Print the fitted parameters, R-squared value, and p-values
a = np.exp(model.intercept_)
b = model.coef_[0]
r_squared = model.score(x_log[:, 1:], y_log)
print("Fitted parameters: a = %g, b = %g" % (a, b))
print("R-squared value: %g" % r_squared)
# print("P-values: a = %g, b = %g" % tuple(p_values))
print(p_values)

# Plot the original data and the fitted curve
plt.plot(x_data, y_data, 'bo', label='Original Data')
plt.plot(x_data, a * np.exp(b * x_log[:, 1:]), 'r-', label='Fitted Curve')
# a + e^(blog(x))
print(np.exp(b * x_log[:, 1:]))
plt.legend(loc='best')
print(a * np.exp(b * np.log(25)))
print(a * np.exp(b * np.log(28)))
plt.show()
