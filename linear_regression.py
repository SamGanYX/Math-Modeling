import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error


data = pd.read_csv("Atlantaforsold.csv")
data = data[["BEDS", "BATHS", "SQUARE FEET", "LOT SIZE", "PRICE"]]
for i in data.columns:
    data = data.dropna(subset=[i])
data = data.sample(frac=1)
num = int(len(data) * 0.8)


train = data[:num]
test = data[num:]

df = train



X = df[["BEDS"]] # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
Y = df['PRICE']

regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)
# print(regr.coef_[0])

plt.subplot(2,2,1)
plt.scatter(df[["BEDS"]], df[["PRICE"]])
plt.plot(df[["BEDS"]], df[["BEDS"]]*regr.coef_[0] + regr.intercept_, color="green")
plt.xlabel("BEDS")
plt.ylabel("PRICE")

plt.subplot(2,2,2)
plt.scatter(df[["BATHS"]], df[["PRICE"]])
plt.plot(df[["BATHS"]], df[["BATHS"]]*regr.coef_[1] + regr.intercept_, color="green")
plt.xlabel("BATHS")
plt.ylabel("PRICE")

plt.subplot(2,2,(3,4))
plt.scatter(df[["SQUARE FEET"]], df[["PRICE"]])
plt.plot(df[["SQUARE FEET"]], df[["SQUARE FEET"]]*regr.coef_[2] + regr.intercept_, color="green")
plt.xlabel("SQUARE FEET")
plt.ylabel("PRICE")

print(test[["PRICE"]])
print((test[["BEDS", "BATHS", "SQUARE FEET"]]*regr.coef_) + regr.intercept_)
plt.show()