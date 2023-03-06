import pandas as pd
data = pd.DataFrame({'years': [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                     'sales': [13.63119073, 24.06322445, 38.80716544, 58.6975764, 81.78714436, 99.59114858, 118.7860906, 126.1580611, 158.4278188, 189.7239199, 227.6965227, 288.4805058, 369, 423, 416]})
data['cum_sum'] = data['sales'].cumsum()
print(data)
from scipy.optimize import curve_fit
def c_t(x, p, q, m):
    return (p+(q/m)*(x))*(m-x)
popt, pcov = curve_fit(c_t, data.cum_sum[0:11], data.sales[1:12])
print(popt)

import matplotlib.pyplot as plt
import numpy as np

# Define the Bass model parameters
p = popt[0]
q = popt[1]
m = popt[2]
n0 = 5

# Define the time horizon
t = np.arange(1, 11)

# Calculate the cumulative adoption curve
n = m * ((1 - np.exp(-p*t)) + (np.exp(-p*t)*(1-np.exp(-q*t))))

# Plot the cumulative adoption curve
plt.plot(t, n)
plt.xlabel('Time')
plt.ylabel('Cumulative adoption')
plt.title('Diffusion of a Bass model')

# Calculate the annual adoption curve
annual_n = np.diff(n)
t_annual = np.arange(2, 11)

# Plot the annual adoption curve
plt.figure()
plt.bar(t_annual, annual_n)
plt.xlabel('Time')
plt.ylabel('Annual adoption')
plt.title('Diffusion of a Bass model')
plt.show()