import numpy as np
from sklearn.utils import resample
import matplotlib.pyplot as plt

# Define the states
states = ['car', 'ebike']

# Define the transition matrix
transition_matrix = np.array([[0.89, 0.11], [0.018, 0.982]])

# Normalize the transition matrix
row_sums = transition_matrix.sum(axis=1)
normalized_transition_matrix = transition_matrix / row_sums[:, np.newaxis]

# Print the normalized transition matrix
print(normalized_transition_matrix)

# Simulate the Markov chain for many iterations
sumCar = 0
sumBike = 0
trials = 1000
num_iterations = 10
car_distributions = np.empty(num_iterations, dtype = float)
car_distributions.fill(num_iterations)
bike_distributions = np.empty(num_iterations, dtype = float)
bike_distributions.fill(num_iterations)
for j in range(trials):
    state_frequencies = {state: 0 for state in states}

    current_state = 'car'
    for i in range(num_iterations):
        state_frequencies[current_state] += 1
        car_distributions[i] += state_frequencies["car"]
        bike_distributions[i] += state_frequencies["ebike"]
        current_state = np.random.choice(states, p=transition_matrix[states.index(current_state)])

    # Estimate the steady-state probability distribution
    steady_state_distribution = np.array([state_frequencies[state] / num_iterations for state in states])
    sumCar += steady_state_distribution[0]
    sumBike += steady_state_distribution[1]
    
year = []
for i in range(num_iterations):
    car_distributions[i] = car_distributions[i]/((i+1)*trials+20)
    bike_distributions[i] = bike_distributions[i]/((i+1)*trials+20)
    year.append(i+1)

plt.plot(year, bike_distributions)
plt.plot(year, car_distributions)

pCar = sumCar/trials
pBike = sumBike/trials
print(f"Car percent: {pCar}")
print(f"Car reduced percent: {pBike}")

# 3.2 trillion miles/year
# 8887g CO2/gallon
# 22 miles gallon
distance = 3.2 * pBike
print(f"reduced by {distance} trillion miles")
gallons = distance/22 #in trillions
print(f"reduced by {gallons * 1000} billion gallons")
kgCO2 = gallons*8876/10000000 # still in trillions
print(f"reduced by {kgCO2 * 10000000} million tons of CO2")

# 1170/100million drivers
rides = distance/11.5
print(f"reduced by {rides * 1000} billion rides")
deaths = 1000000000000*(distance)/14263/100000 * 17.01
# deaths -= rides * 1000000000000 * 19/100000000
print(f"reduced by {deaths} deaths")

# 15 calories per mile
calories = distance * 15
print(f"burned {calories} trillion more calories")

plt.show()