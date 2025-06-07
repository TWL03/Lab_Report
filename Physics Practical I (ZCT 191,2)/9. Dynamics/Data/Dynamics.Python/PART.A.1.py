import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress

# Given data
distances = np.array([10, 12, 15, 17, 20, 23, 25, 27, 30, 32, 35, 37, 40])  # in cm
sqrt_distances = np.sqrt(distances)  # √d

# Time measurements from the table
time_data = {
    "t1": [0.134, 0.149, 0.169, 0.181, 0.195, 0.214, 0.220, 0.230, 0.241, 0.249, 0.264, 0.271, 0.284],
    "t2": [0.133, 0.153, 0.170, 0.181, 0.195, 0.215, 0.225, 0.230, 0.243, 0.247, 0.265, 0.271, 0.284],
    "t3": [0.131, 0.150, 0.170, 0.181, 0.195, 0.215, 0.223, 0.230, 0.243, 0.249, 0.265, 0.272, 0.283],
    "t4": [0.130, 0.148, 0.169, 0.183, 0.196, 0.214, 0.220, 0.231, 0.245, 0.249, 0.264, 0.271, 0.284],
    "t5": [0.132, 0.149, 0.173, 0.182, 0.194, 0.213, 0.221, 0.233, 0.244, 0.249, 0.264, 0.273, 0.284],
    "t6": [0.137, 0.149, 0.175, 0.181, 0.195, 0.214, 0.220, 0.232, 0.241, 0.249, 0.266, 0.272, 0.284],
    "t7": [0.130, 0.149, 0.170, 0.181, 0.195, 0.214, 0.214, 0.233, 0.249, 0.250, 0.266, 0.272, 0.284],
    "t8": [0.131, 0.149, 0.169, 0.181, 0.195, 0.215, 0.215, 0.233, 0.242, 0.249, 0.265, 0.272, 0.284],
    "t9": [0.130, 0.149, 0.172, 0.182, 0.195, 0.313, 0.313, 0.232, 0.235, 0.249, 0.265, 0.274, 0.284],
    "t10": [0.133, 0.151, 0.174, 0.180, 0.196, 0.214, 0.214, 0.232, 0.235, 0.242, 0.265, 0.272, 0.284]
}


# Calculate the average time for each distance
average_times = np.mean([time_data[key] for key in time_data], axis=0)
# Calculate the standard deviation of the times
time_std_devs = np.std([time_data[key] for key in time_data], axis=0, ddof=1)

# Perform linear regression using scipy.stats.linregress to get standard error of the slope
regression_result = linregress(sqrt_distances, average_times)
slope = regression_result.slope
intercept = regression_result.intercept
slope_std_err = regression_result.stderr

# Determine g and its error from the slope and standard error of the slope
g = 2 / slope**2
g_error = (4 * slope_std_err / slope**3) * 2

# Plotting the graph with error bars
plt.figure(figsize=(10, 6))
plt.errorbar(sqrt_distances, average_times, yerr=time_std_devs, fmt='o', label='Experimental Data', capsize=5)
plt.plot(sqrt_distances, intercept + slope * sqrt_distances, '-', label=f'Linear Fit: $\\bar{{t}} = {slope:.2f}\\sqrt{{d}} + {intercept:.2f}$')

plt.xlabel('√d (cm^0.5)')
plt.ylabel('Mean Time (s)')
plt.title('Graph of Mean Time vs. √d')
plt.legend()
plt.grid(True)
plt.show()

# Print the calculated values of g, Δt, and the error in g
print(f"Calculated value of g: {g:.2f} cm/s^2")
print(f"Calculated error in g: {g_error:.2f} cm/s^2")
print(f"Calculated value of Δt: {intercept:.2f} s")
