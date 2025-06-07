import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np

# Given data for 1/m (1/kg) and v^2 (m^2/s^2)
# Masses in grams
masses_g = [190, 200, 210, 220, 230, 240, 250, 260]
# Convert masses to kilograms
masses_kg = np.array(masses_g) / 1000
# Calculate 1/m in (1/kg)
inverse_masses = 1 / masses_kg

# Velocities in m/s
velocities_m1_s = [0.788436268, 0.755667506, 0.718562874, 0.668896321, 0.643776824, 0.617283951, 0.614124872, 0.598802395]
# Convert velocities to m/s
velocities_m_s = np.array(velocities_m1_s)
# Calculate v^2 in (m^2/s^2)
v_squared = velocities_m_s ** 2

# Perform linear regression on v^2 vs 1/m
slope, intercept, r_value, p_value, std_err = linregress(inverse_masses, v_squared)

# Plotting the graph
plt.figure(figsize=(10, 6))
plt.scatter(inverse_masses, v_squared, label='Experimental Data')
plt.plot(inverse_masses, intercept + slope * np.array(inverse_masses), 'r', label=f'Linear Fit: $v^2 = {slope:.4f}(1/m) + {intercept:.4f}$')

plt.xlabel('1/m ($kg^{-1}$)')
plt.ylabel('$v^2$ ($m^2$$s^{-2}$)')
plt.title('Graph of $v^2$ vs. 1/m')
plt.legend()
plt.grid(True)
plt.show()

# Number of rubber bands
n = 2  # Replace with the actual number of rubber bands if different

# Calculate the potential energy of each rubber band
epsilon = slope / (2 * n)
std_err_epsilon = std_err / (2 * n)

print(f"Slope (m): {slope:.4f}")
print(f"Standard Error of Slope (σm): {std_err:.4f}")
print(f"Calculated potential energy of each rubber band (ε): {epsilon:.4f} J")
print(f"Standard Error of potential energy of each rubber band (σε): {std_err_epsilon:.4f} J")
print(f"R-correlation: {r_value:.2f}")



