import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import t
from numpy import sqrt

# Define the data points
R_1 = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])  # Resistance in ohms
I_mA = np.array([3.80, 2.40, 1.80, 1.40, 1.10, 1.00, 0.80, 0.70, 0.60, 0.50])  # Current in milliamps
I_errors = np.array([0.1] * len(I_mA))  # Error in current measurements

# Define the exponential function to fit
def exponential(R, A, b):
    return A * np.exp(-b * R)

# Perform curve fitting with error consideration
params, params_covariance = curve_fit(exponential, R_1, I_mA, sigma=I_errors, absolute_sigma=True, p0=[10, 0.001])

# Extract the parameters
A, b = params
sigma_A, sigma_b = np.sqrt(np.diag(params_covariance))

# Generate a smoother range of Resistance values
R_fine = np.linspace(0, 1000, 400)  # More points for a smoother curve
I_fit = exponential(R_fine, A, b)

# Update b for the fixed A curve
b_fixed = 0.004875
I_fixed = exponential(R_fine, 10, b_fixed)  # Fixed A at 10, updated b

# Target current for calculation
target_I = 1.2
R_target = np.log(target_I / A) / -b

# Calculate uncertainty in R_target using error propagation
dR_target_dA = -1 / (A * b) * (target_I / A)
dR_target_db = -np.log(target_I / A) / b**2
sigma_R_target = sqrt((dR_target_dA * sigma_A)**2 + (dR_target_db * sigma_b)**2)
t_value = t.ppf(0.975, len(R_1) - 2)  # T-value for 95% CI
CI_R_target = t_value * sigma_R_target

# Plot the results
plt.figure(figsize=(8, 5))
plt.errorbar(R_1, I_mA, yerr=I_errors, fmt='x', color='red', elinewidth=2, capsize=4, label='Data Points with Error Bars')
plt.plot(R_fine, I_fit, label=f'Fitted Curve: $I = {A:.2f} e^{{-{b:.4f}R}}$', color='blue')
plt.plot(R_fine, I_fixed, label=f'Additional Curve: $I = 10 e^{{-{b_fixed:.4f}R}}$', color='green', linestyle='--')
plt.errorbar([R_target], [target_I], xerr=CI_R_target, fmt='x', color='black', elinewidth=2, capsize=4,
             label=f'Target Point: $I=1.2$ mA at $R={R_target:.2f} \pm {CI_R_target:.2f}$ Ohms')
plt.title('Exponential Fit to Current against Resistance with Uncertainty')
plt.xlabel('Resistance (Ohms)')
plt.ylabel('Current (mA)')
plt.legend()
plt.grid(True)
plt.show()



