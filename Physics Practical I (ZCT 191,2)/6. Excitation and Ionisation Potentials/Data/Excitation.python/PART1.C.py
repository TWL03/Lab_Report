from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.optimize import curve_fit

######################################################
# Given nodes
nodes = np.array( [
    [0, 0.46381],
    [0.9, 1.17917],
    [2.0, 1.29994],
    [3.0, 1.05947],
    [3.9, 0.83390],
    [4.9, 0.68004],
    [6.0, 0.59198],
    [7.0, 0.54673],
    [8.0, 0.53704],
    [9.0, 0.55442],
    [10.0, 0.59577],
    [11.0, 0.64588]
]
 )

# Extracting x and y values
x = nodes[:,0]
y = nodes[:,1]
######################################################
# Given nodes for region of interest
nodes2 = np.array([
    [6.0, 0.62406],
    [6.4, 0.55486],
    [7.0, 0.55398],
    [7.4, 0.55258],
    [8.0, 0.52231],
    [8.4, 0.52990],
    [9.0, 0.53305]
]
)

x2 = nodes2[:,0]
y2 = nodes2[:,1]

# Defining a function for curve fitting, using a polynomial of degree 3 as an example
def poly_func(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

# Curve fitting
popt, pcov = curve_fit(poly_func, x2, y2)

# Plotting the fitted curve
x_fit = np.linspace(min(x2), max(x2), 500)
y_fit = poly_func(x_fit, *popt)
#################################################################
# Performing spline interpolation
tck, u = interpolate.splprep( [x,y], s = 0 )
xnew, ynew = interpolate.splev( np.linspace( 0, 1, 100 ), tck, der = 0)

#######################################################################
# Calculating the first derivative of the spline over the interpolated range
derivative = interpolate.splev(np.linspace(0, 1, 100), tck, der=1)

# The derivative is a list of arrays, one for each dimension. Since we're working in 1D, we only need the first array.
dx, dy = derivative

# Find zeros of the derivative - places where dy/dx changes sign might indicate local minima or maxima
sign_changes = np.diff(np.sign(dy))  # Find where the derivative changes sign

# Local minima are where the derivative changes from negative to positive, i.e., sign change is positive
minima_indices = np.where(sign_changes > 0)[0] + 1  # +1 because np.diff shifts indices by 1

# Extracting the local minima points
minima_x = xnew[minima_indices]
minima_y = ynew[minima_indices]

if len(minima_x) > 0:
    # Assuming interest in the first local minimum
    first_min_x = minima_x[0]
    first_min_y = minima_y[0]
    print(f"First local minimum point at x = {first_min_x:.2f}, y = {first_min_y:.6f}")
else:
    print("No local minimum found.")
#####################################################################
# Example error values (replace with your actual error data)
x_errors = np.array([0.1] * len(x))  # Standard error for x-values
y_errors = np.array([0.0001] * len(y))  # Constant error for illustration
# Extracting x and y values
############################################################################
# Example error values (replace with your actual error data) for region of interest
x_errors2 = np.array([0.1] * len(x2))  # Standard error for x-values
y_errors2 = np.array([0.0001] * len(y2))  # Constant error for illustration
# Extracting x2 and y2 values
############################################################################
# Plotting
plt.figure(figsize=(8, 6))
plt.plot(xnew ,ynew , '-', color = "blue")
plt.errorbar(x, y, xerr=x_errors, yerr=y_errors, fmt='x', color="black", ecolor='red', elinewidth=2, capsize=6, label='Original Data with Error Bars')  # Add error bars for both x and y
# Add error bars for both x2 and y2 
plt.errorbar(x2,y2, xerr=x_errors2, yerr=y_errors2, fmt='x',color='green',ecolor='orange',elinewidth=2, capsize=4)
plt.plot(x_fit, y_fit, color='green',label='Fitted Curve')
plt.legend(['Spline','Fitted curve','Original Data with error bar','Data of region of interest with error bar'])
plt.title('Graph of current against accelerating voltage')
plt.xlabel('accelerating voltage, V')
plt.ylabel('Anode Current, $\mu$A')
plt.grid(True)
plt.axis([x.min() - 1, x.max() + 1, y.min() - 0.1, y.max() + 1])

# Setting x-axis scale with interval of 1
plt.xticks(np.arange(x.min() , x.max() + 1, 1))
# Setting y-axis scale with interval of 0.1
plt.yticks(np.arange(0 , y.max() + 1, 0.1))

# Adding minor grid with specified interval
ax = plt.gca()  # Get current axis
ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.02))

# Enable minor grid with custom styling
ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')

# Save the plot to a new file
plt.savefig('finalPARTC.png')

plt.show()

