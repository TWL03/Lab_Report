from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

##############################################################################
# Given nodes
nodes = np.array([
    [0, 0.00351],
    [1.0, 0.00354],
    [1.9, 0.00352],
    [3.0, 0.00354],
    [3.9, 0.00354],
    [4.9, 0.00359],
    [5.9, 0.00363],
    [7.0, 0.00366],
    [8.0, 0.00370],
    [8.9, 0.00376],
    [10.0, 0.00399],
    [11.0, 0.00514],
    [11.9, 0.04068],
    [12.8, 0.10813],
    [13.4, 0.30507],
    [12.5, 1.21830],
    [11.4, 2.00749]
]  
)

x = nodes[:,0]
y = nodes[:,1]

############################################################################
# Given nodes for region of interest
nodes2 = np.array([
    [11.0, 0.00543],
    [11.5, 0.01087],
    [11.7, 0.03885],
    [12.5, 0.09480],
    [12.7, 0.10585],
    [12.9, 0.15520],
    [13.4, 0.27357]
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
############################################################################
#first linear regression line
X1 = nodes[:10,0].reshape(-1, 1)  # Reshape for sklearn
Y1 = nodes[:10, 1]

# Create and fit the model
model = LinearRegression()
model.fit(X1, Y1)

# Extending the X1 range for visualization
X1_extended = np.linspace(0, 18, 100).reshape(-1, 1)
Y1_pred_extended = model.predict(X1_extended)
#############################################################################
#second linear regression line
X2 = nodes[-2:, 0].reshape(-1, 1)  # Correctly extract X values for the last three points
Y2 = nodes[-2:, 1]  # Correctly extract Y values for the last three points

# Re-create and fit the model using the last three points
model2 = LinearRegression()
model2.fit(X2, Y2)

# Extending the X2 range for visualization over the same range as defined earlier
X2_extended = np.linspace(11, 15, 100).reshape(-1, 1)
Y2_pred_extended = model2.predict(X2_extended)
#############################################################################
#find the intersection point of two line
# Assuming linear regression models have to be fitted to these datasets
model11 = LinearRegression().fit(X1, Y1)
model22 = LinearRegression().fit(X2, Y2)

# Extract the coefficients (slope) and intercepts of both linear regression models
slope1, intercept1 = model11.coef_[0], model11.intercept_
slope2, intercept2 = model22.coef_[0], model22.intercept_

# Calculate the x-coordinate of the intersection point
x_inter = (intercept2 - intercept1) / (slope1 - slope2)

# Calculate the y-coordinate of the intersection point
y_inter = slope1 * x_inter + intercept1

# Intersection point
print("Intersecion point:{",f"{x_inter:.2f}, {y_inter:.5f}"," }")
############################################################################
# Example error values (replace with your actual error data) for first data
x_errors = np.array([0.1] * len(x))  # Standard error for x-values
y_errors = np.array([0.0001] * len(y))  # Constant error for illustration
# Extracting x and y values
############################################################################
# Example error values (replace with your actual error data) for region of interest
x_errors2 = np.array([0.1] * len(x2))  # Standard error for x-values
y_errors2 = np.array([0.0001] * len(y2))  # Constant error for illustration
# Extracting x2 and y2 values
############################################################################
# Calculate the slope of the line between the last two data points
dx = nodes[-1, 0] - nodes[-2, 0]
dy = nodes[-1, 1] - nodes[-2, 1]
slope = dy / dx

# Extend the line for one more unit
x_extension = nodes[-1, 0] + dx
y_extension = nodes[-1, 1] + dy

# Add this point to the nodes
nodes_extended = np.vstack([nodes, [x_extension, y_extension]])

# Extracting x and y values with the extended node
x_extended = nodes_extended[:, 0]
y_extended = nodes_extended[:, 1]

# Performing spline interpolation with the extended nodes
tck_extended, u_extended = interpolate.splprep([x_extended, y_extended], s=0)
xnew_extended, ynew_extended = interpolate.splev(np.linspace(0, 1, 100), tck_extended, der=0)

###############################################################################
# Finding the maximum x-value from the xnew_extended array
max_x_value = np.max(xnew_extended)

print("Maximum x-value based on the graph:", max_x_value)

################################################################################

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(x_extended, y_extended, 'x', xnew_extended, ynew_extended, '-', color = "blue")
# Add error bars for both x and y
plt.errorbar(x, y, xerr=x_errors, yerr=y_errors, fmt='x', color="black", ecolor='red', elinewidth=2, capsize=4) 
# Add error bars for both x2 and y2 
plt.errorbar(x2,y2, xerr=x_errors2, yerr=y_errors2, fmt='x',color='green',ecolor='orange',elinewidth=2, capsize=4)
plt.plot(x_fit, y_fit, color='green',label='Fitted Curve')
plt.legend(['Extended Data', 'Extended Spline' ,'Fitted curve','Original Data with error bar','Data of region of interest with error bar'])
plt.plot(X1_extended, Y1_pred_extended, '--', color='red')  # Plotting the regression line 1
plt.plot(X2_extended, Y2_pred_extended, '--', color='red')  # Plotting the regression line 2
plt.title('Graph of anode current against accelerating voltage')
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
plt.savefig('finalPARTA.png')

plt.show()

