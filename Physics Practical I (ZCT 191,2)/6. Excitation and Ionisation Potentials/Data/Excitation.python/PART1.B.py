from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

################################################################
# Given nodes
nodes = np.array( [
    [0, 0],
    [1.6, 20],
    [3.4, 40],
    [5.1, 60],
    [6.9, 80],
    [8.5, 110],
    [10.3, 150],
    [10.9, 180],
    [11.5, 210],
    [12.1, 250]
]
 )

# Extracting x and y values
x = nodes[:,0]
y = nodes[:,1]

############################################################################
# Given nodes for region of interest
nodes2 = np.array([
    [8.6, 113],
    [9.0, 123],
    [9.5, 127],
    [9.9, 140],
    [10.4, 150],
    [10.9, 167],
    [11.1, 180],
    [11.4, 200],
    [11.6, 210],
    [11.7, 217],
    [11.9, 223],
    [12.0, 233],
    [12.1, 247]
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
#first linear regression line
X1 = nodes[:4,0].reshape(-1, 1)  # Reshape for sklearn
Y1 = nodes[:4, 1]

# Create and fit the model
model = LinearRegression()
model.fit(X1, Y1)

# Extending the X1 range for visualization
X1_extended = np.linspace(0, 12, 100).reshape(-1, 1)
Y1_pred_extended = model.predict(X1_extended)
#############################################################################
#second linear regression line
X2 = nodes[-4:, 0].reshape(-1, 1)  # Correctly extract X values for the last three points
Y2 = nodes[-4:, 1]  # Correctly extract Y values for the last three points

# Re-create and fit the model using the last three points
model2 = LinearRegression()
model2.fit(X2, Y2)

# Extending the X2 range for visualization over the same range as defined earlier
X2_extended = np.linspace(9, 12, 100).reshape(-1, 1)
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
# Example error values (replace with your actual error data)
x_errors = np.array([0.1] * len(x))  # Standard error for x-values
y_errors = np.array([10] * len(y))  # Constant error for illustration
# Extracting x and y values
############################################################################
# Example error values (replace with your actual error data) for region of interest
x_errors2 = np.array([0.1] * len(x2))  # Standard error for x-values
y_errors2 = np.array([10] * len(y2))  # Constant error for illustration
# Extracting x and y values
############################################################################
# Performing spline interpolation
tck, u = interpolate.splprep( [x,y], s = 0 )
xnew, ynew = interpolate.splev( np.linspace( 0, 1, 100 ), tck, der = 0)

#############################################################################
# Plotting
plt.figure(figsize=(8, 6))
plt.plot(xnew ,ynew , '-', color = "blue")
# Add error bars for both x and y
plt.errorbar(x, y, xerr=x_errors, yerr=y_errors, fmt='x', color="black", ecolor='red', elinewidth=1, capsize=4, label='Original Data with Error Bars')
# Add error bars for both x and y
plt.errorbar(x2,y2, xerr=x_errors2, yerr=y_errors2, fmt='x',color='green',ecolor='orange',elinewidth=2, capsize=4)
plt.plot(x_fit, y_fit, color='green',label='Fitted Curve')
plt.legend(['Spline','Fitted curve','Original Data with error bar','Data of region of interest with error bar'], loc='upper left')
plt.plot(X1_extended, Y1_pred_extended, '--', color='red')  # Plotting the regression line 1
plt.plot(X2_extended, Y2_pred_extended, '--', color='red')  # Plotting the regression line 2
plt.title('Graph of current against voltage')
plt.xlabel('accelerating voltage, V')
plt.ylabel('Anode Current, mA')
plt.grid(True)
plt.axis([x.min() - 1, x.max() + 1, y.min() - 0.1, y.max() + 1])

# Setting x-axis scale with interval of 1
plt.xticks(np.arange(x.min() , x.max() + 2, 1))
# Setting y-axis scale with interval of 0.1
plt.yticks(np.arange(0 , y.max() + 75, 50))

# Adding minor grid with specified interval
ax = plt.gca()  # Get current axis
ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(plt.MultipleLocator(10))

# Enable minor grid with custom styling
ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')

# Save the plot to a new file
plt.savefig('finalPARTB.png')

plt.show()
