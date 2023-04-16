import matplotlib.pyplot as plt
import numpy as np

def get_plot_best_fit_line(slope, intercept, x_array, y_array, title):
    """
    slope: slope of best fit line
    intercept: intercept of best fit line
    x_array: array of x-axis data points
    y_array: array of y-axis data points
    title: title of the plot

    returns: None
    """
    # Convert x_array and y_array to numpy arrays
    x_array = np.array(x_array)
    y_array = np.array(y_array)

    # Calculate the best-fit line's y-axis values
    best_fit_line = slope * x_array + intercept

    # Create a scatter plot of the data points
    plt.scatter(x_array, y_array, label="Data points")

    # Plot the best-fit line
    plt.plot(x_array, best_fit_line, color='red', label="Best-fit line")

    # Set plot labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title(title)
    plt.legend()

    # Display the plot
    plt.show()
