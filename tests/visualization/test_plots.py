import pytest 
import numpy as np 
from visualization import get_plot_best_fit_line

# This test can be ran using 
# !pytest -k "test_linear_plot" --mpl-generate-path visualization/baseline
@pytest.mark.mpl_image_compare 
def test_liner_plot():
    slope = 2.0 
    intercept = 1.0 
    x_array = np.array([1.0, 2.0, 3.0])
    y_array = np.array([3.0, 5.0, 7.0])
    title = "Test plot for linear data"
    return get_plot_best_fit_line(slope, intercept, x_array, y_array, title)

