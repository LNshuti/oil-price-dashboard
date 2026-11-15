import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from scr.visualization.plot import get_plot_best_fit_line

matplotlib.use("Agg")


def test_linear_plot():
    x_array = np.array([1.0, 2.0, 3.0])
    y_array = np.array([3.0, 5.0, 7.0])
    title = "Test plot for linear data"

    fig = get_plot_best_fit_line(2.0, 1.0, x_array, y_array, title)
    ax = fig.axes[0]

    assert ax.get_title() == title
    line = ax.get_lines()[0]
    np.testing.assert_allclose(line.get_ydata(), 2.0 * x_array + 1.0)
    plt.close(fig)
