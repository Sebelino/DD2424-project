import numpy as np

from plotting import plot_elapsed


def test_plot_elapsed():
    samples1 = np.random.normal(3, 1, 50)
    samples2 = np.random.normal(4, 1, 50)
    times = dict(
        baseline=samples1,
        experiment=samples2,
    )
    plot_elapsed("baseline", times)
