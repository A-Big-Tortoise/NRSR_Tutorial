import padasip as pa
from utils import plot_adp_filtered_signal
import numpy as np

def algo_lms(signal, d_signal, mu=0.1, w="random", show=False):
    """
    Adaptive LMS (Least Mean Squares) filtering algorithm for signal processing.

    Parameters:
        signal (array): The input signal to be filtered.
        d_signal (array): The desired signal (reference signal) for comparison.
        mu (float, optional): Learning rate, default is 0.1.
        w (array or str, optional): Initial filter parameters (weights).
            If "random," random initial weights are used. Otherwise, provide a custom array.
        show (bool, optional): If True, display a plot of the filtered signal and error.

    Returns:
        y (array): The filtered output signal.
        error (array): The error signal (the difference between the desired signal and the filtered output).
        weights (array): The set of filter parameters at the end of the simulation.

    Raises:
        ValueError: If the dimension of the input signal is not 1 or 2.

    This function implements an adaptive LMS filter to update filter weights and minimize
    the mean square error between the filtered output (y) and the desired signal (d_signal).

    The LMS algorithm adjusts the filter weights in the direction that reduces the error,
    making it suitable for various applications in signal processing and adaptive filtering.
    """

    if not isinstance(signal, np.ndarray):
        raise TypeError('signal should be np.ndarray')
    if not isinstance(d_signal, np.ndarray):
        raise ValueError('d_signal should be np.ndarray')

    ndim = signal.ndim

    # Check the dimension of the input signal.
    if ndim > 3:
        raise ValueError("dim of signal should be 1 or 2!")

    n = signal.shape[-1]

    # Initialize the LMS filter with the specified parameters.
    lms_filter = pa.filters.FilterLMS(n=n, mu=mu, w=w)

    # Run the LMS filter on the input signal.
    y, error, weights = lms_filter.run(d_signal, signal)

    # Display a plot of the filtered signal and error if 'show' is True.
    if show:
        plot_adp_filtered_signal(y, d_signal, error)

    return y, error, weights