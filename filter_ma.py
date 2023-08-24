import numpy as np

def simple_moving_average_filter(signal, Length):
    """
    Smooths the input signal using a Simple Moving Average (SMA) filter.

    Args:
    signal (array): Input signal array.
    Length (int): Length of the filter window.

    Returns:
    filtered_y (array): Smoothed signal.
    """
    # Create the Simple Moving Average weight array
    SMA = [1 / Length] * Length
    # Use convolution operation to filter the signal, 'same' option ensures output length matches input
    filtered_y = np.convolve(signal, SMA, 'same')

    return filtered_y

def exponential_moving_average_filter(signal, Length, alpha=None):
    """
    Smooths the input signal using an Exponential Moving Average (EMA) filter.

    Args:
    signal (array): Input signal array.
    Length (int): Length of the filter window.
    alpha (float, optional): Smoothing factor, if not provided, uses default value. Default value is 2 / (Length + 1).

    Returns:
    filtered_y (array): Smoothed signal.
    """
    # If alpha is not provided, use the default value
    if alpha is None:
        alpha = 2 / (Length + 1)

    # Create the Exponential Moving Average weight array
    u = np.ones(Length)
    n = np.arange(Length)
    EMA = alpha * (1 - alpha) ** n * u
    # Use convolution operation to filter the signal, 'same' option ensures output length matches input
    filtered_y = np.convolve(signal, EMA, 'same')

    return filtered_y

