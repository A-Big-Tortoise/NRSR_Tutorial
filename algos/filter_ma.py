import numpy as np

def simple_moving_average_filter(signal, length=10):
    """
    Smooths the input signal using a Simple Moving Average (SMA) filter.

    Args:
    signal (array): Input signal array.
    length (int): Length of the filter window.

    Returns:
    filtered_y (array): Smoothed signal.
    """
    # Create the Simple Moving Average weight array
    SMA = [1 / length] * length
    # Use convolution operation to filter the signal, 'same' option ensures output length matches input
    filtered_y = np.convolve(signal, SMA, 'same')

    return filtered_y

def exponential_moving_average_filter(signal, length=10, alpha=None):
    """
    Smooths the input signal using an Exponential Moving Average (EMA) filter.

    Args:
    signal (array): Input signal array.
    length (int): Length of the filter window.
    alpha (float, optional): Smoothing factor, if not provided, uses default value. Default value is 2 / (Length + 1).

    Returns:
    filtered_y (array): Smoothed signal.
    """
    # If alpha is not provided, use the default value
    if alpha is None:
        alpha = 2 / (length + 1)

    # Create the Exponential Moving Average weight array
    u = np.ones(length)
    n = np.arange(length)
    EMA = alpha * (1 - alpha) ** n * u
    # Use convolution operation to filter the signal, 'same' option ensures output length matches input
    filtered_y = np.convolve(signal, EMA, 'same')

    return filtered_y

if __name__ == '__main__':
    from Dataset import load_scg
    import matplotlib.pyplot as plt

    # Load SCG data with specified parameters
    signals, labels, duration, fs = load_scg(0.1, 'train')

    # Choose a specific signal from the loaded dataset
    idx = 0
    signal = signals[idx]

    sma_signal_5 = simple_moving_average_filter(signal, 5)
    sma_signal_10 = simple_moving_average_filter(signal, 10)
    sma_signal_20 = simple_moving_average_filter(signal, 20)

    ema_signal_5 = exponential_moving_average_filter(signal, 5)
    ema_signal_10 = exponential_moving_average_filter(signal, 10)
    ema_signal_20 = exponential_moving_average_filter(signal, 20)

    plt.subplots(2, 1, figsize=(24, 14))
    plt.subplot(2, 1, 1)
    plt.plot(signal, alpha=0.5, label='Original Signal')
    plt.plot(sma_signal_5, label='Simple Moving Average length=5')
    plt.plot(sma_signal_10, label='Simple Moving Average length=10')
    plt.plot(sma_signal_20, label='Simple Moving Average length=20')
    plt.legend()
    plt.title("Simple Moving Average")

    plt.subplot(2, 1, 2)
    plt.plot(signal, alpha=0.5, label='Original Signal')
    plt.plot(ema_signal_5, label='Exponential Moving Average length=5')
    plt.plot(ema_signal_10, label='Exponential Moving Average length=10')
    plt.plot(ema_signal_20, label='Exponential Moving Average length=20')
    plt.title("Exponential Moving Average")
    plt.legend()
    plt.show()


