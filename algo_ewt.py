import ewtpy

# Define a function to apply Empirical Wavelet Transform (EWT) to a signal
def algo_ewt(signal, N):
    """
    Applies the Empirical Wavelet Transform (EWT) to the input signal.

    Args:
    signal (array): Input signal array.
    N (int): Number of scales for the EWT.

    Returns:
    ewt (array): Resulting Empirical Wavelet Transform of the signal.
    """
    ewt, _, _ = ewtpy.EWT1D(signal, N)  # Compute the Empirical Wavelet Transform
    return ewt


if __name__ == '__main__':
    # Import necessary functions/classes from the Dataset module
    from Dataset import load_scg
    import matplotlib.pyplot as plt

    # Load data from the dataset
    signals, labels, duration, fs = load_scg(0.8, 'train')
    signal = signals[0]  # Select the first signal for processing
    N = 5  # Number of scales for EWT

    # Apply EWT to the selected signal
    ewt = algo_ewt(signal, N)

    # Create subplots to visualize the original signal and EWT results
    plt.subplots(N + 1, 1, figsize=(12, N * 5))
    plt.subplot(N + 1, 1, 1)
    plt.plot(signal, color='r')
    plt.title('Original Signal')

    # Plot each model result obtained from EWT
    for i in range(N):
        plt.subplot(N + 1, 1,  i+2)
        plt.plot(ewt[:, i])
        plt.title('Model {}'.format(i+1))

    plt.show()
