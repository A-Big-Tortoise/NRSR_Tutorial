from scipy.signal import iirnotch, lfilter

def notch_filter(signal, cutoff=10, q=10, fs=100):
    """
    Apply a Notch Filter to Remove Interference at a Specific Frequency.

    Args:
        signal (array-like): The input signal to be filtered.
        cutoff (float, optional): The center frequency to be removed (in Hz). Default is 10 Hz.
        q (float, optional): The quality factor or Q factor of the filter. Higher values result in narrower notches. Default is 10.
        fs (float, optional): The sampling frequency of the input signal (in Hz). Default is 100 Hz.

    Returns:
        array-like: The filtered signal with the specified frequency removed.

    Notes:
        - This function uses SciPy's IIR notch filter implementation to suppress interference at the specified frequency.
        - The notch filter is used to eliminate a narrow frequency band around the 'cutoff' frequency.
        - The 'q' parameter controls the width of the notch; higher 'q' values create narrower notches.

    Example:
        >>> import numpy as np
        >>> from scipy.signal import lfilter
        >>> noisy_signal = np.sin(2 * np.pi * 50 * np.linspace(0, 1, 1000)) + 0.5 * np.random.randn(1000)
        >>> filtered_signal = notch_filter(noisy_signal, cutoff=50, q=30, fs=1000)
    """
    # Create an IIR Notch filter with specified parameters
    b, a = iirnotch(cutoff, q, fs)

    # Apply the Notch filter to the input signal
    filtered_signal = lfilter(b, a, signal)

    return filtered_signal
