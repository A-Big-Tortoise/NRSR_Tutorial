import numpy as np

def fft_denoise(signal, threshold):
    """
    Applies FFT-based denoising to a signal.

    Parameters:
    signal (array-like): Input signal to be denoised.
    threshold (float): Threshold for filtering out noise.

    Returns:
    array-like: Denoised signal after applying FFT-based denoising.
    """

    num_samples = len(signal)  # Length of the input signal
    fhat = np.fft.fft(signal)  # Compute the FFT of the signal
    psd = fhat * np.conjugate(fhat) / num_samples  # Compute the power spectral density
    indices = psd > threshold  # Identify indices above the threshold for filtering
    fhat = indices * fhat  # Apply filtering to the FFT coefficients
    ffilt = np.fft.ifft(fhat)  # Compute the inverse FFT
    ffilt = ffilt.real  # Take the real part of the inverse FFT

    return ffilt

if __name__ == '__main__':
    # Import necessary functions/classes from Dataset module
    from Dataset import load_scg
    import matplotlib.pyplot as plt

    # Load SCG data with specified parameters
    # 'train': data split to load (train/validation/test)
    signals, labels, duration, fs = load_scg(0.1, 'train')

    # Choose a specific signal from the loaded dataset
    idx = 0
    signal = signals[idx]

    denoised_signal = fft_denoise(signal, 3e-12)
    plt.figure(figsize=(12, 6))
    plt.plot(signal, label='Noisy Signal')

    plt.plot(denoised_signal, label='Denoised Signal')
    plt.title('FFT Denoising')
    plt.legend()
    plt.show()
