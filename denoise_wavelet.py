import pywt

def wdenoise(data, method, threshold):
    """
    Applies wavelet-based denoising to the input data.

    Parameters:
    data (array-like): Input data to be denoised.
    method (str): Wavelet transform method to be used. like 'sym4' and so on. 
    threshold (float): Threshold for filtering out noise.

    Returns:
    array-like: Denoised data after applying wavelet-based denoising.
    """

    # Create a Wavelet object using the specified method
    w = pywt.Wavelet(method)
    
    # Calculate the maximum decomposition level based on data length and wavelet length
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    
    print("maximum level is " + str(maxlev))
    
    # Perform wavelet decomposition on the input data up to the maximum level
    coeffs = pywt.wavedec(data, method, level=maxlev)
    
    # Loop through the wavelet coefficients (except the first one, which is the approximation)
    for i in range(1, len(coeffs)):
        # Apply thresholding to each coefficient by multiplying with a factor of the maximum coefficient
        coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))
    
    # Reconstruct the denoised data using the modified wavelet coefficients
    datarec = pywt.waverec(coeffs, method)
    
    return datarec

if __name__ == '__main__':
    from Dataset import load_scg
    import matplotlib.pyplot as plt

    # Load SCG data with specified parameters
    signals, labels, duration, fs = load_scg(0.1, 'train')

    # Choose a specific signal from the loaded dataset
    idx = 0
    signal = signals[idx]
    denoised_signal = wdenoise(signal, 'sym4', 0.12)

    plt.figure(figsize=(12, 6))
    plt.plot(signal, label='Noisy Signal')
    plt.plot(denoised_signal, label='Denoised Signal')
    plt.title('Wavelet Denoising')

    plt.legend()
    plt.show()


