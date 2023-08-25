from scipy.signal import butter, lfilter

def butter_bandpass_filter(signal, lowcut, highcut, fs, order=5):
    b, a = butter(order, [lowcut, highcut], btype='bandpass', analog=False, output='ba', fs=fs)
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal

def butter_bandstop_filter(signal, lowcut, highcut, fs, order=5):
    b, a = butter(order, [lowcut, highcut], btype='bandstop', analog=False, output='ba', fs=fs)
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal


def butter_lowpass_filter(signal, cutoff, fs, order=5):
    b, a = butter(order, cutoff, btype='lowpass', analog=False, output='ba', fs=fs)
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal


def butter_highpass_filter(signal, cutoff, fs, order=5):
    b, a = butter(order, cutoff, btype='highpass', analog=False, output='ba', fs=fs)
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal


if __name__ == '__main__':
    # Import necessary functions/classes from Dataset module
    from Dataset import load_scg
    import matplotlib.pyplot as plt

    # Load SCG data with specified parameters
    signals, labels, duration, fs = load_scg(0.1, 'train')

    idx = 0
    signal = signals[idx]

    signal_bandpass = butter_bandpass_filter(signal, 1, 10, fs)
    signal_bandstop = butter_bandstop_filter(signal, 1, 10, fs)
    signal_lowpass = butter_lowpass_filter(signal, 1, fs)
    signal_highpass = butter_highpass_filter(signal, 10, fs)

    plt.subplots(4, 1, figsize=(14, 24))
    plt.subplot(4, 1, 1)
    plt.plot(signal, alpha=0.5, label='Original Signal')
    plt.plot(signal_bandpass, label='Bandpass Signal')
    plt.title("Bandpass Filter:lowcut=1, highcut=10")

    plt.subplot(4, 1, 2)
    plt.plot(signal, alpha=0.5, label='Original Signal')
    plt.plot(signal_bandstop, label='Bandstop Signal')
    plt.title("Bandstop Filter:lowcut=1, highcut=10")

    plt.subplot(4, 1, 3)
    plt.plot(signal, alpha=0.5, label='Original Signal')
    plt.plot(signal_lowpass, label='Lowpass Signal')
    plt.title("Lowpass Filter: cutoff=1")

    plt.subplot(4, 1, 4)
    plt.plot(signal, alpha=0.5, label='Original Signal')
    plt.plot(signal_highpass, label='Highpass Signal')
    plt.title("Highpass Filter:cutoff=10")

    plt.show()
