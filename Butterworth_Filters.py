from scipy.signal import butter, lfilter
import numpy as np

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
