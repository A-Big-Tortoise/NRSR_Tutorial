import numpy as np
from PyEMD import EEMD, EMD, CEEMDAN
from vmdpy import VMD
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import scipy.signal
from scipy.signal import butter, lfilter, iirnotch, correlate
import padasip as pa
from dsp_utils import plot_sim_waves, plot_noise_signal, plot_decomposed_components, plot_filtered_signal
import pywt
import pandas as pd

import os
# ==============================================================================
# ------------------------------------Waves-------------------------------------
# ==============================================================================

def sine_wave(duration=10, sampling_rate=100, amplitude=1, frequency=1, phase=0, show=False):
    """
    Generate a sine wave signal.

    Parameters:
    duration : float, optional
        The duration (in seconds) of the generated square wave signal.
    sampling_rate : int, optional
        The number of samples per second used for discretization.
    amplitude : float
        The peak deviation of the function from zero.
    frequency : float
        The number of oscillations (cycles) that occur each second of time.
    phase : float
        Phase specifies (in radians) where in its cycle the oscillation is at t = 0.

    Returns:
    sine_wave : array-like
        An array containing the values of the sine wave signal at the given time points.

    """
    time = np.linspace(0, duration, duration * sampling_rate, endpoint=False)
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * time + phase)

    if show:
        plot_sim_waves(sine_wave, 'Sine Wave')

    return sine_wave


def triangle_wave(duration=10, sampling_rate=100, amplitude=1, period=2, show=False):
    """
    Generate a triangle wave signal.

    Parameters:
    duration : float, optional
        The duration (in seconds) of the generated square wave signal.
    sampling_rate : int, optional
        The number of samples per second used for discretization.
    Amplitude : float
        The amplitude of the triangle wave.
    period : float
        The period of the triangle wave.

    Returns:
    triangle_wave : array-like
        An array containing the values of the triangle wave signal at the given time points.
    """

    time = np.linspace(0, duration, duration * sampling_rate, endpoint=False)
    # Scale the time values to a normalized range [-1, 1] within each period
    t_scaled = 1 * np.abs(2 * (time / period - np.floor(time / period + 0.5))) - 1

    # Calculate the triangle wave values based on scaled time values
    triangle_wave = (3 * amplitude / period) * np.abs((t_scaled - period / 4) % period - period / 2) - amplitude

    if show:
        plot_sim_waves(triangle_wave, 'Triangle Wave')

    return triangle_wave



def square_wave(duration=10, sampling_rate=100, frequency=1, show=False):
    """
    Generate a square wave signal.

    Parameters:
    duration : float, optional
        The duration (in seconds) of the generated square wave signal.
    sampling_rate : int, optional
        The number of samples per second used for discretization.
    frequency : float
        The frequency of the square wave.

    Returns:
    square_wave : array-like
        An array containing the values of the square wave signal at the given time points.
    """
    time = np.linspace(0, duration, duration * sampling_rate, endpoint=False)
    square_wave = 2 * (2 * np.floor(frequency * time) - np.floor(2 * frequency * time)) + 1

    if show:
        plot_sim_waves(square_wave, 'Square Wave')

    return square_wave


def chirp_wave_linear(duration=10, sampling_rate=100, f0=1, c=1, phase=0, show=False):
    """
    Generate a linear chirp wave signal.

    Parameters:
    duration : float, optional
        The duration (in seconds) of the generated linear chirp wave signal.
    sampling_rate : int, optional
        The number of samples per second used for discretization.
    f0 : float
        Initial frequency of the chirp.
    c : float
        Chirp rate (slope) in Hertz/second.
    phase : float
        Phase offset of the chirp.

    Returns:
    chrip_wave_linear : array-like
        An array containing the values of the linear chirp wave signal at the given time points.
    """
    time = np.linspace(0, duration, duration * sampling_rate, endpoint=False)

    # Calculate the final frequency of the chirp
    f1 = c * duration + f0

    # Check for valid frequency and Nyquist conditions
    if f0 <= 0 or f1 <= 0:
        raise ValueError(f"Initial Frequency is {f0} and Final Frequency {f1}. Frequency should be larger than 0!")

    if sampling_rate / 2 <= f0 or sampling_rate / 2 <= f1:
        raise ValueError(
            f"Sampling rate is {sampling_rate} and Initial Frequency is {f0} and Final Frequency {f1}. Nyquist Error!")

    chirp_wave = np.sin(phase + 2 * np.pi * ((c / 2) * (time ** 2) + f0 * time))

    if show:
        plot_sim_waves(chirp_wave, 'Chirp Wave Linear')

    return chirp_wave



def chirp_wave_exponential(duration=10, sampling_rate=100, f0=1, k=1.2, phase=0, show=False):
    """
    Generate an exponential chirp wave signal.

    Parameters:
    duration : float, optional
        The duration (in seconds) of the generated exponential chirp wave signal.
    sampling_rate : int, optional
        The number of samples per second used for discretization.
    f0 : float
        Initial frequency of the chirp.
    k : float
        Exponential growth factor.
    phase : float
        Phase offset of the chirp.

    Returns:
    chirp_wave_exponential : array-like
        An array containing the values of the exponential chirp wave signal at the given time points.
    """
    time = np.linspace(0, duration, duration * sampling_rate, endpoint=False)

    # Calculate the final frequency of the exponential chirp
    f1 = f0 * (k ** duration - 1)

    # Check for valid frequency and Nyquist conditions
    if f0 <= 0 or f1 <= 0:
        raise ValueError(f"Initial Frequency is {f0} and Final Frequency {f1}. Frequency should be larger than 0!")

    if sampling_rate / 2 <= f0 or sampling_rate / 2 <= f1:
        raise ValueError(
            f"Sampling rate is {sampling_rate} and Initial Frequency is {f0} and Final Frequency {f1}. Nyquist Error!")

    # Generate the exponential chirp wave signal based on the parameters
    chirp_wave = np.sin(phase + 2 * np.pi * f0 * ((k ** time - 1) / np.log(k)))

    if show:
        plot_sim_waves(chirp_wave, 'Chirp Wave Exponential')

    return chirp_wave


def chirp_wave_hyperbolic(duration=10, sampling_rate=100, f0=1, f1=10, phase=0, show=False):
    """
    Generate a hyperbolic chirp wave signal.

    Parameters:
    duration : float, optional
        The duration (in seconds) of the generated hyperbolic chirp wave signal.
    sampling_rate : int, optional
        The number of samples per second used for discretization.
    f0 : float
        Initial frequency of the chirp.
    f1 : float
        Final frequency of the chirp.
    duration : float
        Duration of the chirp.
    phase : float
        Phase offset of the chirp.

    Returns:
    chirp_wave_hyperbolic : array-like
        An array containing the values of the hyperbolic chirp wave signal at the given time points.
    """
    time = np.linspace(0, duration, duration * sampling_rate, endpoint=False)

    # Check for valid frequencies and Nyquist conditions
    if f0 <= 0 or f1 <= 0:
        raise ValueError(f"Initial Frequency is {f0} and Final Frequency {f1}. Frequency should be larger than 0!")

    if sampling_rate / 2 <= f0 or sampling_rate / 2 <= f1:
        raise ValueError(f"Sampling rate is {sampling_rate} and Initial Frequency is {f0} and Final Frequency {f1}. Nyquist Error!")

    chirp_wave = np.sin(phase + 2 * np.pi * ((-1 * f0 * f1 * duration) / (f1 - f0) * np.log(1 - (f1 - f0) / (f1 * duration) * time)))

    if show:
        plot_sim_waves(chirp_wave, 'Chirp Wave Hyperbolic')

    return chirp_wave

def pulse_wave(duration=10, sampling_rate=100, amplitude=1, d=0.5, frequency=1, expansion=5, show=False):
    """
    Generate a pulse wave signal.

    Parameters:
    duration : float, optional
        The duration (in seconds) of the generated pulse wave signal.
    sampling_rate : int, optional
        The number of samples per second used for discretization.
    amplitude : float
        The amplitude of the pulse wave.
    d : float
        Width of the pulse.
    frequency : float
        The frequency of the pulse wave.
    expansion : int
        The number of terms used in the expansion for generating the pulse wave.

    Returns:
    pulse_wave : array-like
        An array containing the values of the pulse wave signal at the given time points.
    """
    time = np.linspace(0, duration, duration * sampling_rate, endpoint=False)

    sum_of_ = 0

    # Check for Nyquist condition
    if sampling_rate / 2 <= frequency:
        raise ValueError(f"Sampling rate is {sampling_rate} and Frequency is {frequency}. Nyquist Error!")

    for n in range(1, expansion+1):
        sum_of_ += np.sinc(n * d) * np.cos(2 * np.pi * n * frequency * time)

    # Calculate the final pulse wave signal
    pulse_wave = amplitude * d * (1 + 2 * sum_of_)

    if show:
        plot_sim_waves(pulse_wave, 'Pulse Wave')

    return pulse_wave


# ==============================================================================
# ------------------------------------Noise-------------------------------------
# ==============================================================================


def add_white_noise(signal, noise_amplitude=0.1, model=0, show=False):
    """
    Add white noise to a signal.

    Parameters:
    signal : array-like
        The input signal to which white noise will be added.
    noise_amplitude : float, optional
        The amplitude of the white noise to be added.
    model : int, optional
        The type of noise model to use:
        - 0: Gaussian noise
        - 1: Laplace noise
    show : bool, optional
        Whether to display a plot of the noisy signal.

    Returns:
    noisy_signal : array-like
        An array containing the values of the signal with added white noise.
    """
    # Calculate the standard deviation of the input signal
    signal_sd = np.std(signal, ddof=1)

    # Calculate the amplitude of the noise to be added
    amp = signal_sd * noise_amplitude

    _noise = 0
    if model == 0:
        # Generate Gaussian noise with the specified amplitude
        _noise = np.random.normal(0, amp, len(signal))
    elif model == 1:
        # Generate Laplace noise with the specified amplitude
        _noise = np.random.laplace(0, amp, len(signal))

    # Add the generated noise to the input signal
    noisy_signal = _noise + signal

    if show:
        # If requested, plot the original and noisy signals
        plot_noise_signal(signal, noisy_signal, 'Add White Noise')

    return noisy_signal


# 和上面的白噪声一样，没有考虑过noise_freq和noise_duration，可能后期需要大改
def add_band_limited_white_noise(
        signal, noise_amplitude=0.1, sampling_rate=100, lowcut=0.1, highcut=5, order=3, show=False
):
    """
    Add band-limited white noise to a signal.

    Parameters:
    signal : array-like
        The input signal to which band-limited white noise will be added.
    noise_amplitude : float, optional
        The amplitude of the white noise to be added.
    sampling_rate : int, optional
        The sampling rate of the signal.
    lowcut : float, optional
        The low cutoff frequency of the bandpass filter.
    highcut : float, optional
        The high cutoff frequency of the bandpass filter.
    order : int, optional
        The order of the bandpass filter.
    show : bool, optional
        Whether to display a plot of the noisy signal.

    Returns:
    noisy_signal : array-like
        An array containing the values of the signal with added band-limited white noise.
    """
    # Calculate the standard deviation of the input signal
    signal_sd = np.std(signal, ddof=1)

    # Calculate the amplitude of the white noise to be added
    amp = signal_sd * noise_amplitude

    # Generate white noise with the specified amplitude
    _noise = np.random.normal(0, amp, len(signal))

    # Design a bandpass filter with the specified parameters
    b, a = butter(order, [lowcut, highcut], btype='band', fs=sampling_rate)

    # Apply the bandpass filter to the generated white noise
    _band_limited_noise = lfilter(b, a, _noise)

    # Add the band-limited noise to the input signal
    noisy_signal = _band_limited_noise + signal

    if show:
        # If requested, plot the original and noisy signals
        plot_noise_signal(signal, noisy_signal, 'Add Band-limited White Noise')

    return noisy_signal


def add_impulsive_noise(
        signal, noise_amplitude=1, rate=None, number=None, show=False
):
    """
    Add impulsive noise to a signal.

    Parameters:
    signal : array-like
        The input signal to which impulsive noise will be added.
    noise_amplitude : float, optional
        The amplitude of the impulsive noise.
    rate : float, optional
        The probability of an impulsive noise event per sample.
    number : int, optional
        The total number of impulsive noise events to add.
    show : bool, optional
        Whether to display a plot of the noisy signal.

    Returns:
    noisy_signal : array-like
        An array containing the values of the signal with added impulsive noise.
    """
    # Calculate the standard deviation of the input signal
    signal_sd = np.std(signal, ddof=1)

    # Calculate the amplitude of the impulsive noise to be added
    amp = signal_sd * noise_amplitude

    # Get the number of samples in the input signal
    num_samples = len(signal)

    # Determine the probability distribution for impulsive noise events based on rate or number
    if rate is not None and number is None:
        pob = [1 - rate, rate]
    elif rate is None and number is not None:
        pob_rate = number / num_samples
        if pob_rate >= 1.0:
            pob_rate = 1
        pob = [1 - pob_rate, pob_rate]
    else:
        return None

    # Generate impulsive noise events based on the probability distribution
    impulsive_noise = np.random.choice([0, 1], size=num_samples, p=pob) * np.random.normal(0, amp, num_samples)

    # Add the impulsive noise to the input signal
    noisy_signal = np.abs(impulsive_noise) + signal

    if show:
        # If requested, plot the original and noisy signals
        plot_noise_signal(signal, noisy_signal, 'Add Impulsive Noise')

    return noisy_signal


def add_burst_noise(
        signal, noise_amplitude=0.3, burst_num_max=1, burst_durations=[10, 100], show=False
):
    """
    Add burst noise to a signal.

    Parameters:
    signal : array-like
        The input signal to which burst noise will be added.
    noise_amplitude : float, optional
        The amplitude of the burst noise.
    burst_num_max : int, optional
        The maximum number of burst noise events to add.
    burst_durations : list, optional
        A list containing the minimum and maximum durations (in samples) of burst noise events.
    burst_intervals : list, optional
        A list containing the minimum and maximum intervals (in samples) between burst noise events.
    show : bool, optional
        Whether to display a plot of the noisy signal.

    Returns:
    noisy_signal : array-like
        An array containing the values of the signal with added burst noise.
    """
    # Get the length of the input signal
    signal_length = len(signal)

    # Initialize an array to store the burst noise
    _noise = np.zeros(signal_length)

    # Calculate the standard deviation of the input signal
    signal_sd = np.std(signal, ddof=1)

    # Calculate the amplitude of the burst noise to be added
    amp = noise_amplitude * signal_sd

    # Initialize the starting point for burst noise events
    burst_start = np.random.randint(0, (signal_length - burst_durations[1] + 1) // burst_num_max)

    # Generate burst noise events based on specified parameters
    for _ in range(burst_num_max):
        burst_duration = np.random.randint(burst_durations[0], burst_durations[1])
        burst_end = burst_start + burst_duration

        if burst_end >= signal_length:
            burst_end = signal_length

        burst_end = burst_start + burst_duration

        _noise[burst_start: burst_end] += np.random.normal(0, amp, size=burst_end-burst_start)

    # Add the burst noise to the input signal
    noisy_signal = _noise + signal

    if show:
        # If requested, plot the original and noisy signals
        plot_noise_signal(signal, noisy_signal, 'Add Burst Noise')

    return noisy_signal


def spectral_density(frequency_range, magnitude=1, noise_exponent=1):
    """
    Calculate the spectral density of pink noise.

    Parameters:
        frequency_range (array-like): Array of positive frequencies.
        Magnitude (float): Magnitude of the noise.
        noise_exponent (float): Exponent determining the slope of the spectral density.

    Returns:
        array: Spectral density values.
    """
    return magnitude / (frequency_range ** noise_exponent)

def add_colored_noise(
        signal, noise_amplitude=0.3, model=0, sampling_rate=100, duration=10,  show=False
):
    """
    Add colored noise to a signal.

    Parameters:
    signal : array-like
        The input signal to which colored noise will be added.
    noise_amplitude : float, optional
        The amplitude of the noise.
    sampling_rate : int, optional
        The sampling rate of the audio signal.
    duration : float, optional
        Duration of the colored noise signal in seconds.
    model : int, optional
        The type of colored noise to generate:
        - 0: Pink noise
        - 1: Brown noise
    show : bool, optional
        Whether to display a plot of the noisy signal.

    Returns:
    noisy_signal : array-like
        An array containing the values of the signal with added colored noise.
    """
    if model == 0:
        # Pink noise
        noise_exponent = 1
        magnitude = 1
    elif model == 1:
        # Brown noise
        noise_exponent = 2
        magnitude = 1

    num_samples = int(sampling_rate * duration)
    frequency_range = np.fft.fftfreq(num_samples)[1: num_samples // 2]

    # Calculate spectral density using the provided function
    _spectral_density = spectral_density(frequency_range, magnitude, noise_exponent)

    # Generate random phases for each frequency component
    random_phases = np.random.uniform(0, 2 * np.pi, len(frequency_range))

    # Combine magnitude and phases to form the complex spectrum
    spectrum = np.sqrt(_spectral_density) * np.exp(1j * random_phases)

    # Perform inverse FFT to convert complex spectrum to time-domain signal
    _colored_noise = np.fft.irfft(spectrum, n=num_samples)

    # Scale the colored noise to achieve the desired maximum amplitude
    _colored_noise *= np.max(signal) * noise_amplitude

    # Add the colored noise to the input signal
    noisy_signal = _colored_noise + signal

    if show:
        # If requested, plot the original and noisy signals
        plot_noise_signal(signal, noisy_signal, 'Add Colored Noise')

    return noisy_signal


def add_flicker_noise(
        signal, noise_amplitude=0.3, sampling_rate=100, duration=10, magnitude=1, noise_exponent=1, show=False
):
    """
    Add flicker (1/f) noise to a signal.

    Parameters:
    signal : array-like
        The input signal to which flicker noise will be added.
    noise_amplitude : float, optional
        The amplitude of the burst noise.
    sampling_rate : int, optional
        The sampling rate of the audio signal.
    duration : float, optional
        Duration of the flicker noise signal in seconds.
    magnitude : float, optional
        Magnitude of the flicker noise.
    noise_exponent : float, optional
        Exponent determining the slope of the spectral density.
    show : bool, optional
        Whether to display a plot of the noisy signal.

    Returns:
    noisy_signal : array-like
        An array containing the values of the signal with added flicker noise.
    """
    num_samples = int(sampling_rate * duration)
    frequency_range = np.fft.fftfreq(num_samples)[1: num_samples // 2]

    # Calculate spectral density using the provided function
    _spectral_density = spectral_density(frequency_range, magnitude, noise_exponent)

    # Generate random phases for each frequency component
    random_phases = np.random.uniform(0, 2 * np.pi, len(frequency_range))

    # Combine magnitude and phases to form the complex spectrum
    spectrum = np.sqrt(_spectral_density) * np.exp(1j * random_phases)

    # Perform inverse FFT to convert complex spectrum to time-domain signal
    _flicker_noise = np.fft.irfft(spectrum, n=num_samples)

    # Scale the flicker noise to achieve the desired maximum amplitude
    _flicker_noise *= np.max(signal) * noise_amplitude

    # Add the flicker noise to the input signal
    noisy_signal = _flicker_noise + signal

    if show:
        # If requested, plot the original and noisy signals
        plot_noise_signal(signal, noisy_signal, 'Add Flicker Noise')

    return noisy_signal

def add_thermal_noise(
        signal, noise_amplitude=0.3, sampling_rate=100, duration=10, Temperature=100, show=False
):
    """
    Add thermal noise to a signal.

    Parameters:
    signal : array-like
        The input signal to which thermal noise will be added.
    noise_amplitude : float, optional
        The amplitude of the burst noise.
    sampling_rate : int, optional
        The sampling rate of the audio signal.
    duration : float, optional
        Duration of the thermal noise signal in seconds.
    Temperature : float, optional
        Temperature in Kelvin, used to calculate thermal noise.
    show : bool, optional
        Whether to display a plot of the noisy signal.

    Returns:
    noisy_signal : array-like
        An array containing the values of the signal with added thermal noise.
    """
    num_samples = int(sampling_rate * duration)
    frequency_range = np.fft.fftfreq(num_samples)[1: num_samples // 2]

    # Calculate spectral density based on temperature
    k = 1.38e-23  # Boltzmann constant
    _spectral_density = k * Temperature / 2

    # Generate random phases for each frequency component
    random_phases = np.random.uniform(0, 2 * np.pi, len(frequency_range))

    # Combine magnitude and phases to form the complex spectrum
    spectrum = np.sqrt(_spectral_density) * np.exp(1j * random_phases)

    # Perform inverse FFT to convert complex spectrum to time-domain signal
    _thermal_noise = np.fft.irfft(spectrum, n=num_samples)

    # Scale the thermal noise to achieve the desired maximum amplitude
    _thermal_noise *= np.max(signal) * noise_amplitude

    # Add the thermal noise to the input signal
    noisy_signal = _thermal_noise + signal

    if show:
        # If requested, plot the original and noisy signals
        plot_noise_signal(signal, noisy_signal, 'Add Thermal Noise')

    return noisy_signal


def add_powerline_noise(
        signal, sampling_rate=100, duration=10, powerline_frequency=50, powerline_amplitude=0.1, show=False
):
    """
    Add powerline noise (mains hum) to a signal.

    Parameters:
    signal : array-like
        The input signal to which powerline noise will be added.
    sampling_rate : int, optional
        The sampling rate of the audio signal.
    duration : float, optional
        Duration of the powerline noise signal in seconds.
    powerline_frequency : float, optional
        Frequency of the powerline (mains) noise in Hertz.
    powerline_amplitude : float, optional
        Amplitude of the powerline noise.
    show : bool, optional
        Whether to display a plot of the noisy signal.

    Returns:
    noisy_signal : array-like
        An array containing the values of the signal with added powerline noise.
    """
    nyquist = sampling_rate * 0.5

    # Check if the specified powerline frequency is above the Nyquist frequency
    if powerline_frequency > nyquist:
        return np.zeros(len(signal))

    # Calculate the standard deviation of the input signal
    signal_sd = np.std(signal, ddof=1)

    # Generate the powerline noise as a sine wave
    powerline_noise = sine_wave(duration=duration, sampling_rate=sampling_rate, amplitude=1, frequency=powerline_frequency, phase=0)


    # Scale the amplitude of the powerline noise
    powerline_amplitude *= signal_sd
    powerline_noise *= powerline_amplitude

    # Add the powerline noise to the input signal
    noisy_signal = powerline_noise + signal

    if show:
        # If requested, plot the original and noisy signals
        plot_noise_signal(signal, noisy_signal, 'Add Powerline Noise')

    return noisy_signal


def add_echo_noise(
    signal, n_echo=5, attenuation_factor=[0.5, 0.4, 0.3, 0.2, 0.1], delay_factor=[5] * 5, show=False
):
    """
    Add echo noise to a signal.

    Parameters:
    signal : array-like
        The input signal to which echo noise will be added.
    n_echo : int, optional
        The number of echo repetitions to add.
    attenuation_factor : list or numpy.ndarray, optional
        A list of attenuation factors for each echo.
    delay_factor : list or numpy.ndarray, optional
        A list of delay factors (in samples) for each echo.
    show : bool, optional
        Whether to display a plot of the noisy signal.

    Returns:
    noisy_signal : array-like
        An array containing the values of the signal with added echo noise.
    """
    # Check the types and lengths of attenuation and delay factors
    if not isinstance(attenuation_factor, (list, np.ndarray)):
        raise ValueError("The type of attenuation_factor must be a list or numpy.ndarray")
    if not isinstance(delay_factor, (list, np.ndarray)):
        raise ValueError("The type of delay_factor must be a list or numpy.ndarray")
    if len(attenuation_factor) != n_echo or len(delay_factor) != n_echo:
        raise ValueError("The lengths of attenuation_factor and delay_factor should be equal to n_echo")

    # Create a copy of the original signal
    original_signal = signal.copy()
    echos = np.zeros(shape=original_signal.shape)
    # Iterate over each echo and apply attenuation and delay
    for a_factor, d_factor in zip(attenuation_factor, delay_factor):
        # Apply attenuation to the signal
        attenuation_signal = original_signal * a_factor

        # Shift the attenuated signal to create the echo effect
        attenuation_signal[d_factor:] = attenuation_signal[:-d_factor]
        attenuation_signal[:d_factor] = 0

        # Add the attenuated and delayed signal to the original signal
        echos += attenuation_signal

    # Combine the original signal with all the echoes to create the noisy signal
    noisy_signal = echos + signal

    if show:
        # If requested, plot the original and noisy signals
        plot_noise_signal(signal, noisy_signal, 'Add Echo Noise')

    return noisy_signal


def add_click_noise(
    signal, noise_amplitude=0.1, n_click=5, show=False
):
    """
    Add click noise to a signal.

    Parameters:
    signal : array-like
        The input signal to which click noise will be added.
    noise_amplitude : float, optional
        Amplitude of the click noise.
    n_click : int, optional
        The number of clicks to add.
    show : bool, optional
        Whether to display a plot of the noisy signal.

    Returns:
    noisy_signal : array-like
        An array containing the values of the signal with added click noise.
    """
    # Calculate the standard deviation of the input signal
    signal_sd = np.std(signal, ddof=1)

    # Calculate the amplitude of the click noise
    amp = signal_sd * noise_amplitude

    # Generate random positions for the clicks within the signal
    noise_pos = (np.random.uniform(0, len(signal), n_click)).astype(int)

    # Create a mask to represent the click positions
    mask = np.zeros(len(signal))
    mask[noise_pos] = 1

    # Generate click noise and apply the mask to place the clicks
    _click_noise = np.random.normal(0, amp, len(signal)) * mask

    # Add the click noise to the input signal
    noisy_signal = _click_noise + signal

    if show:
        plot_noise_signal(signal, noisy_signal, 'Add Click Noise')

    return noisy_signal


def add_distort_noise(
    signal, n_samples, sampling_rate=100, noise_frequency=10, noise_amplitude=0.1, show=False):
    """
    Generate a noisy signal with distorted noise.

    Parameters:
    signal : array-like
        The input signal to which distorted noise will be added.
    n_samples : int
        Number of samples in the output signal.
    sampling_rate : int, optional
        Sampling rate of the signal (default is 1000 Hz).
    noise_frequency : int, optional
        Frequency of the noise signal (default is 100 Hz).
    noise_amplitude : float, optional
        Amplitude of the noise signal (default is 0.1).
    show : bool, optional
        Whether to display a plot of the original and noisy signals.

    Returns:
    noisy_signal : array-like
        An array containing the values of the generated noisy signal.
    """
    # Check if the number of samples matches the length of the input signal
    if n_samples != len(signal):
        print('n_samples should be equal to the length of signal')
        return None

    # Initialize an array to store the generated noise samples
    _noise = np.zeros(n_samples)

    # Apply a very conservative Nyquist criterion to ensure sufficiently sampled signals.
    nyquist = sampling_rate * 0.4
    if noise_frequency > nyquist:
        print(
            f"Skipping requested noise frequency of {noise_frequency} Hz since it cannot be resolved at "
            f"the sampling rate of {sampling_rate} Hz. Please increase sampling rate to {noise_frequency * 2.5} Hz or choose "
            f"frequencies smaller than or equal to {nyquist} Hz."
        )

    # Calculate the duration of the signal
    duration = n_samples / sampling_rate

    # Check if the requested noise frequency is feasible given the signal duration
    if (1 / noise_frequency) > duration:
        print(
            f"Skipping requested noise frequency of {noise_frequency} Hz since its period of {1 / noise_frequency} "
            f"seconds exceeds the signal duration of {duration} seconds. Please choose noise frequencies larger than "
            f"{1 / duration} Hz or increase the duration of the signal above {1 / noise_frequency} seconds."
        )

    # Calculate the duration of the noise in samples
    noise_duration = int(duration * noise_frequency)

    # Generate noise based on the specified shape and amplitude
    _noise = np.random.normal(0, noise_amplitude * np.std(signal), noise_duration)

    # Adjust the length of the noise array to match the specified number of samples
    if len(_noise) != n_samples:
        _noise = scipy.ndimage.zoom(_noise, n_samples / len(_noise))

    # Add the generated noise to the input signal
    noisy_signal = signal + _noise

    # If requested, plot the original and noisy signals
    if show:
        plot_noise_signal(signal, noisy_signal, f'Add Noise of {noise_frequency} Hz')

    return noisy_signal


# ==============================================================================
# ------------------------------------Noise-------------------------------------
# ==============================================================================

def standize_1D(signal):
    return (signal - signal.mean()) / signal.std()

def emd_decomposition(signal, show=False):
    """
    Perform Empirical Mode Decomposition (EMD) on a 1D signal.

    Parameters:
    signal : array-like
        The input signal to be decomposed using EMD.
    show : bool, optional
        Whether to display a plot of the decomposed components.

    Returns:
    imfs : list
        A list of Intrinsic Mode Functions (IMFs) obtained from EMD decomposition.
    """
    # Standardize the input signal
    signal = standize_1D(signal)

    # Create an instance of the EMD class
    emd = EMD()

    # Perform EMD decomposition to obtain IMFs
    imfs = emd(signal)

    if show:
        plot_decomposed_components(signal, imfs, 'EMD')

    return imfs

def eemd_decomposition(signal, noise_width=0.05, ensemble_size=100, show=False):
    """
    Perform Ensemble Empirical Mode Decomposition (EEMD) on a 1D signal.

    Parameters:
    signal : array-like
        The input signal to be decomposed using EEMD.
    noise_width : float, optional
        Width of the white noise to add to the signal for EEMD ensemble generation.
    ensemble_size : int, optional
        Number of ensemble trials to perform EEMD.
    show : bool, optional
        Whether to display a plot of the decomposed components.

    Returns:
    imfs : list
        A list of Intrinsic Mode Functions (IMFs) obtained from EEMD decomposition.
    """
    # Standardize the input signal
    signal = standize_1D(signal)

    # Create an instance of the EEMD class with specified ensemble parameters
    eemd = EEMD(trials=ensemble_size, noise_width=noise_width)

    # Perform EEMD decomposition to obtain IMFs
    imfs = eemd.eemd(signal)

    if show:
        plot_decomposed_components(signal, imfs, 'EEMD')

    return imfs

def ceemd_decomposition(signal, show=False):
    """
    Perform Complete Ensemble Empirical Mode Decomposition with Adaptive Noise (CEEMDAN) on a 1D signal.

    Parameters:
    signal : array-like
        The input signal to be decomposed using CEEMDAN.
    show : bool, optional
        Whether to display a plot of the decomposed components.

    Returns:
    imfs : list
        A list of Intrinsic Mode Functions (IMFs) obtained from CEEMDAN decomposition.
    """
    # Preprocess the input signal (e.g., standardize or denoise if necessary)
    signal = standize_1D(signal)

    # Create an instance of the CEEMDAN class
    ceemdan = CEEMDAN()

    # Perform CEEMDAN decomposition on the preprocessed signal to obtain IMFs
    imfs = ceemdan.ceemdan(signal)

    if show:
        plot_decomposed_components(signal, imfs, 'CEEMDAN')

    # Return the resulting IMFs
    return imfs


def vmd_decomposition(signal, K=5, alpha=2000, tau=0, DC=0, init=1, tol=1e-7, show=False):
    """
    Perform Variational Mode Decomposition (VMD) on a 1D signal.

    Parameters:
    signal : array-like
        The input signal to be decomposed using VMD.
    K : int, optional
        Number of modes to decompose the signal into.
    alpha : float, optional
        Moderate bandwidth constraint for VMD.
    tau : float, optional
        Noise-tolerance parameter (no strict fidelity enforcement).
    DC : int, optional
        Whether to include a DC (direct current) part in the decomposition.
    init : int, optional
        Initialization parameter (1 for uniform initialization of omegas).
    tol : float, optional
        Tolerance parameter.

    Returns:
    u : array-like
        An array containing the decomposed modes obtained from VMD decomposition.
    """
    # Standardize the input signal
    signal = standize_1D(signal)

    # Create an instance of the VMD class with specified parameters
    vmd = VMD(signal, alpha, tau, K, DC, init, tol)

    # Perform VMD decomposition to obtain the modes
    u, _, _ = vmd

    if show:
        plot_decomposed_components(signal, u, 'VMD')

    return u

def seasonal_decomposition(signal, period=100, model=0, show=False):
    """
    Perform seasonal decomposition on a time series signal.

    Parameters:
    signal : array-like
        The input time series signal to be decomposed.
    period : int, optional
        The period of the seasonal component.
    model : int, optional
        Model type for decomposition (0 for "additive", 1 for "multiplicative").
    show : bool, optional
        Whether to display a plot of the decomposed components.

    Returns:
    components : object
        An object containing the decomposed components (seasonal, trend, resid).
    """
    # Standardize the input signal
    signal = standize_1D(signal)

    # Determine the decomposition model type
    stl_model = None
    if model == 0:
        stl_model = "additive"
    elif model == 1:
        stl_model = "multiplicative"

    # Perform seasonal decomposition
    components = seasonal_decompose(signal, model=stl_model, period=period)

    if show:
        plt.subplots(4, 1)

        plt.subplot(4, 1, 1)
        plt.plot(signal, label='Original Signal', color='r')
        plt.title("Seasonal Decomposition")
        plt.legend()

        plt.subplot(4, 1, 2)
        plt.plot(components.trend, label='Trend')
        plt.legend()

        plt.subplot(4, 1, 3)
        plt.plot(components.seasonal, label='Seasonal')
        plt.legend()

        plt.subplot(4, 1, 4)
        plt.plot(components.resid, label='Residual')
        plt.legend()
        plt.show()

    return components


class SSA(object):
    __supported_types = (pd.Series, np.ndarray, list)

    def __init__(self, tseries, L, save_mem=True):
        """
        Decomposes the given time series with a singular-spectrum analysis. Assumes the values of the time series are
        recorded at equal intervals.

        Parameters
        ----------
        tseries : The original time series, in the form of a Pandas Series, NumPy array or list.
        L : The window length. Must be an integer 2 <= L <= N/2, where N is the length of the time series.
        save_mem : Conserve memory by not retaining the elementary matrices. Recommended for long time series with
            thousands of values. Defaults to True.

        Note: Even if an NumPy array or list is used for the initial time series, all time series returned will be
        in the form of a Pandas Series or DataFrame object.
        """

        # Tedious type-checking for the initial time series
        if not isinstance(tseries, self.__supported_types):
            raise TypeError("Unsupported time series object. Try Pandas Series, NumPy array or list.")

        # Checks to save us from ourselves
        self.N = len(tseries)
        if not 2 <= L <= self.N / 2:
            raise ValueError("The window length must be in the interval [2, N/2].")

        self.L = L
        self.orig_TS = pd.Series(tseries)
        self.K = self.N - self.L + 1

        # Embed the time series in a trajectory matrix
        self.X = np.array([self.orig_TS.values[i:L + i] for i in range(0, self.K)]).T

        # Decompose the trajectory matrix
        self.U, self.Sigma, VT = np.linalg.svd(self.X)
        self.d = np.linalg.matrix_rank(self.X)

        self.TS_comps = np.zeros((self.N, self.d))

        if not save_mem:
            # Construct and save all the elementary matrices
            self.X_elem = np.array([self.Sigma[i] * np.outer(self.U[:, i], VT[i, :]) for i in range(self.d)])

            # Diagonally average the elementary matrices, store them as columns in array.
            for i in range(self.d):
                X_rev = self.X_elem[i, ::-1]
                self.TS_comps[:, i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0] + 1, X_rev.shape[1])]

            self.V = VT.T
        else:
            # Reconstruct the elementary matrices without storing them
            for i in range(self.d):
                X_elem = self.Sigma[i] * np.outer(self.U[:, i], VT[i, :])
                X_rev = X_elem[::-1]
                self.TS_comps[:, i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0] + 1, X_rev.shape[1])]

            self.X_elem = "Re-run with save_mem=False to retain the elementary matrices."

            # The V array may also be very large under these circumstances, so we won't keep it.
            self.V = "Re-run with save_mem=False to retain the V matrix."

        # Calculate the w-correlation matrix.
        self.calc_wcorr()

    def components_to_df(self, n=0):
        """
        Returns all the time series components in a single Pandas DataFrame object.
        """
        if n > 0:
            n = min(n, self.d)
        else:
            n = self.d

        # Create list of columns - call them F0, F1, F2, ...
        cols = ["F{}".format(i) for i in range(n)]
        return pd.DataFrame(self.TS_comps[:, :n], columns=cols, index=self.orig_TS.index)

    def reconstruct(self, indices):
        """
        Reconstructs the time series from its elementary components, using the given indices. Returns a Pandas Series
        object with the reconstructed time series.

        Parameters
        ----------
        indices: An integer, list of integers or slice(n,m) object, representing the elementary components to sum.
        """
        if isinstance(indices, int): indices = [indices]

        ts_vals = self.TS_comps[:, indices].sum(axis=1)
        return pd.Series(ts_vals, index=self.orig_TS.index)

    def calc_wcorr(self):
        """
        Calculates the w-correlation matrix for the time series.
        """

        # Calculate the weights
        w = np.array(list(np.arange(self.L) + 1) + [self.L] * (self.K - self.L - 1) + list(np.arange(self.L) + 1)[::-1])

        def w_inner(F_i, F_j):
            return w.dot(F_i * F_j)

        # Calculated weighted norms, ||F_i||_w, then invert.
        F_wnorms = np.array([w_inner(self.TS_comps[:, i], self.TS_comps[:, i]) for i in range(self.d)])
        F_wnorms = F_wnorms ** -0.5

        # Calculate Wcorr.
        self.Wcorr = np.identity(self.d)
        for i in range(self.d):
            for j in range(i + 1, self.d):
                self.Wcorr[i, j] = abs(w_inner(self.TS_comps[:, i], self.TS_comps[:, j]) * F_wnorms[i] * F_wnorms[j])
                self.Wcorr[j, i] = self.Wcorr[i, j]

    def plot_wcorr(self, min=None, max=None):
        """
        Plots the w-correlation matrix for the decomposed time series.
        """
        if min is None:
            min = 0
        if max is None:
            max = self.d

        if self.Wcorr is None:
            self.calc_wcorr()

        ax = plt.imshow(self.Wcorr)
        plt.xlabel(r"$\tilde{F}_i$")
        plt.ylabel(r"$\tilde{F}_j$")
        plt.colorbar(ax.colorbar, fraction=0.045)
        ax.colorbar.set_label("$W_{i,j}$")
        plt.clim(0, 1)

        # For plotting purposes:
        if max == self.d:
            max_rnge = self.d - 1
        else:
            max_rnge = max

        plt.xlim(min - 0.5, max_rnge + 0.5)
        plt.ylim(max_rnge + 0.5, min - 0.5)

# ==============================================================================
# ------------------------------------Filter-------------------------------------
# ==============================================================================


def butter_bandpass_filter(signal, lowcut=1, highcut=10, fs=100, order=5, show=False):
    """
    Apply a bandpass Butterworth filter to the input signal.

    Parameters:
    signal : array-like
        The input signal to be filtered.
    lowcut : float, optional
        The low cutoff frequency of the bandpass filter.
    highcut : float, optional
        The high cutoff frequency of the bandpass filter.
    fs : float, optional
        The sampling frequency of the input signal.
    order : int, optional
        The filter order.

    Returns:
    filtered_signal : array-like
        The signal after applying the bandpass filter.
    """
    b, a = butter(order, [lowcut, highcut], btype='bandpass', analog=False, output='ba', fs=fs)
    filtered_signal = lfilter(b, a, signal)
    if show:
        plot_filtered_signal(filtered_signal, signal, "Bandpass Filter")
    return filtered_signal

def butter_bandstop_filter(signal, lowcut=1, highcut=10, fs=100, order=5, show=False):
    """
    Apply a bandstop Butterworth filter to the input signal.

    Parameters:
    signal : array-like
        The input signal to be filtered.
    lowcut : float, optional
        The low cutoff frequency of the bandstop filter.
    highcut : float, optional
        The high cutoff frequency of the bandstop filter.
    fs : float, optional
        The sampling frequency of the input signal.
    order : int, optional
        The filter order.

    Returns:
    filtered_signal : array-like
        The signal after applying the bandstop filter.
    """
    b, a = butter(order, [lowcut, highcut], btype='bandstop', analog=False, output='ba', fs=fs)
    filtered_signal = lfilter(b, a, signal)
    if show:
        plot_filtered_signal(filtered_signal, signal, "Bandstop Filter")
    return filtered_signal

def butter_lowpass_filter(signal, cutoff=10, fs=100, order=5, show=False):
    """
    Apply a lowpass Butterworth filter to the input signal.

    Parameters:
    signal : array-like
        The input signal to be filtered.
    cutoff : float, optional
        The cutoff frequency of the lowpass filter.
    fs : float, optional
        The sampling frequency of the input signal.
    order : int, optional
        The filter order.

    Returns:
    filtered_signal : array-like
        The signal after applying the lowpass filter.
    """
    b, a = butter(order, cutoff, btype='lowpass', analog=False, output='ba', fs=fs)
    filtered_signal = lfilter(b, a, signal)

    if show:
        plot_filtered_signal(filtered_signal, signal, "Lowpass Filter")

    return filtered_signal

def butter_highpass_filter(signal, cutoff=10, fs=100, order=5, show=False):
    """
    Apply a highpass Butterworth filter to the input signal.

    Parameters:
    signal : array-like
        The input signal to be filtered.
    cutoff : float, optional
        The cutoff frequency of the highpass filter.
    fs : float, optional
        The sampling frequency of the input signal.
    order : int, optional
        The filter order.

    Returns:
    filtered_signal : array-like
        The signal after applying the highpass filter.
    """
    b, a = butter(order, cutoff, btype='highpass', analog=False, output='ba', fs=fs)
    filtered_signal = lfilter(b, a, signal)

    if show:
        plot_filtered_signal(filtered_signal, signal, "Highpass Filter")

    return filtered_signal

def simple_moving_average_filter(signal, length=10, show=False):
    """
    Apply a Simple Moving Average (SMA) filter to smooth the input signal.

    Parameters:
    signal : array-like
        The input signal array to be filtered.
    length : int, optional
        Length of the moving average window.

    Returns:
    filtered_y : array-like
        The smoothed signal after applying the SMA filter.
    """
    # Create the Simple Moving Average weight array
    SMA = [1 / length] * length
    # Use convolution operation to filter the signal, 'same' option ensures output length matches input
    filtered_signal = np.convolve(signal, SMA, 'same')

    if show:
        plot_filtered_signal(filtered_signal, signal, "Simple Moving Average Filter")

    return filtered_signal

def exponential_moving_average_filter(signal, length=10, alpha=None, show=False):
    """
    Apply an Exponential Moving Average (EMA) filter to smooth the input signal.

    Parameters:
    signal : array-like
        The input signal array to be filtered.
    length : int, optional
        Length of the moving average window.
    alpha : float, optional
        Smoothing factor (if not provided, uses default value).

    Returns:
    filtered_y : array-like
        The smoothed signal after applying the EMA filter.
    """
    # If alpha is not provided, use the default value
    if alpha is None:
        alpha = 2 / (length + 1)

    # Create the Exponential Moving Average weight array
    u = np.ones(length)
    n = np.arange(length)
    EMA = alpha * (1 - alpha) ** n * u
    # Use convolution operation to filter the signal, 'same' option ensures output length matches input
    filtered_signal = np.convolve(signal, EMA, 'same')

    if show:
        plot_filtered_signal(filtered_signal, signal, "Exponential Moving Average Filter")

    return filtered_signal

def savgol_filter(signal, window_length=32, polyorder=1, show=False):
    """
    Apply a Savitzky-Golay filter to the input signal for smoothing.

    Parameters:
    signal : array-like
        The input signal array to be filtered.
    window_length : int, optional
        The length of the smoothing window.
    polyorder : int, optional
        The order of the polynomial used for fitting the data.
    show : bool, optional
        Flag to show any plots or visualization (not implemented in this function).

    Returns:
    filtered_signal : array-like
        The smoothed signal after applying the Savitzky-Golay filter.
    """
    filtered_signal = scipy.signal.savgol_filter(signal, window_length, polyorder)

    if show:
        plot_filtered_signal(filtered_signal, signal, "Savitzky-Golay Filter")

    return filtered_signal

def wiener_filter(signal, noise, show=False):
    """
    Apply a Wiener filter to the input signal for noise reduction.

    Parameters:
    signal : array-like
        The noisy input signal array.
    noise : array-like
        The noise signal that needs to be removed from the input signal.
    show : bool, optional
        Flag to show any plots or visualization (not implemented in this function).

    Returns:
    filtered_signal : array-like
        The signal after applying the Wiener filter for noise reduction.
    """
    # Calculate the clean signal by subtracting the noise
    clean_signal = signal - noise

    # Calculate the power spectrum of the clean signal and the noise
    signal_power = np.abs(np.fft.fft(clean_signal))**2
    noise_power = np.abs(np.fft.fft(noise))**2

    # Estimate the noise power as the mean of the noise power spectrum
    noise_power = np.mean(noise_power)

    # Calculate the signal-to-noise ratio (SNR)
    snr = signal_power / noise_power

    # Apply the Wiener filter to the frequency domain
    wiener_ = 1 / (1 + 1 / snr)
    filtered_signal = np.fft.fft(signal) * wiener_
    filtered_signal = np.fft.ifft(filtered_signal)

    if show:
        plot_filtered_signal(filtered_signal, signal, "Wiener Filter")

    return filtered_signal


def rls_filter(x, d, n, mu, show=False):
    """
    Apply Recursive Least Squares (RLS) filter to input signal x to estimate a desired signal d.

    Parameters:
    - x: Input signal.
    - d: Desired signal to be estimated.
    - n: Order of the filter.
    - mu: Convergence factor.

    Returns:
    - y: Output signal (estimated signal).
    - e: Error signal (difference between estimated and desired signals).
    - w: Filter weights after processing the signals.
    """
    x_np = np.array(x)
    d_np = np.array(d)

    # Ensure x and d are 2D arrays
    if x_np.ndim == 1:
        x_np = x_np.reshape(-1, 1)
    if d_np.ndim == 1:
        d_np = d_np.reshape(-1, 1)

    # Create an RLS filter with specified parameters
    f = pa.filters.FilterRLS(n=n, mu=mu, w="random")

    # Run the RLS filter on the input and desired signals
    y, e, w = f.run(d_np, x_np)

    if show:
        plot_filtered_signal(y, x, "Recursive Least Squares (RLS) Filter")

    return y, e, w


def lms_filter(x, d, n, mu, show=False):
    """
    Apply Least Mean Squares (LMS) filter to input signal x to estimate a desired signal d.

    Parameters:
    - x: Input signal.
    - d: Desired signal to be estimated.
    - n: Order of the filter.
    - mu: Convergence factor.

    Returns:
    - y: Output signal (estimated signal).
    - e: Error signal (difference between estimated and desired signals).
    - w: Filter weights after processing the signals.
    """
    x_np = np.array(x)
    d_np = np.array(d)

    # Ensure x and d are 2D arrays
    if x_np.ndim == 1:
        x_np = x_np.reshape(-1, 1)
    if d_np.ndim == 1:
        d_np = d_np.reshape(-1, 1)

    # Create an LMS filter with specified parameters
    f = pa.filters.FilterLMS(n=n, mu=mu, w="random")

    # Run the LMS filter on the input and desired signals
    y, e, w = f.run(d_np, x_np)

    if show:
        plot_filtered_signal(y, x, "Least Mean Squares (LMS) Filter")

    return y, e, w

def notch_filter(signal, cutoff=10, q=10, fs=100, show=False):
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

    if show:
        plot_filtered_signal(filtered_signal, signal, "Notch Filter")

    return filtered_signal

def matched_filter(signal, template, show=False):
    """
    Apply matched filter to a signal using a template.

    Parameters:
    - signal: The input signal.
    - template: The template signal.

    Returns:
    - filtered_output: The output of the matched filter.
    """
    # Ensure inputs are numpy arrays
    signal = np.array(signal)
    template = np.array(template)

    # Reverse the template signal
    template = np.flip(template)

    # Perform convolution using numpy's convolve function
    # filtered_signal = np.convolve(signal, template, mode='full')
    filtered_signal = lfilter(template, 1, signal)

    if show:
        plt.figure()
        plt.plot(filtered_signal, label='Filtered Signal')
        plt.title("Matched Filter")
        plt.legend()
        plt.show()

    return filtered_signal


def fft_denoise(signal, threshold, show=False):
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

    if show:
        plot_filtered_signal(ffilt, signal, "FFT Denoising")

    return ffilt


def wavelet_denoise(data, method, threshold, show=False):
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

    if show:
        plot_filtered_signal(datarec, data, "Wavelet Denoising")

    return datarec


# ==============================================================================
# -------------------------Blind Source Separation------------------------------
# ==============================================================================

from sklearn.decomposition import FastICA, PCA

def bss_ica(X, n_components):
    """
    Apply Independent Component Analysis (ICA) to the input data.

    Parameters:
    X (array-like): Input data matrix with shape (n_samples, n_features).
    n_components (int): Number of independent components to extract.

    Returns:
    S_ (array-like): Reconstructed source signals.
    A_ (array-like): Estimated mixing matrix.
    """
    ica = FastICA(n_components=n_components)

    # Apply ICA to the input data to extract independent components
    S_ = ica.fit_transform(X)  # Reconstruct signals

    A_ = ica.mixing_  # Get estimated mixing matrix

    # Verify the ICA model by checking if the original data can be reconstructed
    # using the estimated mixing matrix and the extracted sources
    assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

    return S_, A_


def bss_pca(X, n_components):
    """
    Apply Principal Component Analysis (PCA) to the input data.

    Parameters:
    X (array-like): Input data matrix with shape (n_samples, n_features).
    n_components (int): Number of principal components to retain.

    Returns:
    transformed_X (array-like): Data projected onto the first n_components principal components.
    """
    pca = PCA(n_components=n_components)

    # Apply PCA to the input data to extract orthogonal components
    transformed_X = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components

    return transformed_X


# ==============================================================================
# -------------------------------------DTW--------------------------------------
# ==============================================================================

from tslearn.barycenters import softdtw_barycenter
from dsp_utils import plot_averaging_center
from scipy.interpolate import CubicSpline
import random

def _traceback(D):
    i, j = np.array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = np.argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return np.array(p), np.array(q)


def dtw_easy(x, y, dist, warp=1, s=1.0):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.

    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    :param int warp: how many shifts are computed.
    :param int w: window size limiting the maximal distance between indices of matched entries |i,j|.
    :param float s: weight applied on off-diagonal moves of the path. As s gets larger, the warping path is increasingly biased towards the diagonal
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    assert s > 0
    r, c = len(x), len(y)

    D0 = np.zeros((r + 1, c + 1))
    D0[0, 1:] = np.inf
    D0[1:, 0] = np.inf

    D1 = D0[1:, 1:]  # view

    for i in range(r):
        for j in range(c):
            D1[i, j] = dist(x[i], y[j])
    C = D1.copy()

    jrange = range(c)
    for i in range(r):
        for j in jrange:
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                i_k = min(i + k, r)
                j_k = min(j + k, c)
                min_list += [D0[i_k, j] * s, D0[i, j_k] * s]
            D1[i, j] += min(min_list)
    if len(x) == 1:
        path = np.zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), np.zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1], C, D1, path

def performSOFTDBA(pieces, show=False):
    """
    Perform Soft-DTW Barycenter Averaging (SOFTDBA) on a list of time series pieces.

    Parameters:
    - pieces (list): List of time series pieces (2D arrays) to be averaged.

    Returns:
    - np.ndarray: Soft-DTW barycenter averaged time series.

    Notes:
    The Soft-DTW Barycenter Averaging is a method for computing a representative time series,
    also known as the barycenter, that minimizes the soft DTW distance to a set of input time series.

    Reference:
    Cuturi, Marco, and Mathieu Blondel. "Soft-dtw: a differentiable loss function for time-series." International conference on machine learning. PMLR, 2017.
    """
    center = softdtw_barycenter(pieces)
    if show:
        plot_averaging_center(center, pieces)
    return center



def performICDTW(pieces, iter_max=10, dist=lambda x, y: np.abs(x - y), Beta_A1=1e-5, Beta_A2=1e5, Beta_B=1, show=False):
    """
    Perform Iterative Constrained Dynamic Time Warping (ICDTW) on a list of time series pieces.

    Parameters:
    - pieces (list): List of time series pieces (1D arrays) to be averaged.
    - iter_max (int): Maximum number of iterations for ICDTW. Default is 10.
    - dist (function): Distance function for Dynamic Time Warping (DTW). Default is absolute difference.
    - Beta_A1, Beta_A2, Beta_B (float): Parameters for ICDTW optimization. Default values provided.

    Returns:
    - np.ndarray: Averaged time series obtained through ICDTW.

    Raises:
    - TypeError: If the input 'pieces' is not a list.
    """

    def make_template(piece1, piece2, path, w1, w2):
        """
        Create a template by interpolating between two time series pieces based on the DTW path.
        Reference Paper:
            Niennattrakul, Vit, Dararat Srisai, and Chotirat Ann Ratanamahatana. "Shape-based template matching for time series data." Knowledge-Based Systems 26 (2012): 1-8.

        Parameters:
        - piece1, piece2 (np.ndarray): Time series pieces to interpolate between.
        - path (tuple): DTW path between the two pieces.
        - w1, w2 (float): Weights for the interpolation.

        Returns:
        - np.ndarray, np.ndarray: Interpolated x and y values of the template.
        """
        path1 = path[0]
        path2 = path[1]
        x, y = [], []

        for x_1, x_2 in zip(path1, path2):
            x_new = (w1 * x_1 + w2 * x_2) / (w1 + w2)
            y_new = (piece1[x_1] * w1 + piece2[x_2] * w2) / (w1 + w2)
            x.append(x_new)
            y.append(y_new)

        return np.array(x), np.array(y)

    def cdtw_averaging(piece1, piece2, w1, w2):
        """
        Perform Constrained DTW (CDTW) averaging between two time series pieces.

        Parameters:
        - piece1, piece2 (np.ndarray): Time series pieces to average.
        - w1, w2 (float): Weights for the averaging.

        Returns:
        - np.ndarray: Averaged time series obtained through CDTW.
        """
        dist = lambda x, y: np.abs(x - y)
        _, _, _, path = dtw_easy(piece1, piece2, dist)

        N_ = min(len(piece1), len(piece2))
        x, y = make_template(piece1, piece2, path, w1, w2)

        # 创建CubicSpline对象
        cs = CubicSpline(x, y)
        new_x = np.linspace(0, N_, N_, endpoint=False)
        new_y = cs(new_x)

        return new_y

    def icdtw_averaging(A, B, w_A, w_B, iter_max, dist, Beta_A1, Beta_A2, Beta_B):
        """
        Perform Iterative Constrained Dynamic Time Warping (ICDTW) averaging between two time series pieces.

        Parameters:
        - A, B (np.ndarray): Time series pieces to average.
        - w_A, w_B (float): Weights for the averaging.
        - iter_max (int): Maximum number of iterations for ICDTW.
        - dist (function): Distance function for DTW.
        - Beta_A1, Beta_A2, Beta_B (float): Parameters for ICDTW optimization.

        Returns:
        - np.ndarray: Averaged time series obtained through ICDTW.
        """

        iter_n = 0
        C = None
        while iter_n < iter_max:
            iter_n += 1
            # print(np.abs(dis_CA - dis_CB))
            Beta_A3 = 0.5 * (Beta_A1 + Beta_A2)
            C = cdtw_averaging(A, B, Beta_A3, Beta_B)
            CA = dtw_easy(C, A, dist) * w_A
            dis_CA = CA[0]
            # dis_CA, _, _, _ = dtw_easy(C, A, dist) * w_A

            CB = dtw_easy(C, B, dist) * w_B
            dis_CB = CB[0]
            # dis_CB, _, _, _ = dtw_easy(C, B, dist) * w_B
            if dis_CA < dis_CB:
                Beta_A1 = Beta_A3
            else:
                Beta_A2 = Beta_A3

        return C

    # Check if 'pieces' is a list
    if not isinstance(pieces, list):
        raise TypeError("Variable 'pieces' must be a list.")

    original_pieces = pieces

    # Initialize weights for each piece
    weights = [1] * len(pieces)

    # Perform ICDTW until only one piece remains
    while len(pieces) > 1:
        A, B = pieces[0], pieces[1]

        w_A, w_B = weights[0], weights[1]
        C = icdtw_averaging(A, B, w_A, w_B, iter_max, dist, Beta_A1, Beta_A2, Beta_B)
        w_C = w_A + w_B
        pieces.append(C)
        weights.append(w_C)

        pieces = pieces[2:]
        weights = weights[2:]

    if show:
        plot_averaging_center(pieces[0], original_pieces)

    return pieces[0]

def performDBA(series, n_iterations=10, show=False):
    """ author is Francois Petitjean
        References:
            Petitjean, François, Alain Ketterlin, and Pierre Gançarski.
            "A global averaging method for dynamic time war**, with applications to clustering."
            Pattern recognition 44.3 (2011): 678-693.
    """
    _initial_missing = object()
    def reduce(function, sequence, initial=_initial_missing):
        it = iter(sequence)
        if initial is _initial_missing:
            try:
                value = next(it)
            except StopIteration:
                raise TypeError("reduce() of empty sequence with no initial value") from None
        else:
            value = initial
        for element in it:
            value = function(value, element)
        return value

    def approximate_medoid_index(series, cost_mat, delta_mat):
        if len(series) <= 50:
            indices = range(0, len(series))
        else:
            indices = np.random.choice(range(0, len(series)), 50, replace=False)

        medoid_ind = -1
        best_ss = 1e20
        for index_candidate in indices:
            candidate = series[index_candidate]
            ss = sum_of_squares(candidate, series, cost_mat, delta_mat)
            if (medoid_ind == -1 or ss < best_ss):
                best_ss = ss
                medoid_ind = index_candidate
        return medoid_ind

    def sum_of_squares(s, series, cost_mat, delta_mat):
        return sum(map(lambda t: squared_DTW(s, t, cost_mat, delta_mat), series))

    def squared_DTW(s, t, cost_mat, delta_mat):
        s_len = len(s)
        t_len = len(t)
        length = len(s)
        fill_delta_mat_dtw(s, t, delta_mat)
        cost_mat[0, 0] = delta_mat[0, 0]
        for i in range(1, s_len):
            cost_mat[i, 0] = cost_mat[i - 1, 0] + delta_mat[i, 0]

        for j in range(1, t_len):
            cost_mat[0, j] = cost_mat[0, j - 1] + delta_mat[0, j]

        for i in range(1, s_len):
            for j in range(1, t_len):
                diag, left, top = cost_mat[i - 1, j - 1], cost_mat[i, j - 1], cost_mat[i - 1, j]
                if (diag <= left):
                    if (diag <= top):
                        res = diag
                    else:
                        res = top
                else:
                    if (left <= top):
                        res = left
                    else:
                        res = top
                cost_mat[i, j] = res + delta_mat[i, j]
        return cost_mat[s_len - 1, t_len - 1]

    def fill_delta_mat_dtw(center, s, delta_mat):
        slim = delta_mat[:len(center), :len(s)]
        np.subtract.outer(center, s, out=slim)
        np.square(slim, out=slim)

    def DBA_update(center, series, cost_mat, path_mat, delta_mat):
        options_argmin = [(-1, -1), (0, -1), (-1, 0)]
        updated_center = np.zeros(center.shape)
        n_elements = np.array(np.zeros(center.shape), dtype=int)
        center_length = len(center)

        for s in series:
            s_len = len(s)
            fill_delta_mat_dtw(center, s, delta_mat)
            cost_mat[0, 0] = delta_mat[0, 0]
            path_mat[0, 0] = -1

            for i in range(1, center_length):
                cost_mat[i, 0] = cost_mat[i - 1, 0] + delta_mat[i, 0]
                path_mat[i, 0] = 2

            for j in range(1, s_len):
                cost_mat[0, j] = cost_mat[0, j - 1] + delta_mat[0, j]
                path_mat[0, j] = 1

            for i in range(1, center_length):
                for j in range(1, s_len):
                    diag, left, top = cost_mat[i - 1, j - 1], cost_mat[i, j - 1], cost_mat[i - 1, j]
                    if (diag <= left):
                        if (diag <= top):
                            res = diag
                            path_mat[i, j] = 0
                        else:
                            res = top
                            path_mat[i, j] = 2
                    else:
                        if (left <= top):
                            res = left
                            path_mat[i, j] = 1
                        else:
                            res = top
                            path_mat[i, j] = 2

                    cost_mat[i, j] = res + delta_mat[i, j]

            i = center_length - 1
            j = s_len - 1

            while (path_mat[i, j] != -1):
                updated_center[i] += s[j]
                n_elements[i] += 1
                move = options_argmin[path_mat[i, j]]
                i += move[0]
                j += move[1]
            assert (i == 0 and j == 0)
            updated_center[i] += s[j]
            n_elements[i] += 1

        return np.divide(updated_center, n_elements)

    max_length = reduce(max, map(len, series))

    cost_mat = np.zeros((max_length, max_length))
    delta_mat = np.zeros((max_length, max_length))
    path_mat = np.zeros((max_length, max_length), dtype=np.int8)

    medoid_ind = approximate_medoid_index(series,cost_mat,delta_mat)
    center = series[medoid_ind]

    for i in range(0,n_iterations):
        center = DBA_update(center, series, cost_mat, path_mat, delta_mat)

    if show:
        plot_averaging_center(center, series)

    return center


def make_template(piece0, piece1, path):
    """
    Create a template by averaging aligned segments of two time series pieces based on a given DTW path.

    Parameters:
    - piece0, piece1 (np.ndarray): Time series pieces to interpolate between.
    - path (tuple): DTW path between the two pieces.

    Returns:
    - np.ndarray: Averaged template based on the DTW path.
    """
    path0 = path[0]
    path1 = path[1]
    new_piece0 = np.array([piece0[idx] for idx in path0])
    new_piece1 = np.array([piece1[idx] for idx in path1])

    template = 0.5 * (new_piece0 + new_piece1)
    return template


def performNLAAF1(pieces, dist=lambda x, y: np.abs(x - y), show=False):
    """
    Perform Non-Linear Adaptive Averaging Filter 1 (NLAAF1) on a list of time series pieces.

    Parameters:
    - pieces (list): List of time series pieces (1D arrays) to be fused.
    - dist (function): Distance function for Dynamic Time Warping (DTW). Default is absolute difference.

    Returns:
    - np.ndarray: Fused time series obtained through NLAAF1.
    """

    # 2^N
    # 1. drop out something
    # 2. use loops to simulate recursive
    # 3. get the template

    pieces_num = len(pieces)
    k = 1
    while pieces_num >= k:
        k *= 2
    k = int(k / 2)

    random_choice = random.sample(range(pieces_num), k)
    chosen_pieces = [pieces[choice] for choice in random_choice]

    this_term = chosen_pieces

    while k > 1:
        last_term = this_term
        this_term = []

        for cnt in range(0, k, 2):
            a = cnt
            b = cnt + 1
            piece1, piece2 = last_term[a], last_term[b]

            _, _, _, path = dtw_easy(piece1, piece2, dist)
            template = make_template(piece1, piece2, path)
            this_term.append(template)

        k = int(k / 2)

    center = np.array(this_term[0])

    if show:
        plot_averaging_center(center, pieces)

    return center


def performNLAAF2(pieces, dist=lambda x, y: np.abs(x - y), show=False):
    """
    Perform  Non-Linear Adaptive Averaging Filter 2 (NLAAF2) on a list of time series pieces.

    Parameters:
    - pieces (list): List of time series pieces (1D arrays) to be fused.
    - dist (function): Distance function for Dynamic Time Warping (DTW). Default is absolute difference.

    Returns:
    - np.ndarray: Fused time series obtained through NLAAF2.
    """
    # one by one
    # 1. use loops to calculate
    # 2. get the template

    pieces_num = len(pieces)
    _, _, _, path = dtw_easy(pieces[0], pieces[1], dist)
    template = make_template(pieces[0], pieces[1], path)

    for cnt in range(2, pieces_num):
        _, _, _, path = dtw_easy(template, pieces[cnt], dist)
        template = make_template(template, pieces[cnt], path)

    if show:
        plot_averaging_center(template, pieces)

    return template
