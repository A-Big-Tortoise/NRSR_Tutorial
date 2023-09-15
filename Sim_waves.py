import numpy as np
from utils import plot_sim_waves

def sine_wave(duration=10, sampling_rate=100, amplitude=1, frequency=1, phase=0, show=False):
    """
    Generate a sine wave signal.

    Parameters:
    time : array-like
        The time values at which the signal is evaluated.
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
    time = np.linspace(0, duration, duration * sampling_rate)
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * time + phase)

    if show:
        plot_sim_waves(sine_wave, 'Sine Wave')

    return sine_wave

def triangle_wave(duration=10, sampling_rate=100, amplitude=1, period=2, show=False):
    """
    Generate a triangle wave signal.

    Parameters:
    time : array-like
        The time values at which the signal is evaluated.
    Amplitude : float
        The amplitude of the triangle wave.
    period : float
        The period of the triangle wave.

    Returns:
    triangle_wave : array-like
        An array containing the values of the triangle wave signal at the given time points.
    """

    time = np.linspace(0, duration, duration * sampling_rate)
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
    time : array-like
        The time values at which the signal is evaluated.
    frequency : float
        The frequency of the square wave.

    Returns:
    square_wave : array-like
        An array containing the values of the square wave signal at the given time points.
    """
    time = np.linspace(0, duration, duration * sampling_rate)
    square_wave = 2 * (2 * np.floor(frequency * time) - np.floor(2 * frequency * time)) + 1

    if show:
        plot_sim_waves(square_wave, 'Square Wave')

    return square_wave


def chirp_wave_linear(duration=10, sampling_rate=100, f0=1, c=1, phase=0, show=False):
    """
    Generate a linear chirp wave signal.

    Parameters:
    time : array-like
        The time values at which the signal is evaluated.
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
    time = np.linspace(0, duration, duration * sampling_rate)
    chirp_wave = np.sin(phase + 2 * np.pi * ((c / 2) * (time ** 2) + f0 * time))

    if show:
        plot_sim_waves(chirp_wave, 'Chirp Wave Linear')

    return chirp_wave

def chirp_wave_exponential(duration=10, sampling_rate=100, f0=1, k=1.2, phase=0, show=False):
    """
    Generate an exponential chirp wave signal.

    Parameters:
    time : array-like
        The time values at which the signal is evaluated.
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
    time = np.linspace(0, duration, duration * sampling_rate)
    chirp_wave = np.sin(phase + 2 * np.pi * f0 * ((k ** time - 1) / np.log(k)))

    if show:
        plot_sim_waves(chirp_wave, 'Chirp Wave Exponential')

    return chirp_wave

def chirp_wave_hyperbolic(duration=10, sampling_rate=100, f0=1, f1=10, phase=0, show=False):
    """
    Generate a hyperbolic chirp wave signal.

    Parameters:
    time : array-like
        The time values at which the signal is evaluated.
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
    time = np.linspace(0, duration, duration * sampling_rate)
    chirp_wave = np.sin(phase + 2 * np.pi * ((-1 * f0 * f1 * time) / (f1 - f0) * np.log(1 - (f1 - f0) / (f1 * time) * t)))

    if show:
        plot_sim_waves(chirp_wave, 'Chirp Wave Hyperbolic')

    return chirp_wave

def pulse_wave(duration=10, sampling_rate=100, amplitude=1, d=0.5, frequency=1, expansion=5, show=False):
    """
    Generate a pulse wave signal.

    Parameters:
    time : array-like
        The time values at which the signal is evaluated.
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
    time = np.linspace(0, duration, duration * sampling_rate)

    sum_of_ = 0
    for n in range(1, expansion+1):
        sum_of_ += np.sinc(n * d) * np.cos(2 * np.pi * n * frequency * time)

    pulse_wave = amplitude * d * (1 + 2 * sum_of_)

    if show:
        plot_sim_waves(pulse_wave, 'Pulse Wave')

    return pulse_wave

