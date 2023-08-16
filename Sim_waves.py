import numpy as np

def sine_wave(time, Amplitude, frequency, phase):
    """
    Generate a sine wave signal.

    Parameters:
    time : array-like
        The time values at which the signal is evaluated.
    Amplitude : float
        The peak deviation of the function from zero.
    frequency : float
        The number of oscillations (cycles) that occur each second of time.
    phase : float
        Phase specifies (in radians) where in its cycle the oscillation is at t = 0.

    Returns:
    sine_wave : array-like
        An array containing the values of the sine wave signal at the given time points.

    """

    return Amplitude * np.sin(2 * np.pi * frequency * time + phase)

def triangle_wave(time, Amplitude, period):
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

    # Scale the time values to a normalized range [-1, 1] within each period
    t_scaled = 1 * np.abs(2 * (time / period - np.floor(time / period + 0.5))) - 1
    
    # Calculate the triangle wave values based on scaled time values
    triangle_wave = (3 * Amplitude / period) * np.abs((t_scaled - period / 4) % period - period / 2) - Amplitude
    
    return triangle_wave

def square_wave(time, frequency):
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
    square_wave = 2 * (2 * np.floor(frequency * time) - np.floor(2 * frequency * time)) + 1
    
    return square_wave


def chirp_wave_linear(time, f0, c, phase):
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
    return np.sin(phase + 2 * np.pi * ((c / 2) * (time ** 2) + f0 * time))

def chirp_wave_exponential(time, f0, k, phase):
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
    return np.sin(phase + 2 * np.pi * f0 * ((k ** time - 1) / np.log(k)))

def chirp_wave_hyperbolic(time, f0, f1, duration, phase):
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
    return np.sin(phase + 2 * np.pi * ((-1 * f0 * f1 * duration) / (f1 - f0) * np.log(1 - (f1 - f0) / (f1 * duration) * time)))

def pulse_wave(time, Amplitude, d, frequency, expansion):
    """
    Generate a pulse wave signal.

    Parameters:
    time : array-like
        The time values at which the signal is evaluated.
    Amplitude : float
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
    sum_of_ = 0
    for n in range(1, expansion+1):
        sum_of_ += np.sinc(n * d) * np.cos(2 * np.pi * n * frequency * time)

    return Amplitude * d * (1 + 2 * sum_of_)

