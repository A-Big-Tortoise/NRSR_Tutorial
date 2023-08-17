import numpy as np
from scipy.signal import butter
from Sim_waves import sine_wave

"""
可能潜藏着的问题：
1. 生成的噪声是否需要考虑noise_freq和nosie_duration
2. 1/f, colored, thermal noise的大小范围应该有什么来控制呢? 手动输入数值还是像powerline一样 由signal_sd来控制?
3. impulse noise的实现是否能再优化一下
4. transient noise究竟该如何实现
"""


def white_noise_nk2(
        signal,  noise_amplitude=0.1, model='gaussian'
        ):
    """maybe need to check the nyquist """
    signal_sd = np.std(signal, ddof=1)
    amp = signal_sd * noise_amplitude

    if model.lower() == 'gaussian':
        _noise = np.random.normal(0, amp, len(signal))
    elif model.lower() == 'laplace':
        _noise = np.random.laplace(0, amp, len(signal))

    return _noise


# 和上面的白噪声一样，没有考虑过noise_freq和noise_duration，可能后期需要大改
def band_limited_white_noise_nk2(
    signal, noise_amplitude, sampling_rate, lowcut, highcut, order=4
    ):
    # Generate white noise
    signal_sd = np.std(signal, ddof=1)
    amp = signal_sd * noise_amplitude
    _noise = np.random.normal(0, amp, len(signal))
    
    # Define bandpass filter parameters
    b, a = butter(order, [lowcut, highcut], btype='band', fs=sampling_rate)
    
    _band_limited_noise = signal.lfilter(b, a, _noise)
    
    return _band_limited_noise

def impulsive_noise(
    signal, noise_amplitude, rate=None, number=None
):
    signal_sd = np.std(signal, ddof=1)
    amp = signal_sd * noise_amplitude
    num_samples = len(signal)

    # rate or number
    if rate is not None and number is None:
        pob = [1 - rate, rate]
    elif rate is None and number is not None:
        pob_rate = number / num_samples
        if pob_rate >= 1.0:
            pob_rate = 1
        pob = [1 - pob_rate, pob_rate]
    else:  
        return None

    impulsive_noise = np.random.choice([0, 1], size=num_samples, p=pob) * np.random.normal(0, amp, num_samples)
    
    return impulsive_noise

def generate_burst_noise(signal, noise_amplitude, burst_num_max, burst_durations=[10, 50], burst_intervals=[100, 300]):

    signal_length = len(signal_length)

    _noise = np.zeros(signal_length)
    signal_sd = np.std(signal, ddof=1)

    amp = noise_amplitude * signal_sd

    burst_start = np.random.randint(0, (signal_length - burst_durations[1] + 1 )// burst_num_max)

    for _ in range(burst_num_max):
        burst_duration = np.random.uniform(burst_durations[0], burst_durations[1])
        burst_end = burst_start + burst_duration

        if burst_end >= signal_length:
            break
        
        burst_interval = np.random.uniform(burst_intervals[0], burst_intervals[1])
        burst_start = burst_end + burst_interval

        _noise[burst_start: burst_end] += np.random.normal(0, amp)
    
    return _noise


def spectral_density(frequency_range, Magnitude, noise_exponent):
    """
    Calculate the spectral density of pink noise.
    
    Parameters:
        frequency_range (array-like): Array of positive frequencies.
        Magnitude (float): Magnitude of the noise.
        noise_exponent (float): Exponent determining the slope of the spectral density.
        
    Returns:
        array: Spectral density values.
    """
    return Magnitude / (frequency_range ** noise_exponent)

def colored_noise(sampling_rate, duration, noise_max=1, model='pink'):
    """
    Generate colored noise using the specified parameters.
    
    Parameters:
        sampling_rate (int): Sampling rate of the audio signal.
        duration (float): Duration of the colored noise signal in seconds.
        Magnitude (float): Magnitude of the noise.
        noise_exponent (float): Exponent determining the slope of the spectral density.
        noise_max (float): Maximum desired amplitude of the colored noise.
        
    Returns:
        array: Generated colored noise signal.
    """

    if model.lower() == 'pink':
        noise_exponent = 1
        Magnitude = 1
    elif model.lower() in ['brown', 'brownian']:
        noise_exponent = 2
        Magnitude = 1

    num_samples = int(sampling_rate * duration)
    frequency_range = np.fft.fftfreq(num_samples)[1: num_samples // 2]
    
    # Calculate spectral density using the provided function
    _spectral_density = spectral_density(frequency_range, Magnitude, noise_exponent)
    
    # Generate random phases for each frequency component
    random_phases = np.random.uniform(0, 2 * np.pi, len(frequency_range))
    
    # Combine magnitude and phases to form the complex spectrum
    spectrum = np.sqrt(_spectral_density) * np.exp(1j * random_phases)
    
    # Perform inverse FFT to convert complex spectrum to time-domain signal
    _colored_noise = np.fft.irfft(spectrum, n=num_samples)
    
    # Scale the colored noise to achieve the desired maximum amplitude
    scaling = _colored_noise.max() / noise_max
    _colored_noise /= scaling

    return _colored_noise


def flicker_noise(sampling_rate, duration, Magnitude=1, noise_exponent=1, noise_max=1):

    num_samples = int(sampling_rate * duration)
    frequency_range = np.fft.fftfreq(num_samples)[1: num_samples // 2]
    
    # Calculate spectral density using the provided function
    _spectral_density = spectral_density(frequency_range, Magnitude, noise_exponent)
    
    # Generate random phases for each frequency component
    random_phases = np.random.uniform(0, 2 * np.pi, len(frequency_range))
    
    # Combine magnitude and phases to form the complex spectrum
    spectrum = np.sqrt(_spectral_density) * np.exp(1j * random_phases)
    
    # Perform inverse FFT to convert complex spectrum to time-domain signal
    _flicker_noise = np.fft.irfft(spectrum, n=num_samples)
    
    # Scale the flicker noise to achieve the desired maximum amplitude
    scaling = _flicker_noise.max() / noise_max
    _flicker_noise /= scaling

    return _flicker_noise


def thermal_noise(sampling_rate, duration, Temperature, noise_max=1):

    num_samples = int(sampling_rate * duration)
    frequency_range = np.fft.fftfreq(num_samples)[1: num_samples // 2]
    
    # Calculate spectral density
    k = 1.38e-23 # Boltzmann constant
    _spectral_density = k * Temperature / 2
    
    # Generate random phases for each frequency component
    random_phases = np.random.uniform(0, 2 * np.pi, len(frequency_range))
    
    # Combine magnitude and phases to form the complex spectrum
    spectrum = np.sqrt(_spectral_density) * np.exp(1j * random_phases)
    
    # Perform inverse FFT to convert complex spectrum to time-domain signal
    _thermal_noise = np.fft.irfft(spectrum, n=num_samples)
    
    # Scale the thermal noise to achieve the desired maximum amplitude
    scaling = _thermal_noise.max() / noise_max
    _thermal_noise /= scaling

    return _thermal_noise


def powerline_noise(
    signal, sampling_rate=100, duration=10, powerline_frequency=50, powerline_amplitude=0.1
):
    nyquist = sampling_rate * 0.5
    if powerline_frequency > nyquist:
        return np.zeros(len(signal))

    signal_sd = np.std(signal, ddof=1)
    time = np.linsapce(0, duration, duration * sampling_rate)

    powerline_noise = sine_wave(time=time, Amplitude=1, frequency=powerline_frequency, phase=0)

    powerline_amplitude *= signal_sd
    powerline_noise *= powerline_amplitude

    return powerline_noise