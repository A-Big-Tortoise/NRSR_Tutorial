import numpy as np
from PyEMD import EEMD, EMD, CEEMDAN
from vmdpy import VMD
from statsmodels.tsa.seasonal import seasonal_decompose
# from Tutorial.utils import plot_decomposed_components
from utils import plot_decomposed_components
import matplotlib.pyplot as plt

def standize_1D(signal):
    return (signal - signal.mean()) / signal.std()

def emd_decomposition(signal, show=False):
    signal = standize_1D(signal)
    emd = EMD()
    imfs = emd(signal)
    if show:
        plot_decomposed_components(signal, imfs, 'EMD')
    return imfs

def eemd_decomposition(signal, noise_width=0.05, ensemble_size=100, show=False):
    signal = standize_1D(signal)
    eemd = EEMD(trails=ensemble_size, noise_width=noise_width)
    imfs = eemd.eemd(signal)
    if show:
        plot_decomposed_components(signal, imfs, 'EEMD')
    return imfs

def serial_emd_decomposition(signals, num_interval, show=False):
    # Serial - EMD: Fast Empirical Mode Decomposition Method for Multi - dimensional Signals Based on Serialization
    # https://github.com/ffbear1993/serial-emd

    def concatenate(matrix_x: np.ndarray, num_interval: int):
        num_length = matrix_x.shape[0]
        num_signal = matrix_x.shape[1]

        matrix_a = matrix_x[:num_interval, 1:]
        matrix_b = matrix_x[-num_interval:, :-1]

        vector_a = np.linspace(0, 1, num_interval + 2)[1:-1].reshape(-1, 1)
        vector_u = np.ones(num_signal - 1).reshape(-1, 1)

        # transition
        matrix_t_a = np.flipud(matrix_a) * np.dot(vector_a, vector_u.T)
        matrix_t_b = np.flipud(matrix_b) * np.dot(np.flipud(vector_a), vector_u.T)

        matrix_t = matrix_t_a + matrix_t_b

        matrix_z = np.zeros(num_interval).reshape(-1, 1)

        # concatenate transition with zeros
        matrix_t = np.concatenate([matrix_t, matrix_z], axis=1)

        # result
        matrix_r = np.concatenate([matrix_x, matrix_t], axis=0)
        matrix_r = matrix_r.flatten(order='F')
        matrix_r = matrix_r[:-num_interval].reshape(-1, 1)

        return matrix_r

    def deconcatenate(matrix_r: np.ndarray, num_interval: int, num_signal: int):
        num_mode = matrix_r.shape[1]

        # fill zeros
        matrix_z = np.zeros([num_interval, num_mode])
        matrix_r = np.concatenate([matrix_r, matrix_z], axis=0)

        matrix_imf = matrix_r.reshape([-1, num_signal, num_mode], order='F')
        matrix_imf = matrix_imf[:-num_interval, :, :]
        matrix_imf = matrix_imf.transpose((0, 2, 1))

        return matrix_imf

    # signals (num_signal, length)
    num_signal, _ = signals.shape
    signals = signals.T
    signals = (signals - signals.mean(0, keepdims=True)) / signals.std(0, keepdims=True)

    serilized_x = concatenate(signals, num_interval).reshape(-1)
    serilized_imfs = EMD()(serilized_x).T
    # imfs for each signal, [num_length, num_imf_modes, num_signal]
    imfs = deconcatenate(serilized_imfs, num_interval, num_signal)

    print(f'concatenated signal shape:\t{serilized_x.shape}')
    print(f'concatenated imfs shape:\t{serilized_imfs.shape}')
    print(f'multi-dimensional imfs shape:\t{imfs.shape}')

    # the shape of imfs (signal_length, decomposed_components, num_signal)
    return imfs

def ceemd_decomposition(signal, show=False):
    signal = standize_1D(signal)
    ceemdan = CEEMDAN()
    imfs = ceemdan.ceemdan(signal)
    if show:
        plot_decomposed_components(signal, imfs, 'CEEMD')
    return imfs

def vmd_decomposition(signal, K=5, alpha=2000, tau=0, DC=0, init=1, tol=1e-7, show=False):
    """
    K: how many modes
    alpha: moderate bandwidth constraint
    tau: noise-tolerance (no strict fidelity enforcement)
    DC: whether have DC part imposed
    init: initialize omegas uniformly
    tol:

    Reference: link: https://vmd.robinbetz.com/
    """
    signal = standize_1D(signal)
    u, _, _ = VMD(signal, alpha, tau, K, DC, init, tol)
    if show:
        plot_decomposed_components(signal, u, 'VMD')
    return u

def seasonal_decomposition(signal, period=100, model=0, show=False):
    """
    Parameters:
    model(int) : 0->"addative" or 1->"multiplicative"
    period : Period of the series.
    returns :
    results: get values of results by
    result.seasonal, result.trend, result.resid
    """
    signal = standize_1D(signal)

    stl_model = None
    if model == 0:
        stl_model = "addative"
    elif model == 1:
        stl_model = "multiplicative"
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
        plt.plot(components.resid, label='Resident')
        plt.legend()
        plt.show()

    return components


if __name__ == '__main__':
    from Dataset import load_scg

    signals, labels, duration, fs = load_scg(0.8, 'train')

    idx = 0
    signal = signals[idx]
    seasonal_decomposition(signal, show=True)
    emd_decomposition(signal, show=True)
    eemd_decomposition(signal, show=True)
    ceemd_decomposition(signal, show=True)
    vmd_decomposition(signal, show=True)
    serial_emd_decomposition(signals[:5, :], num_interval=5)

# if __name__ == '__main__':
#     from Dataset import load_scg
#     import matplotlib.pyplot as plt
#
#     # Load SCG data with specified parameters
#     signals, labels, duration, fs = load_scg(0.8, 'train')
#     signals_clean, _, _, _ = load_scg(0, 'train')
#
#     idx = 0
#     signal = signals[idx]
#     signal_clean = signals_clean[idx]
#
#     signal = (signal - signal.mean()) / signal.std()
#     signal_clean = (signal_clean - signal_clean.mean()) / signal_clean.std()
#
#     # emd_signal = emd_decomposition(signal)
#     #
#     # n_models = len(emd_signal)
#     # plt.subplots(2, 1, figsize=(14, 4* 2))
#     # plt.subplot(2, 1, 1)
#     # plt.plot(signal, label='Original Signal', color='r')
#     # plt.plot(signal_clean, label='Clean Signal', alpha=0.3)
#     # plt.title("Reconstructed by EMD")
#     # plt.legend()
#     #
#     # reconstructed_signal = np.zeros_like(signal)
#     # for i in range(0, 4):
#     #     reconstructed_signal += emd_signal[i]
#     #
#     # plt.subplot(2, 1, 2)
#     # plt.plot(signal, label='Original Signal', color='r', alpha=0.3)
#     # plt.plot(reconstructed_signal, label='Reconstructed Signal')
#     # plt.legend()
#     #
#     # plt.savefig('./emd_re.jpg', dpi=300)
#     # plt.show()
#     #
#     # eemd_signal = eemd_decomposition(signal)
#     # n_models = len(eemd_signal)
#     # plt.subplots(2, 1, figsize=(14, 4* 2))
#     # plt.subplot(2, 1, 1)
#     # plt.plot(signal, label='Original Signal', color='r')
#     # plt.plot(signal_clean, label='Clean Signal', alpha=0.3)
#     # plt.title("Reconstructed by EEMD")
#     # plt.legend()
#     #
#     # reconstructed_signal = np.zeros_like(signal)
#     # for i in range(0, 4):
#     #     reconstructed_signal += eemd_signal[i]
#     #
#     # plt.subplot(2, 1, 2)
#     # plt.plot(signal, label='Original Signal', color='r', alpha=0.3)
#     # plt.plot(reconstructed_signal, label='Reconstructed Signal')
#     # plt.legend()
#     #
#     # plt.savefig('./eemd_re.jpg', dpi=300)
#     # plt.show()
#
#     # vmd_signal, _, _ = vmd_decomposition(signal, K=5)
#     # n_models = len(vmd_signal)
#     # plt.subplots(2, 1, figsize=(14, 4* 2))
#     # plt.subplot(2, 1, 1)
#     # plt.plot(signal, label='Original Signal', color='r')
#     # plt.plot(signal_clean, label='Clean Signal', alpha=0.3)
#     # plt.title("Reconstructed by VMD")
#     # plt.legend()
#     #
#     # reconstructed_signal = np.zeros_like(signal)
#     # for i in range(1, n_models):
#     #     reconstructed_signal += vmd_signal[i]
#     # plt.subplot(2, 1, 2)
#     # plt.plot(reconstructed_signal, label='Reconstructed Signal')
#     # plt.plot(signal_clean, label='Clean Signal', alpha=0.3)
#     # plt.legend()
#     # plt.savefig('./vmd_re.jpg', dpi=300)
#     #
#     # plt.show()
#
#     seasonal_signal = seasonal_decomposition(signal, period=int(6000/labels[idx, 2]))
#     print(int(6000/labels[idx, 2]))
#     plt.subplots(2, 1, figsize=(14, 4*2))
#     plt.subplot(2, 1, 1)
#     plt.plot(signal, label='Original Signal', color='r')
#     plt.plot(signal_clean, label='Clean Signal', alpha=0.3)
#     plt.title("Reconstructed by Seasonal Decomposition")
#     plt.legend()
#
#     plt.subplot(2, 1, 2)
#     plt.plot(seasonal_signal.seasonal + seasonal_signal.resid, label='Reconstructed Signal')
#     plt.plot(signal_clean, label='Clean Signal', alpha=0.3)
#     plt.legend()
#
#     plt.savefig('./seasonal_re.jpg', dpi=300)
#
#     plt.show()
#
