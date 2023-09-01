import numpy as np
from PyEMD import EEMD, EMD
from vmdpy import VMD
from statsmodels.tsa.seasonal import seasonal_decompose

def emd_decomposition(signal):
    emd = EMD()
    imfs = emd(signal)
    return imfs

def eemd_decomposition(signal, noise_width=0.05, ensemble_size=100):
    eemd = EEMD(trails=ensemble_size, noise_width=noise_width)
    imfs = eemd.eemd(signal)
    return imfs

def vmd_decomposition(signal, K, alpha=2000, tau=0, DC=0, init=1, tol=1e-7):
    """
    K: how many modes
    alpha: moderate bandwidth constraint
    tau: noise-tolerance (no strict fidelity enforcement)
    DC: whether have DC part imposed
    init: initialize omegas uniformly
    tol:

    Reference: link: https://vmd.robinbetz.com/
    """
    u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)
    return u, u_hat, omega

def seasonal_decomposition(signal, period, model="addative"):
    """
    Parameters:
    model : "addative" or "multiplicative"
    period : Period of the series.
    returns :
    results: get values of results by
    result.seasonal, result.trend, result.resid
    """
    result = seasonal_decompose(signal, model=model, period=period)
    return result


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
#
#     signal = (signal - signal.mean()) / signal.std()
#     signal_clean = (signal_clean - signal_clean.mean()) / signal_clean.std()
#
#     emd_signal = emd_decomposition(signal)
#
#     n_models = len(emd_signal)
#     plt.subplots(n_models+1, 1, figsize=(14, 4*n_models))
#     plt.subplot(n_models+1, 1, 1)
#     plt.plot(signal, label='Original Signal', color='r')
#     plt.plot(signal_clean, label='Clean Signal', alpha=0.3)
#     plt.title("EMD")
#     plt.legend()
#
#     for i in range(1, n_models+1):
#         plt.subplot(n_models+1, 1, i+1)
#         plt.plot(emd_signal[i-1], label='Imf_{}'.format(i))
#         plt.plot(signal_clean, label='Clean Signal', alpha=0.3)
#         plt.legend()
#     plt.savefig('./emd_re.jpg', dpi=300)
#     plt.show()
#
#     eemd_signal = eemd_decomposition(signal)
#     n_models = len(eemd_signal)
#     plt.subplots(n_models+1, 1, figsize=(14, 4*n_models))
#     plt.subplot(n_models+1, 1, 1)
#     plt.plot(signal, label='Original Signal', color='r')
#     plt.plot(signal_clean, label='Clean Signal', alpha=0.3)
#     plt.title("EEMD")
#     plt.legend()
#
#     for i in range(1, n_models+1):
#         plt.subplot(n_models+1, 1, i+1)
#         plt.plot(signal_clean, label='Clean Signal', alpha=0.3)
#         plt.plot(eemd_signal[i-1], label='Imf_{}'.format(i))
#         plt.legend()
#     plt.savefig('./eemd_re.jpg', dpi=300)
#     plt.show()
#
#     vmd_signal, _, _ = vmd_decomposition(signal, K=4)
#     n_models = len(vmd_signal)
#     plt.subplots(n_models+1, 1, figsize=(14, 4*n_models))
#     plt.subplot(n_models+1, 1, 1)
#     plt.plot(signal, label='Original Signal', color='r')
#     plt.plot(signal_clean, label='Clean Signal', alpha=0.3)
#     plt.title("VMD")
#     plt.legend()
#
#     for i in range(1, n_models+1):
#         plt.subplot(n_models+1, 1, i+1)
#         plt.plot(vmd_signal[i-1], label='Imf_{}'.format(i))
#         plt.plot(signal_clean, label='Clean Signal', alpha=0.3)
#         plt.legend()
#     plt.savefig('./vmd_re.jpg', dpi=300)
#
#     plt.show()
#
#     seasonal_signal = seasonal_decomposition(signal, period=int(6000/labels[idx, 2]))
#     print(int(6000/labels[idx, 2]))
#     plt.subplots(4, 1, figsize=(14, 4*4))
#     plt.subplot(4, 1, 1)
#     plt.plot(signal, label='Original Signal', color='r')
#     plt.plot(signal_clean, label='Clean Signal', alpha=0.3)
#     plt.title("Seasonal Decomposition")
#     plt.legend()
#
#     plt.subplot(4, 1, 2)
#     plt.plot(seasonal_signal.trend, label='Trend')
#     plt.plot(signal_clean, label='Clean Signal', alpha=0.3)
#     plt.legend()
#
#     plt.subplot(4, 1, 3)
#     plt.plot(seasonal_signal.seasonal, label='Seasonal')
#     plt.plot(signal_clean, label='Clean Signal', alpha=0.3)
#     plt.legend()
#
#     plt.subplot(4, 1, 4)
#     plt.plot(seasonal_signal.resid, label='Resident')
#     plt.plot(signal_clean, label='Clean Signal', alpha=0.3)
#     plt.legend()
#
#     plt.savefig('./seasonal_re.jpg', dpi=300)
#
#     plt.show()

if __name__ == '__main__':
    from Dataset import load_scg
    import matplotlib.pyplot as plt

    # Load SCG data with specified parameters
    signals, labels, duration, fs = load_scg(0.8, 'train')
    signals_clean, _, _, _ = load_scg(0, 'train')

    idx = 0
    signal = signals[idx]
    signal_clean = signals_clean[idx]

    signal = (signal - signal.mean()) / signal.std()
    signal_clean = (signal_clean - signal_clean.mean()) / signal_clean.std()

    # emd_signal = emd_decomposition(signal)
    #
    # n_models = len(emd_signal)
    # plt.subplots(2, 1, figsize=(14, 4* 2))
    # plt.subplot(2, 1, 1)
    # plt.plot(signal, label='Original Signal', color='r')
    # plt.plot(signal_clean, label='Clean Signal', alpha=0.3)
    # plt.title("Reconstructed by EMD")
    # plt.legend()
    #
    # reconstructed_signal = np.zeros_like(signal)
    # for i in range(0, 4):
    #     reconstructed_signal += emd_signal[i]
    #
    # plt.subplot(2, 1, 2)
    # plt.plot(signal, label='Original Signal', color='r', alpha=0.3)
    # plt.plot(reconstructed_signal, label='Reconstructed Signal')
    # plt.legend()
    #
    # plt.savefig('./emd_re.jpg', dpi=300)
    # plt.show()
    #
    # eemd_signal = eemd_decomposition(signal)
    # n_models = len(eemd_signal)
    # plt.subplots(2, 1, figsize=(14, 4* 2))
    # plt.subplot(2, 1, 1)
    # plt.plot(signal, label='Original Signal', color='r')
    # plt.plot(signal_clean, label='Clean Signal', alpha=0.3)
    # plt.title("Reconstructed by EEMD")
    # plt.legend()
    #
    # reconstructed_signal = np.zeros_like(signal)
    # for i in range(0, 4):
    #     reconstructed_signal += eemd_signal[i]
    #
    # plt.subplot(2, 1, 2)
    # plt.plot(signal, label='Original Signal', color='r', alpha=0.3)
    # plt.plot(reconstructed_signal, label='Reconstructed Signal')
    # plt.legend()
    #
    # plt.savefig('./eemd_re.jpg', dpi=300)
    # plt.show()

    # vmd_signal, _, _ = vmd_decomposition(signal, K=5)
    # n_models = len(vmd_signal)
    # plt.subplots(2, 1, figsize=(14, 4* 2))
    # plt.subplot(2, 1, 1)
    # plt.plot(signal, label='Original Signal', color='r')
    # plt.plot(signal_clean, label='Clean Signal', alpha=0.3)
    # plt.title("Reconstructed by VMD")
    # plt.legend()
    #
    # reconstructed_signal = np.zeros_like(signal)
    # for i in range(1, n_models):
    #     reconstructed_signal += vmd_signal[i]
    # plt.subplot(2, 1, 2)
    # plt.plot(reconstructed_signal, label='Reconstructed Signal')
    # plt.plot(signal_clean, label='Clean Signal', alpha=0.3)
    # plt.legend()
    # plt.savefig('./vmd_re.jpg', dpi=300)
    #
    # plt.show()

    seasonal_signal = seasonal_decomposition(signal, period=int(6000/labels[idx, 2]))
    print(int(6000/labels[idx, 2]))
    plt.subplots(2, 1, figsize=(14, 4*2))
    plt.subplot(2, 1, 1)
    plt.plot(signal, label='Original Signal', color='r')
    plt.plot(signal_clean, label='Clean Signal', alpha=0.3)
    plt.title("Reconstructed by Seasonal Decomposition")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(seasonal_signal.seasonal + seasonal_signal.resid, label='Reconstructed Signal')
    plt.plot(signal_clean, label='Clean Signal', alpha=0.3)
    plt.legend()

    plt.savefig('./seasonal_re.jpg', dpi=300)

    plt.show()

