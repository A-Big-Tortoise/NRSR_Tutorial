import numpy as np
from scipy.signal import correlate

def matched_filter(signal, model=0):
    # model: 0->'full', 1->'valid', 2->'same'
    if model==0:
        model_str = 'full'
    elif model==1:
        model_str = 'valid'
    elif model==2:
        model_str = 'same'

    return correlate(signal, signal, mode=model_str)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    received_signal = np.array([0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0]) + 0.5 * np.random.randn(13)
    matched_output = matched_filter(received_signal)

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(received_signal, marker='o', linestyle='-', color='b')
    plt.title('received_signal')

    plt.subplot(2, 1, 2)
    plt.plot(matched_output, marker='o', linestyle='-', color='r')
    plt.title('output of matched_filter')

    plt.tight_layout()
    plt.show()

