from sklearn.decomposition import PCA

def algo_pca(X, n_components):
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

if __name__ == '__main__':

    # Import necessary functions/classes from Dataset module
    from Dataset import load_scg
    from decompose import eemd_decomposition
    import matplotlib.pyplot as plt

    # Load SCG data with specified parameters
    # 'train': data split to load (train/validation/test)
    signals, labels, duration, fs = load_scg(0.8, 'train')

    idx = 0
    signal = signals[idx]
    standized_signal = (signal-signal.mean()) / signal.std()

    decomposed_signal = eemd_decomposition(standized_signal)
    reconstructed_signal = algo_pca(decomposed_signal[:2].T, 1)

    plt.title('EEMD Decomposition and PCA-based Denoise')
    plt.plot(standized_signal, alpha=0.5, label='Noisy Signal')
    plt.plot(reconstructed_signal, label='Denoised Signal')
    plt.legend()
    plt.show()







