from sklearn.decomposition import FastICA, PCA

def algo_bss_ica(X, n_components):
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


def algo_bss_pca(X, n_components):
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
    import numpy as np
    import matplotlib.pyplot as plt
    from Dataset import load_scg

    signals, labels, duration, fs = load_scg(0.1, 'train')

    # We simulate three people lying in bed, and there are three sensors detecting their SCG signals.
    # For each sensor, it receives a combination of three SCG signals, each with a certain time delay
    # and attenuation.
    s1 = signals[3]
    s2 = signals[4]
    s3 = signals[5]

    time = np.linspace(0, duration, duration * fs)
    S = np.c_[s1, s2, s3]
    print(S.shape)

    S /= S.std(axis=0)  # Standardize data

    # Mix data
    A = np.array([[0.9, 0.8, 1], [0.5, 0.85, 1.0], [0.9, 0.6, 0.8]])  # Mixing matrix
    X = np.dot(S, A.T)  # Generate observations

    S_, A_ = algo_bss_ica(X, 3)
    H = algo_bss_pca(X, 3)

    plt.figure()

    models = [X, S, S_, H]
    names = ['Observations (mixed signal)', 'True Sources', 'ICA recovered signals', 'PCA recovered signals']
    colors = ['red', 'steelblue', 'orange']


    plt.figure(figsize=(16, 8))
    for ii, (model, name) in enumerate(zip(models, names), 1):
        plt.subplot(4, 1, ii)
        plt.title(name)
        for sig, color in zip(model.T, colors):
            plt.plot(sig, color=color)

    plt.tight_layout()
    plt.show()
