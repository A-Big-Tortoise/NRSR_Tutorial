from sklearn.decomposition import FastICA, PCA

def algo_bbs_ica(X, n_components):
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


def algo_bbs_pca(X, n_components):
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
    from scipy import signal
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(0)
    n_samples = 2000
    time = np.linspace(0, 8, n_samples)

    s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
    s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
    s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

    S = np.c_[s1, s2, s3]
    S += 0.2 * np.random.normal(size=S.shape)  # Add noise

    S /= S.std(axis=0)  # Standardize data
    # Mix data
    A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
    X = np.dot(S, A.T)  # Generate observations

    S_, A_ = algo_bbs_ica(X, 3)
    H = algo_bbs_ica(X, 3)

    plt.figure()

    models = [X, S, S_, H]
    names = ['Observations (mixed signal)',
             'True Sources',
             'ICA recovered signals',
             'PCA recovered signals']
    colors = ['red', 'steelblue', 'orange']

    plt.figure(figsize=(16, 8))
    for ii, (model, name) in enumerate(zip(models, names), 1):
        plt.subplot(4, 1, ii)
        plt.title(name)
        for sig, color in zip(model.T, colors):
            plt.plot(sig, color=color)

    plt.tight_layout()
    plt.show()
