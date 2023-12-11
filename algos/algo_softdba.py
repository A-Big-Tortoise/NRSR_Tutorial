from tslearn.barycenters import softdtw_barycenter

def performSOFTDBA(pieces):
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
    template = softdtw_barycenter(pieces)

    return template


