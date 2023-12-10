from DTW import dtwPlotTwoWay, dtw_easy
import numpy as np
from scipy.interpolate import CubicSpline


def performICDTW(pieces, iter_max=10, dist=lambda x, y: np.abs(x - y), Beta_A1=1e-5, Beta_A2=1e5, Beta_B=1, show=False):
    """
    Perform Iterative Constrained Dynamic Time Warping (ICDTW) on a list of time series pieces.

    Parameters:
    - pieces (list): List of time series pieces (1D arrays) to be averaged.
    - iter_max (int): Maximum number of iterations for ICDTW. Default is 10.
    - dist (function): Distance function for Dynamic Time Warping (DTW). Default is absolute difference.
    - Beta_A1, Beta_A2, Beta_B (float): Parameters for ICDTW optimization. Default values provided.

    Returns:
    - np.ndarray: Averaged time series obtained through ICDTW.

    Raises:
    - TypeError: If the input 'pieces' is not a list.
    """

    def make_template(piece1, piece2, path, w1, w2):
        """
        Create a template by interpolating between two time series pieces based on the DTW path.
        Reference Paper:
            Niennattrakul, Vit, Dararat Srisai, and Chotirat Ann Ratanamahatana. "Shape-based template matching for time series data." Knowledge-Based Systems 26 (2012): 1-8.

        Parameters:
        - piece1, piece2 (np.ndarray): Time series pieces to interpolate between.
        - path (tuple): DTW path between the two pieces.
        - w1, w2 (float): Weights for the interpolation.

        Returns:
        - np.ndarray, np.ndarray: Interpolated x and y values of the template.
        """
        path1 = path[0]
        path2 = path[1]
        x, y = [], []

        for x_1, x_2 in zip(path1, path2):
            x_new = (w1 * x_1 + w2 * x_2) / (w1 + w2)
            y_new = (piece1[x_1] * w1 + piece2[x_2] * w2) / (w1 + w2)
            x.append(x_new)
            y.append(y_new)

        return np.array(x), np.array(y)

    def cdtw_averaging(piece1, piece2, w1, w2):
        """
        Perform Constrained DTW (CDTW) averaging between two time series pieces.

        Parameters:
        - piece1, piece2 (np.ndarray): Time series pieces to average.
        - w1, w2 (float): Weights for the averaging.

        Returns:
        - np.ndarray: Averaged time series obtained through CDTW.
        """
        dist = lambda x, y: np.abs(x - y)
        _, _, _, path = dtw_easy(piece1, piece2, dist)

        N_ = min(len(piece1), len(piece2))
        x, y = make_template(piece1, piece2, path, w1, w2)

        # 创建CubicSpline对象
        cs = CubicSpline(x, y)
        new_x = np.linspace(0, N_, N_, endpoint=False)
        new_y = cs(new_x)

        return new_y

    def icdtw_averaging(A, B, w_A, w_B, iter_max, dist, Beta_A1, Beta_A2, Beta_B):
        """
        Perform Iterative Constrained Dynamic Time Warping (ICDTW) averaging between two time series pieces.

        Parameters:
        - A, B (np.ndarray): Time series pieces to average.
        - w_A, w_B (float): Weights for the averaging.
        - iter_max (int): Maximum number of iterations for ICDTW.
        - dist (function): Distance function for DTW.
        - Beta_A1, Beta_A2, Beta_B (float): Parameters for ICDTW optimization.

        Returns:
        - np.ndarray: Averaged time series obtained through ICDTW.
        """

        iter_n = 0
        C = None
        while iter_n < iter_max:
            iter_n += 1
            # print(np.abs(dis_CA - dis_CB))
            Beta_A3 = 0.5 * (Beta_A1 + Beta_A2)
            C = cdtw_averaging(A, B, Beta_A3, Beta_B)
            CA = dtw_easy(C, A, dist) * w_A
            dis_CA = CA[0]
            # dis_CA, _, _, _ = dtw_easy(C, A, dist) * w_A

            CB = dtw_easy(C, B, dist) * w_B
            dis_CB = CB[0]
            # dis_CB, _, _, _ = dtw_easy(C, B, dist) * w_B
            if dis_CA < dis_CB:
                Beta_A1 = Beta_A3
            else:
                Beta_A2 = Beta_A3

        return C

    # Check if 'pieces' is a list
    if not isinstance(pieces, list):
        raise TypeError("Variable 'pieces' must be a list.")

    # Initialize weights for each piece
    weights = [1] * len(pieces)

    # Perform ICDTW until only one piece remains
    while len(pieces) > 1:
        A, B = pieces[0], pieces[1]

        w_A, w_B = weights[0], weights[1]
        C = icdtw_averaging(A, B, w_A, w_B, iter_max, dist, Beta_A1, Beta_A2, Beta_B)
        w_C = w_A + w_B
        pieces.append(C)
        weights.append(w_C)

        pieces = pieces[2:]
        weights = weights[2:]

    return pieces[0]


if __name__ == '__main__':
    arr = [[12, 3, 4, 5], [3, 1, 1, 1]]
    Beta_A1, Beta_A2, Beta_B = 1e-5, 1e5, 1
    iter_max = 10
    dist = lambda x, y: np.abs(x - y)
    template = performICDTW(arr, iter_max, dist, Beta_A1, Beta_A2, Beta_B)