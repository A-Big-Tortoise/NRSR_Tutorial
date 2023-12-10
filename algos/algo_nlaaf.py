from DTW import dtwPlotTwoWay, dtw_easy
import numpy as np
import random


def make_template(piece0, piece1, path):
    """
    Create a template by averaging aligned segments of two time series pieces based on a given DTW path.

    Parameters:
    - piece0, piece1 (np.ndarray): Time series pieces to interpolate between.
    - path (tuple): DTW path between the two pieces.

    Returns:
    - np.ndarray: Averaged template based on the DTW path.
    """
    path0 = path[0]
    path1 = path[1]
    new_piece0 = np.array([piece0[idx] for idx in path0])
    new_piece1 = np.array([piece1[idx] for idx in path1])

    template = 0.5 * (new_piece0 + new_piece1)
    return template


def performNLAAF1(pieces, dist=lambda x, y: np.abs(x - y)):
    """
    Perform Non-Local Adaptive Averaging Fusion 1 (NLAAF1) on a list of time series pieces.

    Parameters:
    - pieces (list): List of time series pieces (1D arrays) to be fused.
    - dist (function): Distance function for Dynamic Time Warping (DTW). Default is absolute difference.

    Returns:
    - np.ndarray: Fused time series obtained through NLAAF1.
    """

    # 2^N
    # 1. drop out something
    # 2. use loops to simulate recursive
    # 3. get the template

    pieces_num = len(pieces)
    k = 1
    while pieces_num >= k:
        k *= 2
    k = int(k / 2)

    random_choice = random.sample(range(pieces_num), k)
    chosen_pieces = [pieces[choice] for choice in random_choice]

    this_term = chosen_pieces

    while k > 1:
        last_term = this_term
        this_term = []

        for cnt in range(0, k, 2):
            a = cnt
            b = cnt + 1
            piece1, piece2 = last_term[a], last_term[b]

            _, _, _, path = dtw_easy(piece1, piece2, dist)
            template = make_template(piece1, piece2, path)
            this_term.append(template)

        k = int(k / 2)

    return np.array(this_term[0])


def performNLAAF2(pieces, dist=lambda x, y: np.abs(x - y)):
    """
    Perform Non-Local Adaptive Averaging Fusion 2 (NLAAF2) on a list of time series pieces.

    Parameters:
    - pieces (list): List of time series pieces (1D arrays) to be fused.
    - dist (function): Distance function for Dynamic Time Warping (DTW). Default is absolute difference.

    Returns:
    - np.ndarray: Fused time series obtained through NLAAF2.
    """
    # one by one
    # 1. use loops to calculate
    # 2. get the template

    pieces_num = len(pieces)
    _, _, _, path = dtw_easy(pieces[0], pieces[1], dist)
    template = make_template(pieces[0], pieces[1], path)

    for cnt in range(2, pieces_num):
        _, _, _, path = dtw_easy(template, pieces[cnt], dist)
        template = make_template(template, pieces[cnt], path)

    return template

if __name__ == '__main__':
    pieces = [[12, 3, 4, 5], [3, 1, 1, 1]]
    template1 = performNLAAF1(pieces)
    template2 = performNLAAF2(pieces)