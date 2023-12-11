from numpy import array, zeros, full, argmin, inf, ndim
from math import isinf
from matplotlib import collections as mc
import numpy as np
import matplotlib.pyplot as plt

# def dtwPlotTwoWay(path1, path2, xts, yts, xoffset, yoffset, match_col):
#
#     xts = xts + xoffset
#     yts = yts + yoffset
#
#     maxlen = max(len(xts), len(yts))
#     times = np.arange(maxlen)
#     xts = np.pad(xts, (0, maxlen - len(xts)), "constant", constant_values=np.nan)
#     yts = np.pad(yts, (0, maxlen - len(yts)), "constant", constant_values=np.nan)
#
#     fig, ax = plt.subplots()
#
#     ax.plot(times, xts, color='k')
#     ax.plot(times, yts)
#
#     # https://stackoverflow.com/questions/21352580/matplotlib-plotting-numerous-disconnected-line-segments-with-different-colors
#     idx = np.linspace(0, len(path1) - 1)
#
#     idx = np.array(idx).astype(int)
#
#     col = []
#     for i in idx:
#         col.append([(path1[i], xts[path1[i]]),
#                     (path2[i], yts[path2[i]])])
#
#     lc = mc.LineCollection(col, linewidths=1, linestyles=":", colors=match_col)
#     ax.add_collection(lc)
#
#     plt.show()

def dtwPlotTwoWay(path1, path2, xts, yts, xoffset, yoffset, match_col):

        xts = xts + xoffset
        yts = yts + yoffset

        maxlen = max(len(xts), len(yts))
        times = np.arange(maxlen)
        xts = np.pad(xts, (0, maxlen - len(xts)), "constant", constant_values=np.nan)
        yts = np.pad(yts, (0, maxlen - len(yts)), "constant", constant_values=np.nan)

        fig, ax = plt.subplots(1, 2, figsize=(14, 3))


        ax[0].plot(times, xts, color='k')
        ax[0].plot(times, yts)

        # https://stackoverflow.com/questions/21352580/matplotlib-plotting-numerous-disconnected-line-segments-with-different-colors
        idx = np.linspace(0, len(path1) - 1)

        idx = np.array(idx).astype(int)

        col = []
        for i in idx:
            col.append([(path1[i], xts[path1[i]]),
                        (path2[i], yts[path2[i]])])

        lc = mc.LineCollection(col, linewidths=1, linestyles=":", colors=match_col)
        ax[0].add_collection(lc)

        ax[1].plot(xts-xoffset)
        ax[1].plot(yts-yoffset)

        plt.show()


def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)


def dtw_easy(x, y, dist, warp=1, s=1.0):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.

    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    :param int warp: how many shifts are computed.
    :param int w: window size limiting the maximal distance between indices of matched entries |i,j|.
    :param float s: weight applied on off-diagonal moves of the path. As s gets larger, the warping path is increasingly biased towards the diagonal
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    assert s > 0
    r, c = len(x), len(y)

    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf

    D1 = D0[1:, 1:]  # view

    for i in range(r):
        for j in range(c):
            D1[i, j] = dist(x[i], y[j])
    C = D1.copy()

    jrange = range(c)
    for i in range(r):
        for j in jrange:
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                i_k = min(i + k, r)
                j_k = min(j + k, c)
                min_list += [D0[i_k, j] * s, D0[i, j_k] * s]
            D1[i, j] += min(min_list)
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1], C, D1, path

def dtw(x, y, dist, warp=1, w=inf, s=1.0):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.

    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    :param int warp: how many shifts are computed.
    :param int w: window size limiting the maximal distance between indices of matched entries |i,j|.
    :param float s: weight applied on off-diagonal moves of the path. As s gets larger, the warping path is increasingly biased towards the diagonal
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    assert isinf(w) or (w >= abs(len(x) - len(y)))
    assert s > 0
    r, c = len(x), len(y)

    if not isinf(w):
        D0 = full((r + 1, c + 1), inf)
        for i in range(1, r + 1):
            D0[i, max(1, i - w):min(c + 1, i + w + 1)] = 0
        D0[0, 0] = 0
    else:
        D0 = zeros((r + 1, c + 1))
        D0[0, 1:] = inf
        D0[1:, 0] = inf

    D1 = D0[1:, 1:]  # view

    for i in range(r):
        for j in range(c):
            if (isinf(w) or (max(0, i - w) <= j <= min(c, i + w))):
                D1[i, j] = dist(x[i], y[j])
    C = D1.copy()

    jrange = range(c)
    for i in range(r):
        if not isinf(w):
            jrange = range(max(0, i - w), min(c, i + w + 1))
        for j in jrange:
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                i_k = min(i + k, r)
                j_k = min(j + k, c)
                min_list += [D0[i_k, j] * s, D0[i, j_k] * s]
            D1[i, j] += min(min_list)
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1], C, D1, path

def wdtw(x, y, weight_vector, sakoe_chiba_band, alpha):
    n = len(x)
    m = len(y)

    dist = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            dist[i][j] = (1 / (pow(weight_vector[j], alpha))) * (x[i]-y[j])**2

    # Cost Matrix with Sakoe-Chiba Band
    # -------------------------------------------------------------------------------------------
    dtw_cost = np.empty((n, m))
    dtw_cost.fill(0)
    dtw_cost[0][0] = dist[0][0]
    for i in range(1,n):
        dtw_cost[i][0] = dtw_cost[i-1][0] + dist[i][0]
    for j in range(1,m):
        dtw_cost[0][j] = dtw_cost[0][j-1] + dist[0][j]
    for i in range(1,n):
        for j in range(1,m):
            if abs(i-j) <= sakoe_chiba_band:
                choices = dtw_cost[i-1][j], dtw_cost[i][j-1], dtw_cost[i-1][j-1]
                dtw_cost[i][j] = dist[i][j] + min(choices)
            else:
                dtw_cost[i][j] = float('inf')

    # Compute Warping Path
    # -------------------------------------------------------------------------------------------
    i = n-1
    j = m-1
    path = np.empty((n, m))
    path.fill(0)
    path[n-1][m-1] = 1
    size_warping_path = 1
    while i > 0 or j > 0:
        if i == 0:
            j = j - 1
        elif j == 0:
            i = i - 1
        else:
            choices = dtw_cost[i-1][j], dtw_cost[i][j-1], dtw_cost[i-1][j-1]
            if dtw_cost[i-1,j-1] == min(choices):
                i = i - 1
                j = j - 1
            elif dtw_cost[i,j-1] == min(choices):
                j = j - 1
            else:
                i = i - 1
        path[i][j] = 1
        size_warping_path += size_warping_path

    # Return Weighted Dynamic Time Warping Distance
    # -------------------------------------------------------------------------------------------
    return dtw_cost[-1][-1]