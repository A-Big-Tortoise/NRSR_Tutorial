import padasip as pa
import numpy as np

def filter_rls(x, d, n, mu):
    x_np = np.array(x)
    d_np = np.array(d)

    if x_np.ndim == 1:
        x_np = x_np.reshape(-1, 1)
    if d_np.ndim == 1:
        d_np = d_np.reshape(-1, 1)

    f = pa.filters.FilterRLS(n=n, mu=mu, w="random")
    y, e, w = f.run(d_np, x_np)

    return y, e, w

def filter_lms(x, d, n, mu):
    x_np = np.array(x)
    d_np = np.array(d)

    if x_np.ndim == 1:
        x_np = x_np.reshape(-1, 1)
    if d_np.ndim == 1:
        d_np = d_np.reshape(-1, 1)

    f = pa.filters.FilterLMS(n=n, mu=mu, w="random")
    y, e, w = f.run(d_np, x_np)
    return y, e, w
