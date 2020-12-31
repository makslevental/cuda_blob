import numbers

import numpy as np


def nd_to_text(A, w=None, h=None):
    if A.ndim == 1:
        if w is None:
            return str(A)
        else:
            s = "["
            for i, AA in enumerate(A[:-1]):
                s += str(AA) + " " * (max(w[i], len(str(AA))) - len(str(AA)) + 1)
            s += (
                    str(A[-1])
                    + " " * (max(w[-1], len(str(A[-1]))) - len(str(A[-1])))
                    + "] "
            )
    elif A.ndim == 2:
        w1 = [max([len(str(s)) for s in A[:, i]]) for i in range(A.shape[1])]
        w0 = sum(w1) + len(w1) + 1
        s = u"\u250c" + u"\u2500" * w0 + u"\u2510" + "\n"
        for AA in A:
            s += " " + nd_to_text(AA, w=w1) + "\n"
        s += u"\u2514" + u"\u2500" * w0 + u"\u2518"
    elif A.ndim == 3:
        h = A.shape[1]
        s1 = u"\u250c" + "\n" + (u"\u2502" + "\n") * h + u"\u2514" + "\n"
        s2 = u"\u2510" + "\n" + (u"\u2502" + "\n") * h + u"\u2518" + "\n"
        strings = [nd_to_text(a) + "\n" for a in A]
        strings.append(s2)
        strings.insert(0, s1)
        s = "\n".join("".join(pair) for pair in zip(*map(str.splitlines, strings)))
    return s


def print_nd_array(arr: np.ndarray, round=3):
    print(nd_to_text(arr.round(round)))


def gaussian_kernel(width: int = 21, sigma: int = 3, dim: int = 2) -> np.ndarray:
    """Gaussian kernel
    Parameters
    ----------
    width: bandwidth of the kernel
    sigma: std of the kernel
    dim: dimensions of the kernel (images -> 2)

    Returns
    -------
    kernel : gaussian kernel
    """
    assert width > 2

    if isinstance(width, numbers.Number):
        width = [width] * dim
    if isinstance(sigma, numbers.Number):
        sigma = [sigma] * dim
    kernel = 1
    meshgrids = np.meshgrid(*[np.arange(size, dtype=np.float32) for size in width])
    for size, std, mgrid in zip(width, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= (
            1 / (std * np.sqrt(2 * np.pi)) * np.exp(-(((mgrid - mean) / std) ** 2) / 2)
        )

    # Make sure sum of values in gaussian kernel equals 1.
    return kernel / np.sum(kernel)