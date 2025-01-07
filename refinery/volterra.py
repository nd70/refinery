import numpy as np
import sys


def volterra_pipeline(d, wit1, wit2, M):
    """
    Use this function to run the second order volterra filter
    in order to regress the dirty system signal d with the witness
    channels wit1 & wit2. Use M-taps. NOTE: the volterra kernel
    grows as (M+1)^4 so things will crawl to a stop very quickly if
    you're not careful!

    Parameters
    ----------
    d : `numpy.ndarray`
        noisy system signal
    wit1 : `numpy.ndarray`
        witness channel for noise regression
    wit2 : `numpy.ndarray`
        witness channel for noise regression
    M : `int`
        number of filter taps (a.k.a filter order)

    Returns
    -------
    clean : `numpy.ndarray`
        cleaned system signal
    """
    P = three_point_static_corr(d, wit1, wit2, M=M)
    out = four_point_corr(wit1, wit2, wit1, wit2, M)
    vc = out.reshape(((M + 1) ** 2, (M + 1) ** 2)).T
    weights = np.linalg.pinv(vc).dot(P)
    est = apply_weights_2d(wit1, wit2, weights.reshape(((M + 1), (M + 1))))
    clean = d - est
    return clean


def three_point_static_corr(a, b, c, M):
    """
    calculate the 3-point correlation of witness channels
    b & c with the target channel a. Mathematically, we find the
    matrix P by the following,

        P[i,j] = <a[k] b[k-i] c[k-j]>

    Note that the target array a does not get a time shift, hence
    the difference between a three-point correlation and a static
    three-point correlation

    Parameters
    ----------
    a : `numpy.ndarray`
        target array
    b : `numpy.ndarray`
        witness channe
    c : `numpy.ndarray`
        witness channel
    M : `int`
        number of filter taps (a.k.a filter order)

    Returns
    -------
    arr : `numpy.ndarray'
        2D bi-correlation array of size (M+1) x (M+1)
    """

    if a.size != b.size or b.size != c.size:
        sys.exit("Array sizes do not match!")

    # pad and stack (makes rolling cleaner)
    a = np.pad(a, (M, M))
    b = np.pad(b, (M, M))
    c = np.pad(c, (M, M))

    # fill the correlation matrix
    out = np.zeros((2 * M + 1, 2 * M + 1))
    for i in range(-M, M + 1):
        for mp in range(-M, M + 1):
            out[mp, i] = np.sum(a * np.roll(b, mp) * np.roll(c, i))

    out = out[: M + 1, : M + 1].flatten()
    return out


def four_point_corr(a, b, c, d, M):
    """
    calculate the 4-point correlation of witness channels
    a, b, c and d. Mathematically, we find the matrix C[m, m', i, j]
    by the following,

        C[m, m',i, j] = <a[k-m] b[k-m'] c[k-i] d[k-j]>

    In practice, with the Volterra filter, a[k] = c[k] and
    b[k] = d[k] giving

        C[m, m',i, j] = <a[k-m] b[k-m'] a[k-i] b[k-j]>

    See the notes on overleaf here:
    https://www.overleaf.com/project/5e2704c502c5d3000118937b

    Parameters
    ----------
    a : `numpy.ndarray`
        numpy array
    b : `numpy.ndarray`
        numpy channe
    c : `numpy.ndarray`
        numpy channel
    c : `numpy.ndarray`
        numpy channel
    M : `int`
        number of filter taps (a.k.a filter order)

    Returns
    -------
    arr : `numpy.ndarray'
        2D bi-correlation array of size (M+1) x (M+1)
    """

    if a.size != b.size or b.size != c.size or c.size != d.size:
        sys.exit("Array sizes do not match!")

    # pad and stack (makes rolling cleaner)
    a = np.pad(a, (M, M))
    b = np.pad(b, (M, M))
    c = np.pad(c, (M, M))
    d = np.pad(d, (M, M))

    # fill the correlation matrix
    out = np.zeros((2 * M + 1, 2 * M + 1, 2 * M + 1, 2 * M + 1))
    for j in range(-M, M + 1):
        for i in range(-M, M + 1):
            for mp in range(-M, M + 1):
                for m in range(-M, M + 1):
                    out[m, mp, i, j] = np.sum(
                        np.roll(a, m) * np.roll(b, mp) * np.roll(c, i) * np.roll(d, j)
                    )

    out = out[: M + 1, : M + 1, : M + 1, : M + 1]
    return out


def apply_weights_2d(wit1, wit2, a):
    """
    Convolve 2d filter coefficients and the
    input witness channels.
    """
    M = a.shape[0] - 1

    wit1 = np.pad(wit1, (M, 0), constant_values=0)
    wit2 = np.pad(wit2, (M, 0), constant_values=0)

    y = np.zeros(wit1.size - M)
    for ii in range(wit1.size - M):
        wit_mat = np.outer(wit1[ii : ii + (M + 1)][::-1], wit2[ii : ii + (M + 1)][::-1])
        y[ii] = np.tensordot(wit_mat, a)

    return y
