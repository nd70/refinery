import numpy as np
import scipy.signal as sig
import scipy.linalg as sl
import itertools


def xcorr(x, y, maxlag=None):
    """Calculate cross correlation of `x` and `y`, which have equal lengths.

    This function accepts a `maxlag` parameter, which truncates the result to
    only return the cross correlation of samples that are within `maxlag`
    samples of each other.

    For long input vectors, it is more efficient to calculate this convolution
    via FFT. As a rough heuristic, this function switches to an FFT based
    method when the input length exceeds 500 samples.

    Parameters
    ----------
    x : array_like
        First input array.
    y : array_like
        Second input array.

    Returns
    -------
    c : ndarray
        Cross correlation result.
    """
    xl = x.size
    yl = y.size
    if xl != yl:
        raise ValueError("x and y must be equal length")

    if maxlag is None:
        maxlag = xl - 1
    else:
        maxlag = int(maxlag)
    if maxlag >= xl or maxlag < 1:
        raise ValueError("maglags must be None or strictly positive")

    if xl > 500:  # Rough estimate of speed crossover
        c = sig.fftconvolve(x, y[::-1])
        c = c[xl - 1 - maxlag : xl + maxlag]
    else:
        c = np.zeros(2 * maxlag + 1)
        for i in range(maxlag + 1):
            c[maxlag - i] = np.correlate(
                x[0 : min(xl, yl - i)], y[i : i + min(xl, yl - i)]
            )
            c[maxlag + i] = np.correlate(
                x[i : i + min(xl - i, yl)], y[0 : min(xl - i, yl)]
            )
    return c


def block_levinson(y, L):
    """
    Solve the matrix equation T x = y for symmetric, block Toeplitz, T.

    Block Levinson recursion for efficiently solving symmetric block Toeplitz
    matrix equations. The matrix T is never stored in full (because it is
    large and mostly redundant), so the input parameter L is actually the
    leftmost "block column" of T (the leftmost d columns where d is the block
    dimension).

    References
    ----------
        Akaike, Hirotugu (1973). "Block Toeplitz Matrix Inversion".  SIAM J.
        Appl. Math. 24 (2): 234-241
    """
    d = L.shape[1]  # Block dimension
    N = int(L.shape[0] / d)  # Number of blocks

    # This gets the bottom block row B from the left block column L
    B = np.reshape(L, [d, N, d], order="F")
    B = B.swapaxes(1, 2)
    B = B[..., ::-1]
    B = np.reshape(B, [d, N * d], order="F")

    f = np.linalg.inv(L[:d, :])  # "Forward" vector
    b = f  # "Backward" vector
    x = np.dot(f, y[:d])  # Solution vector

    Ai = np.eye(2 * d)
    G = np.zeros((d * N, 2 * d))
    for n in range(2, N + 1):
        ef = np.dot(B[:, (N - n) * d : N * d], np.vstack((f, np.zeros((d, d)))))
        eb = np.dot(L[: n * d, :].T, np.vstack((np.zeros((d, d)), b)))
        ex = np.dot(B[:, (N - n) * d : N * d], np.vstack((x, np.zeros((d, 1)))))
        Ai[:d, d:] = eb
        Ai[d:, :d] = ef
        A = np.linalg.inv(Ai)
        ll = d * (n - 1)
        G[:ll, :d] = f
        G[d : ll + d, d:] = b
        fn = np.dot(G[: ll + d, :], A[:, :d])
        bn = np.dot(G[: ll + d, :], A[:, d:])
        f = fn
        b = bn
        x = np.vstack((x, np.zeros((d, 1)))) + np.dot(b, y[(n - 1) * d : n * d] - ex)

    W = x
    return W


def wiener_fir(tar, wit, N=8, method="brute"):
    """
    Calculate the optimal FIR Wiener subtraction filter for multiple inputs.

    This function may use the Levinson-Durbin algorithm to greatly enhance the
    speed of calculation, at the expense of instability when given highly
    coherence input signals. Brute-force inversion is available as an
    alternative.

    Parameters
    ----------
    tar : array_like
        Time series of target signal.
    wit : list of 1D arrays, or MxN array
        List of the time series of M witness signals, each witness must have
        the same length as the target signal, N.
    N : integer
        FIR filter order to be used. The filter response time is given by the
        product N * fs, where fs is the sampling frequency of the input
        signals.
    method : { 'levinson', 'brute' }, optional
        Selects the matrix inversion algorithm to be used. Defaults to
        'levinson'.

    Returns
    -------
    W : ndarray, shape (N, M)
        Columns of FIR filter coefficents that optimally estimate the target
        signal from the witness signals.
    """

    method = method.lower()
    if method not in ["levinson", "brute"]:
        raise ValueError("Unknown method type")

    N = int(N)
    if isinstance(wit, np.ndarray):
        if len(wit.shape) == 1:
            wit = np.reshape(wit, (1, wit.size), order="A")
        M = wit.shape[0]
    elif isinstance(wit, list):
        M = len(wit)
        wit = np.vstack([w for w in wit])

    P = np.zeros(M * (N + 1))
    # Cross correlation
    for m in range(M):
        top = m * (N + 1)
        bottom = (m + 1) * (N + 1)
        p = xcorr(tar, wit[m, :], N)
        P[top:bottom] = p[N : 2 * N + 1]

    if method.lower() == "levinson":
        P = np.reshape(P, [N + 1, M], order="F")
        P = np.reshape(P.T, [M * (N + 1), 1], order="F")
        R = np.zeros((M * (N + 1), M))
        for m in range(M):
            for ii in range(m + 1):
                rmi = xcorr(wit[m, :], wit[ii, :], N)
                Rmi = np.flipud(rmi[: N + 1])
                top = m * (N + 1)
                bottom = (m + 1) * (N + 1)
                R[top:bottom, ii] = Rmi
                if ii != m:
                    Rmi = rmi[N:]
                    top = ii * (N + 1)
                    bottom = (ii + 1) * (N + 1)
                    R[top:bottom, m] = Rmi

        R = np.reshape(R, [N + 1, M, M], order="F")
        R = R.swapaxes(0, 1)
        R = np.reshape(R, [M * (N + 1), M], order="F")

        W = block_levinson(P, R)

        #  Return each witness' filter as a row
        W = np.reshape(W, [M, N + 1], order="F")

    elif method.lower() == "brute":
        R = np.zeros((M * (N + 1), M * (N + 1)))
        for m in range(M):
            for ii in range(m, M):
                rmi = xcorr(wit[m, :], wit[ii, :], N)
                Rmi = sl.toeplitz(np.flipud(rmi[: N + 1]), rmi[N : 2 * N + 1])
                top = m * (N + 1)
                bottom = (m + 1) * (N + 1)
                left = ii * (N + 1)
                right = (ii + 1) * (N + 1)
                R[top:bottom, left:right] = Rmi
                if ii != m:
                    R[left:right, top:bottom] = Rmi.T
        W = np.linalg.solve(R, P)

        #  Return each witness' filter as a row
        W = np.reshape(W, [M, N + 1])

    return W


def cross_corr(x, y, M=32):
    """
    channel cross correlation

    Parameters
    ----------
    x : `numpy.ndarray`
        channel 1
    y : `numpy.ndarray`
        channel 2
    M : `int`
        number of filter taps
    """
    xl = x.size
    xc = sig.fftconvolve(x, y[::-1])
    xc = xc[xl - 1 - M : xl + M]
    return xc


def extended_wf(tar, wits, M):
    """
    Generalized wiener filter which takes advantage
    of the cross-correlations of the witness channels
    as well as the auto-correlations

    Parameters
    ----------
    tar : `numpy.ndarray`
        target signal
    wit : `numpy.ndarray`
        input signals (chans x lenght)
    M : `int`
        Filter order

    Returns
    -------
    W : `numpy.ndarray`
        filter coefficients (chans x coeffs)
    """
    if len(wits.shape) == 1:
        wits = np.reshape(wits, (1, wits.size))

    chans = wits.shape[0]
    combos = itertools.product(np.arange(chans), np.arange(chans))

    # cross correlation vector
    P = np.zeros((M + 1) * chans)
    for ii in range(chans):
        p = cross_corr(tar, wits[ii, :], M)
        P[ii * (M + 1) : (ii + 1) * (M + 1)] = p[M : 2 * M + 1]

    # correlation matrix
    R = np.zeros(((M + 1) * chans, (M + 1) * chans))
    for combo in combos:
        a, b = combo
        corr = cross_corr(wits[a, :], wits[b, :], M)
        toep = sl.toeplitz(np.flipud(corr[: (M + 1)]), corr[M : 2 * M + 1])
        R[a * (M + 1) : (a + 1) * (M + 1), b * (M + 1) : (b + 1) * (M + 1)] = toep

    W = np.linalg.solve(R, P)
    W = np.reshape(W, [chans, (M + 1)])
    return W


def miso_wiener_fir(tar, wits, M):
    """
    Multiple-in single-out wiener filter

    Parameters
    ----------
    tar : `numpy.ndarray`
        target signal
    wit : `numpy.ndarray`
        input signals (chans x lenght)
    M : `int`
        Filter order

    Returns
    -------
    W : `numpy.ndarray`
        filter coefficients (chans x coeffs)
    """
    chans = wits.shape[0]

    # cross correlation vector
    P = np.zeros((M + 1) * chans)
    for ii in range(chans):
        p = cross_corr(tar, wits[ii, :], M)
        P[ii * (M + 1) : (ii + 1) * (M + 1)] = p[M : 2 * M + 1]

    # correlation matrix
    R = np.zeros(((M + 1) * chans, (M + 1) * chans))
    for ii in range(chans):
        corr = cross_corr(wits[ii, :], wits[ii, :], M)
        toep = sl.toeplitz(np.flipud(corr[: (M + 1)]), corr[M : 2 * M + 1])
        R[ii * (M + 1) : (ii + 1) * (M + 1), ii * (M + 1) : (ii + 1) * (M + 1)] = toep

    W = np.linalg.solve(R, P)
    W = np.reshape(W, [chans, (M + 1)])
    return W


def siso_wiener_fir(tar, wit, M=8):
    """
    Single-in single-out wiener filter

    Parameters
    ----------
    tar : `numpy.ndarray`
        target signal
    wit : `numpy.ndarray`
        input signal (1D)
    M : `int`
        Filter order

    Returns
    -------
    W : `numpy.ndarray`
        filter coefficients
    """
    # Cross correlation
    p = cross_corr(tar, wit, M)
    P = p[M : 2 * M + 1]

    # correlation matrix
    rmi = cross_corr(wit, wit, M)
    R = sl.toeplitz(np.flipud(rmi[: M + 1]), rmi[M : 2 * M + 1])

    # solve and return
    W = np.linalg.solve(R, P)

    return W


def wiener_filter_pipeline(tar, wits, M):
    """
    run the wiener filter over the input data

    Parameters
    ----------
    M : `int`
        filter tap length
    tar : `numpy.ndarray`
        target array
    wit : `numpy.ndarray`
        stacked (or single) witness channel shape = [chans, timesteps]

    Returns
    -------
    clean : `numpy.ndarray`
        cleaned estimate of the true signal
    """
    W = extended_wf(tar, wits, M)
    est = np.zeros(tar.size)

    if wits.ndim == 1:
        wits = wits.reshape((1, wits.size))

    for ii in range(W.shape[0]):
        est += sig.lfilter(W[ii, :], 1.0, wits[ii, :])
    clean = tar - est
    return clean
