from __future__ import division
import numpy as np
import scipy.signal as sig
import scipy.linalg as sl
import scipy.fftpack as sc
import itertools
import nds2
import sys


def butter_filter(dataset,
                 low   = 4.0,
                 high  = 20.0,
                 order = 8,
                 btype = 'bandpass',
                 fs    = 512):
    """
    Phase preserving filter (bandpass, lowpass or highpass)
    based on a butterworth filter

    Parameters
    ----------
    dataset : `numpy.ndarray`
        input data (chans x lenght) where chan==0 is target
    lowcut : `float`
        low knee frequency
    highcut : `float`
        high knee frequency
    order : `int`
        filter order
    btype : `str`
        type of filter (bandpass, highpass, lowpass)
    fs : `int`
        sample rate

    Returns
    -------
    dataset : `numpy.ndarray`
        filtered dataset
    """

    # Normalize the frequencies
    nyq  = 0.5 * fs
    low  /= nyq
    high /= nyq

    # Make and apply filter
    if 'high' in btype:
        z, p, k = sig.butter(order, low, btype=btype, output='zpk')
    elif 'band' in btype:
        z, p, k = sig.butter(order, [low, high], btype=btype, output='zpk')
    elif 'low' in btype:
        z, p, k = sig.butter(order, high, btype=btype, output='zpk')
    sos = sig.zpk2sos(z, p, k)

    if dataset.ndim == 2:
        for i in range(dataset.shape[0]):
            dataset[i, :] = sig.sosfiltfilt(sos, dataset[i, :])
    else:
        dataset = sig.sosfiltfilt(sos, dataset)

    return dataset


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
        raise ValueError('x and y must be equal length')

    if maxlag is None:
        maxlag = xl - 1
    else:
        maxlag = int(maxlag)
    if maxlag >= xl or maxlag < 1:
        raise ValueError('maglags must be None or strictly positive')

    if xl > 500:  # Rough estimate of speed crossover
        c = sig.fftconvolve(x, y[::-1])
        c = c[xl - 1 - maxlag:xl + maxlag]
    else:
        c = np.zeros(2*maxlag + 1)
        for i in range(maxlag+1):
            c[maxlag-i] = np.correlate(x[0:min(xl, yl-i)],
                                       y[i:i+min(xl, yl-i)])
            c[maxlag+i] = np.correlate(x[i:i+min(xl-i, yl)],
                                       y[0:min(xl-i, yl)])
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
    d = L.shape[1]               # Block dimension
    N = int(L.shape[0]/d)        # Number of blocks

    # This gets the bottom block row B from the left block column L
    B = np.reshape(L, [d, N, d], order='F')
    B = B.swapaxes(1, 2)
    B = B[..., ::-1]
    B = np.reshape(B, [d, N*d], order='F')

    f = np.linalg.inv(L[:d, :])  # "Forward" vector
    b = f                        # "Backward" vector
    x = np.dot(f, y[:d])         # Solution vector

    Ai = np.eye(2*d)
    G = np.zeros((d*N, 2*d))
    for n in range(2, N+1):
        ef = np.dot(B[:, (N-n)*d:N*d], np.vstack((f, np.zeros((d, d)))))
        eb = np.dot(L[:n*d, :].T, np.vstack((np.zeros((d, d)), b)))
        ex = np.dot(B[:, (N-n)*d:N*d], np.vstack((x, np.zeros((d, 1)))))
        Ai[:d, d:] = eb
        Ai[d:, :d] = ef
        A = np.linalg.inv(Ai)
        l = d*(n-1)
        G[:l, :d] = f
        G[d:l+d, d:] = b
        fn = np.dot(G[:l+d, :], A[:, :d])
        bn = np.dot(G[:l+d, :], A[:, d:])
        f = fn
        b = bn
        x = np.vstack((x, np.zeros((d, 1)))) + np.dot(b, y[(n-1)*d:n*d]-ex)

    W = x
    return W


def wiener_fir(tar, wit, N=8, method='brute'):
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
    if method not in ['levinson', 'brute']:
        raise ValueError('Unknown method type')

    N = int(N)
    if isinstance(wit, np.ndarray):
        if len(wit.shape) == 1:
            wit = np.reshape(wit, (1, wit.size), order='A')
        M = wit.shape[0]
    elif isinstance(wit, list):
        M = len(wit)
        wit = np.vstack([w for w in wit])

    P = np.zeros(M*(N+1))
    # Cross correlation
    for m in range(M):
        top = m * (N+1)
        bottom = (m+1) * (N+1)
        p = xcorr(tar, wit[m, :], N)
        P[top:bottom] = p[N:2*N+1]

    if method.lower() == 'levinson':
        P = np.reshape(P, [N+1, M], order='F')
        P = np.reshape(P.T, [M*(N+1), 1], order='F')
        R = np.zeros((M*(N+1), M))
        for m in range(M):
            for ii in range(m+1):
                rmi = xcorr(wit[m, :], wit[ii, :], N)
                Rmi = np.flipud(rmi[:N+1])
                top = m * (N+1)
                bottom = (m+1) * (N+1)
                R[top:bottom, ii] = Rmi
                if ii != m:
                    Rmi = rmi[N:]
                    top = ii * (N+1)
                    bottom = (ii+1) * (N+1)
                    R[top:bottom, m] = Rmi

        R = np.reshape(R, [N+1, M, M], order='F')
        R = R.swapaxes(0, 1)
        R = np.reshape(R, [M*(N+1), M], order='F')

        W = block_levinson(P, R)

        #  Return each witness' filter as a row
        W = np.reshape(W, [M, N+1], order='F')

    elif method.lower() == 'brute':
        R = np.zeros((M*(N+1), M*(N+1)))
        for m in range(M):
            for ii in range(m, M):
                rmi = xcorr(wit[m, :], wit[ii, :], N)
                Rmi = sl.toeplitz(np.flipud(rmi[:N+1]), rmi[N:2*N+1])
                top = m * (N+1)
                bottom = (m+1) * (N+1)
                left = ii * (N+1)
                right = (ii+1) * (N+1)
                R[top:bottom, left:right] = Rmi
                if ii != m:
                    R[left:right, top:bottom] = Rmi.T
        W = np.linalg.solve(R, P)

        #  Return each witness' filter as a row
        W = np.reshape(W, [M, N+1])

    return W


def stream_data(start, channels,
                dur  = 600,
                fsup = 512,
                ifo  = 'H1'):
    """
    Collect LIGO data using nds2

    Parameters
    ----------
    start : `int`
        GPS start time (UTC)
    channels : `list`
        channel names to scrape
    dur : `int`
        duration (in seconds) of data to scrape
    fsup : `int`
        sample rate to return data with
    ifo : `str`
        Interferometer ('H1' or 'L1')

    Returns
    -------
    vdata : `numpy.ndarray`
        array of collected data
    """

    import nds2
    if ifo == 'H1':
        server = 'nds.ligo-wa.caltech.edu'
    else:
        server = 'nds.ligo-la.caltech.edu'

    # Setup connection to the NDS
    conn = nds2.connection(server, 31200)
    data = []
    for i in range(len(channels)):
        temp = conn.fetch(start, start + dur, [channels[i]])
        data.append(temp)

    # Get the data and stack it (the data are the columns)
    vdata = []
    for k in range(len(channels)):
        fsdown = data[k][0].channel.sample_rate
        down_factor = int(fsdown // fsup)

        fir_aa = sig.firwin(20 * down_factor + 1, 0.8 / down_factor,
                            window='blackmanharris')

        # Using fir_aa[1:-1] cuts off a leading and trailing zero
        downdata = sig.decimate(data[k][0].data, down_factor,
                                ftype = sig.dlti(fir_aa[1:-1], 1.0),
                                zero_phase = True)
        vdata.append(downdata)

    return np.array(vdata)


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
    xc = xc[xl - 1 - M: xl + M]
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
    P = np.zeros((M+1) * chans)
    for ii in range(chans):
        p = cross_corr(tar, wits[ii, :], M)
        P[ii*(M+1):(ii+1)*(M+1)] = p[M:2*M+1]

    # correlation matrix
    R = np.zeros(((M+1)*chans, (M+1)*chans))
    for combo in combos:
        a, b = combo
        corr = cross_corr(wits[a, :], wits[b, :], M)
        toep = sl.toeplitz(np.flipud(corr[:(M+1)]), corr[M:2*M+1])
        R[a*(M+1):(a+1)*(M+1), b*(M+1):(b+1)*(M+1)] = toep

    W = np.linalg.solve(R, P)
    W = np.reshape(W, [chans, (M+1)])
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
    P = np.zeros((M+1) * chans)
    for ii in range(chans):
        p = cross_corr(tar, wits[ii, :], M)
        P[ii*(M+1):(ii+1)*(M+1)] = p[M:2*M+1]

    # correlation matrix
    R = np.zeros(((M+1)*chans, (M+1)*chans))
    for ii in range(chans):
        corr = cross_corr(wits[ii, :], wits[ii, :], M)
        toep = sl.toeplitz(np.flipud(corr[:(M+1)]), corr[M:2*M+1])
        R[ii*(M+1):(ii+1)*(M+1), ii*(M+1):(ii+1)*(M+1)] = toep

    W = np.linalg.solve(R, P)
    W = np.reshape(W, [chans, (M+1)])
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
    P = p[M:2*M+1]

    # correlation matrix
    rmi = cross_corr(wit, wit, M)
    R = sl.toeplitz(np.flipud(rmi[:M+1]), rmi[M:2*M+1])

    # solve and return
    W = np.linalg.solve(R, P)

    return W


def rls(d, x, M=4, L=0.90):
    """
    Recursive least squares adaptive filter

    Parameters
    ----------
    d : `numpy.ndarray`
        target array
    x : `numpy.ndarray`
        input channel
    M : `int`
        filter order
    L : `float`
        Forgetting factor

    Returns
    -------
    y : `numpy.ndarray`
        bilinear noise estimate
    """
    inv_L = 1/L
    w = np.random.rand(M+1)
    P = np.eye(M+1)*0.01
    y = np.zeros_like(d)
    e = np.zeros_like(d)

    for k in range(M+1, d.size):
        xx = x[k-(M+1):k]
        y[k] = np.dot(w, xx)
        e[k] = d[k] - y[k]
        r = inv_L * np.dot(P, xx)
        g = r / (1 + np.dot(xx, r))
        w = w + g * e[k]
        P = inv_L * P - np.outer(g, r)

    return y


def nlms(d, x, M=1, mu=0.01, psi=0, leak=0):
    """
    Leaky, normalized LMS adaptive filter extended to
    analyze multiple inputs at a time

    Parameters
    ----------
    d : `numpy.ndarray`
        target array
    x : `numpy.ndarray`
        input channel
    M : `int`
        filter order
    mu : `float`
        step size
    psi : `float`
        stability factor when input power is low
    leak : `float`
        forgetting factor for weight updates

    Returns
    -------
    y : `numpy.ndarray`
        bilinear noise estimate
    """

    if len(x.shape) == 1:
        x = np.reshape(x, (1, x.size))

    estimate = np.zeros(x.shape[1])

    for ii in range(x.shape[0]):
        # initialize
        w = np.random.rand(M+1)
        e = np.zeros_like(d)
        y = np.zeros_like(d)
        xf = x[ii, :]

        if mu == None:
            eig = np.max(np.dot(xf, xf))
            mu = 1 / (4 * eig)

        # run the filter
        for k in range(M+1, d.size):
            xx = xf[k-(M+1):k]
            y[k] = np.dot(w, xx)
            e[k] = d[k] - y[k]
            w = w*(1-leak) + mu*e[k]*xx/(psi+np.dot(xx,xx))

        estimate += y

    return estimate


def soaf(d, x, M=1, mu=0.01, beta = 0.01):
    """
    Self-orthogonalizing adaptive filter. Essentially,
    this is an LMS filter where the input signal is
    first cosine transformed

    Parameters
    ----------
    d : `numpy.ndarray`
        target array
    x : `numpy.ndarray`
        input channel
    M : `int`
        filter order
    mu : `float`
        step size
    beta : `float`
        power forgetting factor (1 = no memory)

    Returns
    -------
    y : `numpy.ndarray`
        bilinear noise estimate
    """
    y = np.zeros_like(d)
    e = np.zeros_like(d)
    w = np.random.rand(M+1)

    for k in range(M+1, d.size):
        xx = x[k-(M+1):k]
        u = sc.dct(xx)
        if k == M+1:
            power = np.dot(u, u)
        y[k] = np.dot(w, u)
        power = (1-beta)*power + beta * np.dot(u, u)
        inv_sqrt_power = 1/(np.sqrt(power + 1e-5))
        e[k] = d[k] - y[k]
        w = w + mu*e[k]*inv_sqrt_power * u

    return y


def nlms_2nd_order(d, x1, x2, M=1, mu=1e-3):
    """
    Second order adaptive filter. Updates are based
    upon the outer product of the input signals. The
    format is that in an LMS filter.

    Parameters
    ----------
    d : `numpy.ndarray`
        target array
    x1 : `numpy.ndarray`
        input channel 1
    x2 : `numpy.ndarray`
        input channel 2
    M : `int`
        filter order
    mu : `float`
        step size

    Returns
    -------
    y : `numpy.ndarray`
        bilinear noise estimate
    """
    # initialize
    w = np.random.rand(M+1, M+1)
    e = np.zeros_like(d)
    y = np.zeros_like(d)

    if mu == None:
        eig = np.max([np.dot(x1, x1), np.dot(x2, x2)])
        mu = 1 / (4 * eig)

    # run the filter
    for k in range(M+1, d.size):
        xx1 = x1[k-(M+1):k]
        xx2 = x2[k-(M+1):k]
        y[k] = np.dot(xx1, np.dot(w, xx2))
        e[k] = d[k] - y[k]
        w = w + 2 * mu * e[k] * np.outer(xx1, xx2)

    return y


def nonlinear_AF(d, x1, x2, M=1, mu=1e-4):
    """
    Ordinary LMS adaptive filter with bilinear
    input.

    Parameters
    ----------
    d : `numpy.ndarray`
        target array
    x1 : `numpy.ndarray`
        input channel 1
    x2 : `numpy.ndarray`
        input channel 2
    M : `int`
        filter order
    mu : `float`
        step size

    Returns
    -------
    y : `numpy.ndarray`
        bilinear noise estimate
    """
    # initialize
    w = np.random.rand((M+1)**2)
    e = np.zeros_like(d)
    y = np.zeros_like(d)

    if mu == None:
        eig = np.max([np.dot(x1, x1), np.dot(x2, x2)])
        mu = 1 / (4 * eig)

    # run the filter
    for k in range(M+1, d.size):
        x = np.flipud(np.matrix.flatten(np.outer(x1[k-(M+1):k], x2[k-(M+1):k]).T))
        y[k] = np.dot(w, x)
        e[k] = d[k] - y[k]
        w = w + 2 * mu * e[k] * x

    return y


def whiten(data, fid_dat, time=60, fs=512):
    """
    whiten data using a fiducial time. length of data
    must be at least 2 * time [seconds] long

    Parameters
    ----------
    data : `numpy.ndarray`
        data array to be whitened
    fid_dat : `numpy.ndarray`
        fiducal data for the whitened array
    time : `int`
        number of seconds to be whitened
    fs : `int`
        sample rate

    Returns
    -------
    white_dat : `numpy.ndarray`
        whitened data
    """

    # get the data we want (dat)
    dat = data[:fid_dat.size]

    # get the FFT and PSD (offset the windows so we can dewhiten later)
    win = np.hanning(len(dat))
    win[0] += 1e-6
    win[-1] += 1e-6
    wf = np.mean(win**2)
    dfft = np.fft.rfft(dat*win) / np.sqrt(wf)
    _, dpsd = sig.welch(fid_dat, fs=fs, nperseg=len(fid_dat))

    # whiten FFT with the ASD and normalize by 1/fs
    norm = 1/(fs)
    whitened = norm * (dfft / np.sqrt(dpsd))
    white_dat = np.fft.irfft(whitened, n=len(dat))

    return white_dat


def dewhiten(white_dat, fid_dat, time, fs):
    """
    dewhiten data using a fiducial time. length of
    data must be at least 2 * time [seconds] long

    Parameters
    ----------
    white_dat : `numpy.ndarray`
        data array to be de-whitened
    fid_dat : `numpy.ndarray`
        fiducal array used in whitening process. return
        value from whiten() function
    time : `int`
        number of seconds to be whitened
    fs : `int`
        sample rate

    Returns
    -------
    dat : `numpy.ndarray`
        de-whitened data
    """
    n = time * fs
    whitened = np.fft.rfft(white_dat, n=n)

    norm = 1/fs
    _, dpsd = sig.welch(fid_dat, fs=fs, nperseg=len(fid_dat))
    dfft = whitened * np.sqrt(dpsd) / norm

    win = np.hanning(n)
    win[0] += 1e-6
    win[-1] += 1e-6
    wf = np.mean(win**2)
    dat = np.fft.irfft(np.sqrt(wf) * dfft) / win

    return dat


def whiten_zpk(signal, fs, flow=0.3, fhigh=30, order=6):

    signal = np.array(signal)
    if len(signal.shape) == 2:
        axis = np.argmax(signal.shape)
    else:
        axis = -1

    zc = [-2*np.pi*flow]*(order//2)
    pc = [-2*np.pi*fhigh]*(order//2)
    kc = 1.0

    sysc = sig.ZerosPolesGain(zc, pc, kc)
    sysd = sysc.to_discrete(dt=1/float(fs), method='bilinear')

    sosW = sig.zpk2sos(sysd.zeros, sysd.poles, sysd.gain)
    signal = sig.sosfiltfilt(sosW, signal, axis=axis)

    return signal


def apply_weights_1d(wit, a):
    """
    Convolve filter coefficients and witness
    Same as scipy.signal.lfilter(a, 1.0, wit)
    """
    wsize = wit.size
    asize = a.size
    a = np.pad(a[::-1], (0, wsize-1), constant_values=0)
    wit = np.pad(wit, (asize-1, 0), constant_values=0)
    out = np.zeros(wsize)
    for ii in range(out.size):
        out[ii] = wit.dot(np.roll(a, ii))
    return out


def apply_weights_2d(wit1, wit2, a):
    """
    Convolve 2d filter coefficients and the
    input witness channels.
    """
    M = a.shape[0] - 1

    wit1 = np.pad(wit1, (M, 0), constant_values=0)
    wit2 = np.pad(wit2, (M, 0), constant_values=0)

    y = np.zeros(wit1.size-M)
    for ii in range(wit1.size-M):
        wit_mat = np.outer(wit1[ii:ii + (M+1)][::-1], wit2[ii:ii+(M+1)][::-1])
        y[ii] = np.tensordot(wit_mat, a)

    return y


def correlation(a,b,M):
    """
    Same an np.correlate(a,b) or cross_corr(a,b,M)
    """
    a = np.pad(a, M, constant_values=0)
    b = np.pad(b, (0, 2*M), constant_values=0)
    out = np.zeros(2*M+1)
    for ii in range(2*M+1):
        out[ii] = a.dot(np.roll(b,ii))
    return out


def two_point_correlation(a, b, M):

    if a.size != b.size:
        sys.exit('Array sizes do not match!')

    # pad and stack (makes rolling cleaner)
    a = np.pad(a, (M,M))
    b = np.pad(b, (M,M))

    # fill the correlation matrix
    out = np.zeros((2*M+1,2*M+1))
    for mp in range(-M,M+1):
        for m in range(-M,M+1):
            out[m, mp] = np.sum(np.roll(a, m) * np.roll(b, mp))

    out = out[:M+1, :M+1]
    return out


def apply_weights_2d(wit1, wit2, coeffs):
    """
    apply the optimal weights for the second order Volterra
    filter to the witness channels being applied to the filter.
    This convolution is the 2D analog of scipy.signal.lfilter()

    Parameters
    ----------
    wit1 : `numpy.ndarray`
        witness channel to attempt to regress
    wit2 : `numpy.ndarray`
        witness channel to attempt to regress
    coeffs : `numpy.ndarray`
        matrix of filter coefficients

    Returns
    -------
    y : `numpy.ndarray`
        optimal estimate of the bilinear noise in the system signal
        due to the input witness channels (independent of coupling)
    """
    M = coeffs.shape[0] - 1

    wit1 = np.pad(wit1, (M, 0), constant_values=0)
    wit2 = np.pad(wit2, (M, 0), constant_values=0)

    y = np.zeros(wit1.size-M)
    for ii in range(wit1.size-M):
        wit_mat = np.outer(wit1[ii:ii+(M+1)][::-1], wit2[ii:ii+(M+1)][::-1])
        y[ii] = np.tensordot(wit_mat, coeffs)

    return y


def align_phase(tar, est, shift=2):
    """
    high frequencies cause a lag in the adaptation of the
    filters. this function minimizes the mse to set the
    correct phase shift for the filter estimate

    Parameters
    ----------
    tar : `numpy.ndarray`
        target array
    est : `numpy.ndarray`
        cleaned estimate
    shift : `int`
        optional. max lag/lead to check

    Returns
    -------
    wit_shift : `numpy.ndarray`
        shifted estimate
    """

    phase_shift = 0
    for s in np.linspace(-shift, shift, shift*2 + 1, dtype=int):
        if s > 0:
            s_mse = np.sum(np.square(tar[s:] - est[:-s])) / len(tar[s:])
        if s < 0:
            s_mse = np.sum(np.square(tar[:s] - est[-s:])) / len(tar[:s])
        if s == 0:
            s_mse = np.sum(np.square(tar - est)) / len(tar)

        if s == -shift:
            mse = s_mse
        if s_mse < mse:
            phase_shift = s
            mse = s_mse

    # shift the estimate
    wit_shift = np.zeros_like(tar)
    if phase_shift > 0:
        wit_shift[phase_shift:] = est[:-phase_shift]
    if phase_shift < 0:
        wit_shift[:phase_shift] = est[-phase_shift:]
    if phase_shift == 0:
        wit_shift = est

    return wit_shift


def adaptive_filter(data, mu=0.1, M=3, c_ops=None, shift=2, doShift=False):
    """
    Parameters
    ----------
    data : `numpy.ndarray`
        vertically stacked data array (chans x timesteps)
        0th chan is the target
    mu : `float`
        step size
    M : `int`
        filter order (a.k.a filter taps)
    c_opt : `dict`
        dictionary of the optimal mu and M params
    shift : `int`
        optional. max lag/lead to check
    doShift : `bool`
        if True, run align_phase()

    Returns
    -------
    combined_est : `numpy.ndarray`
        estimate of the true signal
    """

    tar = data[0, :]
    wits = data[1:, :]
    N = len(tar)
    est = np.zeros_like(wits)
    e = np.zeros_like(wits)

    # filter one channel at a time (output is (pseudo)linear & nonstationary)
    for chan in range(wits.shape[0]):
        if c_ops != None:
            mu, M = c_ops[chan + 1]
        xx = np.zeros(M)

        # start with random weigts for each channel
        w = np.random.rand(M)

        # run the M-tap filter
        for k in range(M, N):
            xx = wits[chan, k-M:k]
            est[chan, k] = np.dot(w, xx)
            e[chan, k] = tar[k] - est[chan, k]
            w = w + 2 * mu * e[chan, k] * xx

        if doShift:
            # shift to account for filter lag
            est[chan, :] = align_phase(tar, est[chan, :], shift=shift)

    # get the combined estimate
    combined_est = np.zeros_like(tar)
    for chan in range(est.shape[0]):
        combined_est += est[chan, :]

    return combined_est


def find_optimal_params(system_data, mu=[], M=[]):
    """
    grid search to get the optimal number of taps and
    learning rate (using the adaptive_filter() function)
    given the current system

    Parameters
    ----------
    system_data : `numpy.ndarray`
        vertically stacked data array (chans x timesteps)
        0th chan is the target
    mu : `list`
        list of floats of learning rates to check
    M : `list`
        list of ints of filter taps to check

    Returns
    -------
    c_opt : `dict`
        dictionary of the optimal mu and M params

    """
    mse = np.inf
    d = system_data[0, :]
    opt_mu, opt_M = 0.001, 3
    _, df = sig.welch(system_data[0, :], fs=fs, nperseg=4*fs)
    c_ops = {}

    for c in range(1, system_data.shape[0]):
        data = np.vstack((system_data[0, :], system_data[c, :]))
        for test_mu in mu:
            for test_M in M:
                ce = adaptive_filter(data, mu=test_mu, M=test_M)
                _, fres = sig.welch(data[0, :]-ce, fs=fs, nperseg=4*fs)
                loop_res = np.sum(fres/df)
                if loop_res < mse:
                    opt_mu, opt_M = test_mu, test_M
                    mse = loop_res
        c_ops[c] = [opt_mu, opt_M]

    return c_ops


def read_chans(chan_file):
    """
    collect chans from a txt file

    Parameters
    ----------
    chan_file : `str`
        txt file to read channels from

    Returns
    -------
    chans : `list`
        list of strings (channel names)
    """
    with open(chan_file) as f:
        lines = f.readlines()
    chans = [x.strip('\n') for x in lines]
    return chans


def prepare_multi_wit(*pargs, M=1, lstm=False):
    """
    Input data arrays (first one must be the target array)
    and create a matrix with the correct structure for a
    neural network (in Keras w/ tensorflow) which takes M taps
    of lookback.

    Parameters
    ----------
    *pargs : `numpy.ndarray`
        data channel to analyze
    M : `int`
        filter tap length
    lstm : `bool`
        if True, the data will be prepared differently
        so that Keras may interpret the data and lookback
        correctly for an LSTM network

    Returns
    -------
    output : `numpy.ndarray`
        output data array with M-taps of lookback
    """
    if M == 0:
        print('WARNING: tried to set 0-dimensional channel')
        print('Reverting to M=1')
        M = 1

    if M >= pargs[0].shape[1]:
        print('WARNING: desired channel taps exceeds witness size')
        print('Falling back to M={}'.format(pargs[0].shape[1] - 1))
        M = pargs[0].shape[1] - 1

    wit_input = np.vstack((pargs))
    output = np.zeros((wit_input.shape[1] - (M - 1), M, wit_input.shape[0]))
    for r in range(output.shape[0]):
        output[r, :, :] = wit_input[:, r:r+M].T

    # flatten for dense network
    if not lstm:
        output = output.reshape(output.shape[0], output.shape[1] * output.shape[2])

    return output


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
        sys.exit('Array sizes do not match!')

    # pad and stack (makes rolling cleaner)
    a = np.pad(a, (M,M))
    b = np.pad(b, (M,M))
    c = np.pad(c, (M,M))
    d = np.pad(d, (M,M))

    # fill the correlation matrix
    out = np.zeros((2*M+1,2*M+1, 2*M+1, 2*M+1))
    for j in range(-M,M+1):
        for i in range(-M,M+1):
            for mp in range(-M,M+1):
                for m in range(-M,M+1):
                    out[m, mp, i, j] = np.sum(np.roll(a, m) * np.roll(b, mp)\
                            * np.roll(c, i) * np.roll(d, j))

    out = out[:M+1, :M+1, :M+1, :M+1]
    return out


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
        sys.exit('Array sizes do not match!')

    # pad and stack (makes rolling cleaner)
    a = np.pad(a, (M,M))
    b = np.pad(b, (M,M))
    c = np.pad(c, (M,M))

    # fill the correlation matrix
    out = np.zeros((2*M+1, 2*M+1))
    for i in range(-M,M+1):
        for mp in range(-M,M+1):
            out[mp, i] = np.sum(a * np.roll(b, mp)\
                    * np.roll(c, i))

    out = out[:M+1, :M+1].flatten()
    return out


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
    vc = out.reshape(((M+1)**2, (M+1)**2)).T
    weights = np.linalg.pinv(vc).dot(P)
    est = apply_weights_2d(wit1, wit2, weights.reshape(((M+1), (M+1))))
    clean = d - est
    return clean
