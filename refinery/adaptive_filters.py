import numpy as np
import scipy.signal as sig
import scipy.fftpack as sc


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


def bilinear_nlms(d, x1, x2, mu=1e-2, M=1, psi=1e-6):
    """
    LMS adaptive filter with bilinear input

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
    a = np.random.rand(M+1, M+1)
    y = np.zeros(d.size)
    e = np.zeros(d.size)

    for ix in range(M, d.size):
        xx1   = x1[ix-M:ix+1]
        xx2   = x2[ix-M:ix+1]
        y[ix] = np.tensordot(a, np.outer(xx1, xx2))
        e[ix] = d[ix] - y[ix]
        a = a + 2 * mu * e[ix] * np.outer(xx1, xx2) /\
                ((xx1.dot(xx1) + xx2.dot(xx2))/1 + psi)

    return y, e


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
