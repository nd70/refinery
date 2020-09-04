import numpy as np
import scipy.signal as sig
import nds2


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
