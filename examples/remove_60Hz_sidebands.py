import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.style.use('gruvbox')
import numpy as np
import library as lib
import scipy.signal as sig
import scipy.io as sio
import itertools
import os


#-------------------------------------------------------------------------------
#                        Load Data and Make Testing Set
#-------------------------------------------------------------------------------
darm_chan = 'H1:GDS-CALIB_STRAIN'
pem_chan  = 'H1:PEM-CS_MAINSMON_EBAY_1_DQ'
asc_chans = ['H1:ASC-PRC1_Y_INMON',
             'H1:ASC-PRC2_P_INMON',
             'H1:ASC-SRC2_Y_INMON',
             'H1:ASC-DHARD_Y_INMON',
             'H1:ASC-INP1_P_INMON',
             'H1:ASC-INP1_Y_INMON',
             'H1:ASC-MICH_P_INMON',
             'H1:ASC-MICH_Y_INMON',
             'H1:ASC-PRC1_P_INMON',
             'H1:ASC-PRC2_Y_INMON',
             'H1:ASC-SRC1_P_INMON',
             'H1:ASC-SRC1_Y_INMON',
             'H1:ASC-SRC2_P_INMON',
             'H1:ASC-DHARD_P_INMON',
             'H1:ASC-CHARD_P_INMON',
             'H1:ASC-CHARD_Y_INMON',
             'H1:ASC-DSOFT_P_INMON',
             'H1:ASC-DSOFT_Y_INMON',
             'H1:ASC-CSOFT_P_INMON',
             'H1:ASC-CSOFT_Y_INMON']

fs_fast, fs_slow, dur = 512, 16, 2048
st = 1242457832
et = st + dur

if not os.path.isfile('darm.mat'):
    darm = lib.stream_data(st, [darm_chan], fs=fs_fast, dur=dur)
    darm = darm.reshape(darm.shape[1])
    sio.savemat('darm.mat', mdict={'data':darm})
else:
    darm = sio.loadmat('darm.mat')['data']
    darm = darm.reshape(darm.shape[1])

low, high = 56, 64
darm = lib.butter_filter(darm, low=low, high=high, fs=fs_fast)

if not os.path.isfile('pem.mat'):
    pem = lib.stream_data(st, [pem_chan], fs=fs_fast, dur=dur)
    pem = pem.reshape(pem.shape[1])
    sio.savemat('pem.mat', mdict={'data':pem})
else:
    pem = sio.loadmat('pem.mat')['data']
    pem = pem.reshape(pem.shape[1])

if not os.path.isfile('asc.mat'):
    asc = lib.stream_data(st, asc_chans, fs=fs_slow, dur=dur)
    sio.savemat('asc.mat', mdict={'data':asc})
else:
    asc = sio.loadmat('asc.mat')['data']

temp = np.zeros((asc.shape[0], dur*fs_fast))
for ix in range(1, temp.shape[0]):
    temp[ix, :] = sig.resample(asc[ix, :], dur*fs_fast)
asc = np.copy(temp); del temp

#-------------------------------------------------------------------------------
#                         Run the Filter && Visualize
#-------------------------------------------------------------------------------
for ix in range(asc.shape[0]):
    print('Running {0} of {1}'.format(ix+1, asc.shape[0]))
    ch2 = asc[ix, :]
    vf_bilin = lib.volterra_pipeline(darm, pem, ch2, 1)

    # plots
    nps = len(darm)//20
    freq, tar = sig.welch(darm, fs=fs_fast, nperseg=nps)
    _, vf_bl = sig.welch(vf_bilin, fs=fs_fast, nperseg=nps)
    vf_bl = vf_bl.reshape(len(freq))
    tar = tar.reshape(len(freq))

    # Volterra Filter Results
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=False, figsize=(6, 6))
    ax[0].semilogy(freq, tar, label='DARM')
    ax[0].semilogy(freq, vf_bl, label='{}'.format(asc_chans[ix]))
    ax[0].set_title('Volterra Filter Results')
    ax[0].set_ylabel('Scaled Strain')
    ax[0].set_ylim([1e-47, 1e-41])
    ax[0].legend()
    ax[0].grid(True)

    ax[1].semilogy(freq, vf_bl/tar, label='Bilinear Sub')
    ax[1].semilogy(freq, np.ones(tar.size), ls='--')
    ax[1].set_ylabel('Scaled Strain')
    ax[1].set_xlabel('Frequency [Hz]')
    ax[1].set_xlim([low, high])
    ax[1].set_ylim([np.min(vf_bl/tar)/5, 2])
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig('{0}-{1}.png'.format(pem_chan, asc_chans[ix]))
    plt.close()
