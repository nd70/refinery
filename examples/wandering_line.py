#! /usr/bin/python3
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.style.use('dark_pastel')
import numpy as np
import scipy.signal as sig
import scipy.io as sio
import library as lib
import itertools
import sys
import scipy.linalg as sl
from gwpy.timeseries import TimeSeries


#-------------------------------------------------------------------------------
#                                  Functions
#-------------------------------------------------------------------------------
def nlms(d, x, M=1, mu=0.01, psi=0, leak=0):
    estimate = np.zeros_like(x)

    # initialize
    w = np.random.rand(M+1)
    e = np.zeros_like(d)
    y = np.zeros_like(d)

    if mu == None:
        eig = np.max(np.dot(x, x))
        mu = 1 / (4 * eig)

    # run the filter
    for k in range(M+1, d.size):
        xx = x[k-(M+1):k]
        y[k] = np.dot(w, xx)
        e[k] = d[k] - y[k]
        w = w*(1-leak) + mu*e[k]*xx/(psi+np.dot(xx,xx))
        # w = w + mu*e[k]*xx

    estimate += y

    return estimate


#-------------------------------------------------------------------------------
#                         Collect Data and Run Filter
#-------------------------------------------------------------------------------
chan1 = 'H1:SQZ-OMC_TRANS_RF3_Q_NORM_DQ'
darm  = 'H1:GDS-CALIB_STRAIN'
st    = 1241098218
ifo   = 'H1'
fs    = 256
dur   = 720
M     = 32
low   = 75
high  = 105

# collect the data
darm = TimeSeries.get(darm, st, st+dur, nproc=4, verbose=False)
darm = darm.resample(fs)
darm = darm.value
darm_copy = np.copy(darm)
darm = butter_filter(darm, low=low, high=high, fs=fs)

data = TimeSeries.get(chan1, st, st+dur, nproc=4)
if data.sample_rate.value != fs:
    data = data.resample(fs)
wit = data.value
wit = butter_filter(wit, low=low, high=high, fs=fs)
wit = wit.reshape(dur*fs)

clean = darm_copy - nlms(darm, wit, M=1, leak=0.0, psi=1e-6, mu=0.5)

#-------------------------------------------------------------------------------
#                               Plot if Good Results
#-------------------------------------------------------------------------------
clean = clean[fs*10:]
darm = darm[fs*10:]
ts = TimeSeries(clean, sample_rate=fs)
specgram = ts.spectrogram(2, fftlength=1, overlap=.5)**(1/2.)
mn, mx = np.min(specgram.value), np.max(specgram.value)
plot = specgram.imshow(norm='log', vmin=mn/10, vmax=mx*10)
ax = plot.gca()
ax.set_yscale('log')
ax.set_ylim(10, 200)
ax.colorbar(label=r'Gravitational-wave amplitude [strain/$\sqrt{\mathrm{Hz}}$]')
plt.savefig('wandering.png')
plt.close()

dts = TimeSeries(darm_copy, sample_rate=fs)
specgram = dts.spectrogram(2, fftlength=1, overlap=.5)**(1/2.)
plot = specgram.imshow(norm='log', vmin=mn/10, vmax=mx*10)
ax = plot.gca()
ax.set_yscale('log')
ax.set_ylim(10, 200)
ax.colorbar(label=r'Gravitational-wave amplitude [strain/$\sqrt{\mathrm{Hz}}$]')
plt.savefig('wandering_darm.png')
plt.close()
