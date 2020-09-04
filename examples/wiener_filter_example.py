import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-colorblind')
import numpy as np
import scipy.signal as sig
import scipy.io as sio


#-------------------------------------------------------------------------------
#                           Generate / Load Data
#-------------------------------------------------------------------------------
# load mat file
mf = sio.loadmat('/home/rich.ormiston/refinery/scratch/H1_data_array.mat')
fs = int(mf['fsample'])
data = mf['data']

# cut down the dataset
dur = 1024  # seconds
data = data[:, :fs*dur]

# bandpass
low, high = 32, 42
data = lib.butter_filter(data, lowcut=low, highcut=high, btype='bandpass', fs=fs)

# split darm and witnesses
darm = data[0, :]
wits = data[1:, :]

#-------------------------------------------------------------------------------
#                                Run the Filter
#-------------------------------------------------------------------------------
# Run SISO WF filter only over unique PCAL channels
chans = [18, 20, 21]
clean = np.copy(darm)
est = np.zeros_like(darm)
for ii in chans:
    Ws = lib.siso_wiener_fir(darm, wits[ii, :], M=1)
    est += sig.lfilter(Ws, 1.0, wits[ii, :])
clean = darm - est

# Run Extended WF over the whole channel list
W0 = lib.extended_wf(darm, wits, M=fs)
est0 = np.zeros_like(darm)
for i in range(wits.shape[0]):
    est0 += sig.lfilter(W0[i, :], 1.0, wits[i, :])
clean0 = darm - est0

#-------------------------------------------------------------------------------
#                                  Visualize
#-------------------------------------------------------------------------------
print('making plots')
# PSDs
f, darm_psd = sig.welch(darm, fs=fs, nperseg=len(darm)//8)
_, clean_psd = sig.welch(clean, fs=fs, nperseg=len(darm)//8)
_, clean0_psd = sig.welch(clean0, fs=fs, nperseg=len(darm)//8)

fig, ax = plt.subplots(2, 1, sharex=True, sharey=False, figsize=(10, 6),
                       gridspec_kw={'height_ratios': [2,1]})

ax[0].semilogy(f, darm_psd, label='DARM')
ax[0].semilogy(f, clean_psd, label='SISO WF')
ax[0].semilogy(f, clean0_psd, label='Extended WF')
ax[0].set_xlim([0, 1.25])
lidx = np.where(f >= 0)[0][0]
hidx = np.where(f <= 1.25)[0][-1]
lf = np.min(clean_psd[lidx:hidx])
hf = np.max(darm_psd[lidx:hidx])
ax[0].set_ylim([lf, hf*2])
ax[0].legend()
ax[0].set_ylabel('Strain [$1/\sqrt{Hz}$]')
ax[0].set_title('Wiener Filter Comparison')
ax[0].grid(True)

ax[1].semilogy(f, np.ones_like(f), ls='--', lw=1.0)
ax[1].semilogy(f, clean0_psd/darm_psd, label='Extended_WF/DARM')
ax[1].semilogy(f, clean_psd/darm_psd, label='SISO_WF/DARM')
lf = np.min((clean_psd/darm_psd)[lidx:hidx])
hf = np.max((clean0_psd/darm_psd)[lidx:hidx])
ax[1].set_ylim([1e-1, 1e3])
ax[1].set_xlabel('Frequency [Hz]')
ax[1].legend(loc='lower right')
ax[1].grid(True)

plt.tight_layout()
plt.savefig('ext_wf_psd.png')
plt.close()

# Timeseries
fig, ax = plt.subplots(2, 1, sharex=True, sharey=False, figsize=(10, 6))
t = np.linspace(0, 2, 2*fs)
ax[0].plot(t, darm[-10*fs:-8*fs], label='DARM')
ax[0].plot(t, est[-10*fs:-8*fs], label='WF')
ax[0].set_title('WF Comparison')
ax[0].legend()
ax[0].grid(True)

ax[1].plot(t, darm[-10*fs:-8*fs], label='DARM')
ax[1].plot(t, est0[-10*fs:-8*fs], label='Extended WF')
ax[1].legend()
ax[1].set_xlabel('Time [s]')
ax[1].grid(True)

plt.tight_layout()
plt.savefig('ext_wf_ts.png')
plt.close()
