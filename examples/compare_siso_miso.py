import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import library as lib


# -------------------------------------------------------------------------------
#                             Load and Trim Data
# -------------------------------------------------------------------------------
# Since we want coherent data, grab x,y and z seismic channels.
# Add these to random noise and inject a 22 Hz line
fs, dur = 128, 512
channels = [
    "H1:PEM-CS_SEIS_LVEA_VERTEX_X_DQ",
    "H1:PEM-CS_SEIS_LVEA_VERTEX_Y_DQ",
    "H1:PEM-CS_SEIS_LVEA_VERTEX_Z_DQ",
]
wits = lib.stream_data(1253370524, channels, dur=dur, fsup=fs, ifo="H1")
for chan in range(wits.shape[0]):
    wits[chan, :] -= np.mean(wits[chan, :])

fig, ax = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(8, 7.5))
f, xpsd = sig.welch(wits[0, :], fs=fs)
_, ypsd = sig.welch(wits[1, :], fs=fs)
_, zpsd = sig.welch(wits[2, :], fs=fs)

ff, xy = sig.coherence(wits[0, :], wits[1, :], fs=fs)
ff, xz = sig.coherence(wits[0, :], wits[2, :], fs=fs)
ff, yz = sig.coherence(wits[1, :], wits[2, :], fs=fs)

ax[0].semilogy(f, xpsd, label="$\hat{x}$")
ax[0].semilogy(f, ypsd, label="$\hat{y}$")
ax[0].semilogy(f, zpsd, label="$\hat{z}$")
ax[0].grid(True)
ax[0].legend()
ax[0].set_ylabel("Strain [$1/\sqrt{Hz}$]")

ax[1].semilogy(ff, xy, label="x-y")
ax[1].semilogy(ff, xz, label="x-z")
ax[1].semilogy(ff, yz, label="y-z")
ax[1].grid(True)
ax[1].legend()
ax[1].set_xlabel("Frequency [Hz]")
ax[1].set_ylabel("Coherence")

ax[2].plot(np.linspace(0, 5, 5 * fs), wits[0, -5 * fs :], label="x")
ax[2].plot(np.linspace(0, 5, 5 * fs), wits[1, -5 * fs :], label="y")
ax[2].plot(np.linspace(0, 5, 5 * fs), wits[2, -5 * fs :], label="z")
ax[2].grid(True)
ax[2].legend()
ax[2].set_xlabel("Time [s]")

plt.tight_layout()
plt.savefig("seis_example_psd.png")
plt.close()

# make signals
t = np.linspace(0, dur, dur * fs)
s = np.sin(2 * np.pi * 22 * t)
true = 50 * s + np.random.rand(len(t))
d = true + wits[0, :] + wits[1, :] + wits[2, :]

# -------------------------------------------------------------------------------
#                               Run the Filter
# -------------------------------------------------------------------------------
# Extended WF
W = lib.extended_wf(d, wits, 1)
est = np.zeros_like(d)
for ii in range(wits.shape[0]):
    est += sig.lfilter(W[ii, :], 1.0, wits[ii, :])
clean = d - est

# SISO WF
siso_est = np.zeros_like(d)
for ii in range(wits.shape[0]):
    w = lib.siso_wiener_fir(d, wits[ii, :], M=1)
    siso_est += sig.lfilter(w, 1.0, wits[ii, :])
siso_clean = d - siso_est

# -------------------------------------------------------------------------------
#                                  Visualize
# -------------------------------------------------------------------------------
matplotlib.use("agg")
plt.style.use("seaborn-colorblind")
print("Making plots")

# PSDs
freq, exwf = sig.welch(clean, fs=fs, nperseg=len(clean) // 8)
_, sisowf = sig.welch(siso_clean, fs=fs, nperseg=len(clean) // 8)
_, df = sig.welch(d, fs=fs, nperseg=len(clean) // 8)
_, tf = sig.welch(true, fs=fs, nperseg=len(clean) // 8)

mn = np.min([df, exwf, sisowf])
mx = np.max([df, exwf, sisowf])

fig, ax = plt.subplots(2, 1, sharex=True, sharey=False, figsize=(8, 9))
ax[0].semilogy(freq, df, label="DARM")
ax[0].semilogy(freq, exwf, label="Extended WF")
ax[0].semilogy(freq, sisowf, label="SISO WF", ls="--")
ax[0].semilogy(freq, tf, label="true signal")
ax[0].set_ylabel("Scaled Strain")
ax[0].set_ylim([mn / 2, mx * 2])
ax[0].set_xlim([20, 38])
ax[0].set_title("SISO vs MISO WF Estimate")
ax[0].legend()
ax[0].grid(True)

ratio = exwf / df
r2 = sisowf / df
mn = np.min([ratio, r2])
mx = np.max([ratio, r2])
ax[1].semilogy(freq, ratio, label="Extended_WF/DARM")
ax[1].semilogy(freq, r2, label="SISO_WF/DARM", ls="--")
ax[1].semilogy(freq, np.ones_like(freq), ls="--")
ax[1].set_ylim([mn / 2, mx * 2])
ax[1].set_xlabel("Frequency [Hz]")
ax[1].set_ylabel("Clean/$h(t)$ Ratio")
ax[1].legend(loc="best")
ax[1].grid(True)

plt.tight_layout()
plt.savefig("corr_seis_noise_wf.png")
plt.close()
