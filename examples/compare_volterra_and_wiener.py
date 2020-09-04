import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.style.use('gruvbox')
import numpy as np
import library as lib
import scipy.signal as sig


#-------------------------------------------------------------------------------
#                        Load Data and Making Testing Set
#-------------------------------------------------------------------------------
print('Loading data and creating testing sets')
fs, dur = 256, 512
channels = ['H1:PEM-CS_SEIS_LVEA_VERTEX_X_DQ', 'H1:PEM-CS_SEIS_LVEA_VERTEX_Y_DQ',
            'H1:PEM-CS_SEIS_LVEA_VERTEX_Z_DQ']

wits = lib.stream_data(1253370524, channels, dur=dur, fsup=fs, ifo='H1')
for chan in range(wits.shape[0]):
    wits[chan, :] -= np.mean(wits[chan, :])

# Generate Target (true) Signal
t = np.linspace(0, dur, dur * fs)
true = wits[2, :]

# Single Linear Witness Channel
d_single_lin_wit = true + wits[0, :]

# Multiple Linear Witness Channels
d_multi_lin_wit = true + wits[0, :] + wits[1, :]

# Bilinear Channels
coupled = wits[0, :] * wits[1, :]
norm_coup = coupled * np.max(true)/np.max(coupled)*2
d_bilin_wit = true + norm_coup

#-------------------------------------------------------------------------------
#                              Run the Filters
#-------------------------------------------------------------------------------
# Multiple Linear Witness Channels
print('Running the Wiener Filter')
wf_multi_lin = lib.wiener_filter_pipeline(d_multi_lin_wit, wits[:2, :], 1)

print('Running the Volterra Filter')
vf_bilin = lib.volterra_pipeline(d_bilin_wit, wits[0, :], wits[1, :], 1)

#-------------------------------------------------------------------------------
#                                  Visualize
#-------------------------------------------------------------------------------
nps = fs * 8
freq, tar = sig.welch(true, fs=fs, nperseg=nps)
_, d_multi = sig.welch(d_multi_lin_wit, fs=fs, nperseg=nps)
_, d_bilin = sig.welch(d_bilin_wit, fs=fs, nperseg=nps)
_, wf_ml = sig.welch(wf_multi_lin, fs=fs, nperseg=nps)
_, vf_bl = sig.welch(vf_bilin, fs=fs, nperseg=nps)

# Wiener Filter Results
fig, ax = plt.subplots(2, 1, sharex=True, sharey=False, figsize=(6, 6))
ax[0].semilogy(freq, d_multi, label='system')
ax[0].semilogy(freq, tar, label='target')
ax[0].semilogy(freq, wf_ml, label='WF Multi')
ax[0].set_title('Wiener Filter Results')
ax[0].set_ylabel('Scaled Strain')
ax[0].legend()
ax[0].grid(True)

ax[1].semilogy(freq, wf_ml/d_multi, label='WF Multi')
ax[1].semilogy(freq, np.ones(tar.size), ls='--')
ax[1].set_ylabel('Scaled Strain')
ax[1].set_xlabel('Frequency [Hz]')
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.savefig('weiner_filter_test_results.png')
plt.close()

# Volterra Filter Results
fig, ax = plt.subplots(2, 1, sharex=True, sharey=False, figsize=(6, 6))
ax[0].semilogy(freq, d_bilin, label='system')
ax[0].semilogy(freq, tar, label='target')
ax[0].semilogy(freq, vf_bl, label='VF Bilin')
ax[0].set_title('Volterra Filter Results')
ax[0].set_ylabel('Scaled Strain')
ax[0].legend()
ax[0].grid(True)

ax[1].semilogy(freq, vf_bl/d_bilin, label='VF Bilin')
ax[1].semilogy(freq, np.ones(tar.size), ls='--')
ax[1].set_ylabel('Scaled Strain')
ax[1].set_xlabel('Frequency [Hz]')
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.savefig('volterra_filter_test_results.png')
plt.close()
