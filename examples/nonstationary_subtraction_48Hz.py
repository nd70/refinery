import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import scipy.io as sio
import os
import library as lib
from gwpy.timeseries import TimeSeries


# -------------------------------------------------------------------------------
#                                  Load Data
# -------------------------------------------------------------------------------
st = 1243834818
dur = 2048
fs_slow, fs_fast = 16, 256
low, high = 36, 58
M = 1

slow_chans_list = [
    "H1:ASC-CHARD_P_SMOOTH_INMON",
    "H1:ASC-DC1_P_INMON",
    "H1:ASC-ETMX_PIT_INMON",
    "H1:ASC-ETMY_PIT_INMON",
    "H1:ASC-OMCR_B_SUM_INMON",
    "H1:ASC-POP_A_SEG4_INMON",
    "H1:ASC-RM1_PIT_INMON",
    "H1:ASC-RPC_DSOFT_Y_INMON",
    "H1:ASC-Y_TR_A_YAW_INMON",
    "H1:ASC-Y_TR_B_SEG3_INMON",
    "H1:OMC-ASC_P2_Q_INMON",
    "H1:OMC-ASC_PD_IN_INMON",
    "H1:OMC-ASC_Y1_I_INMON",
    "H1:OMC-ASC_Y1_Q_INMON",
    "H1:OMC-ASC_Y1_X_COS_INMON",
    "H1:OMC-ASC_Y1_X_SIN_INMON",
    "H1:OMC-ASC_Y2_I_INMON",
    "H1:OMC-ASC_Y2_Q_INMON",
    "H1:PEM-CS_SEIS_LVEA_VERTEX_X_BLRMS_INPUT_INMON",
    "H1:PEM-CS_SEIS_LVEA_VERTEX_Y_BLRMS_INPUT_OUTPUT",
    "H1:PEM-CS_SEIS_LVEA_VERTEX_Z_BLRMS_INPUT_OUT16",
    "H1:PEM-CS_SEIS_LVEA_VERTEX_Z_BLRMS_INPUT_OUTPUT",
    "H1:PEM-EY_SEIS_VEA_FLOOR_X_BLRMS_INPUT_OUTPUT",
    "H1:PEM-MX_SEIS_VEA_FLOOR_Y_BLRMS_INPUT_INMON",
    "H1:PEM-MX_SEIS_VEA_FLOOR_Z_BLRMS_INPUT_OUTPUT",
    "H1:PEM-MY_SEIS_VEA_FLOOR_X_BLRMS_INPUT_OUTPUT",
    "H1:PEM-MY_SEIS_VEA_FLOOR_Y_BLRMS_INPUT_OUTPUT",
    "H1:PEM-MY_SEIS_VEA_FLOOR_Z_BLRMS_INPUT_OUTPUT",
    "H1:PEM-VAULT_SEIS_1030X195Y_STS2_X_BLRMS_INPUT_INMON",
    "H1:PEM-VAULT_SEIS_1030X195Y_STS2_X_BLRMS_INPUT_OUT16",
    "H1:PEM-VAULT_SEIS_1030X195Y_STS2_Y_BLRMS_INPUT_INMON",
    "H1:SQZ-ASC_ANG_Y_INMON",
]

fast_chans_list = [
    "H1:PEM-CS_ACC_HAM3_PR2_Y_DQ",
    "H1:HPI-HAM3_BLND_L4C_RZ_IN1_DQ",
    "H1:HPI-HAM3_BLND_L4C_X_IN1_DQ",
    "H1:HPI-HAM3_SENSCOR_X_LOCAL_DIFF_DQ",
    "H1:ISI-HAM3_BLND_GS13Y_IN1_DQ",
]

darm = ["H1:CAL-DELTAL_EXTERNAL_DQ"]

# get the slow, modulating witnesses
print("getting slow channels")
if os.path.isfile("48Hz_slow_chans.mat"):
    slow_chans = sio.loadmat("48Hz_slow_chans.mat")["data"]
else:
    slow_chans = lib.stream_data(st, slow_chans_list, dur=dur, fs=fs_slow)
    sio.savemat("48Hz_slow_chans.mat", mdict={"data": slow_chans})
slow_chans = sig.resample(slow_chans, fs_fast * dur, axis=-1)

# get the witnesses to the 48.5Hz line
print("getting fast channels")
if os.path.isfile("48Hz_fast_chans.mat"):
    fast_chans = sio.loadmat("48Hz_fast_chans.mat")["data"]
else:
    fast_chans = lib.stream_data(st, fast_chans_list, dur=dur, fs=fs_fast)
    sio.savemat("48Hz_fast_chans.mat", mdict={"data": fast_chans})

# get darm (DELTAL_EXTERNAL), and bandpass
print("getting darm")
if os.path.isfile("48Hz_darm.mat"):
    darm = sio.loadmat("48Hz_darm.mat")["data"]
else:
    darm = lib.stream_data(st, darm, dur=dur, fs=fs_fast)
    sio.savemat("48Hz_darm.mat", mdict={"data": darm})

darm_copy = np.copy(darm)
darm_copy = darm_copy.reshape(darm_copy.shape[1])
darm = lib.butter_filter(darm, low=low, high=high, fs=fs_fast)
darm = darm.reshape(darm.shape[1])

# -------------------------------------------------------------------------------
#                          Run the Filter && Visualize
# -------------------------------------------------------------------------------
print("running filter")
matplotlib.use("agg")
plt.style.use("dark_pastel")
for ix in range(fast_chans.shape[0]):
    fc = fast_chans[ix, :]
    fc = (fc - np.mean(fc)) / np.std(fc)
    est = lib.nlms(darm, fc * np.max(darm), M=32, mu=1.0, leak=0)
    clean = darm_copy - lib.butter_filter(est, low=low, high=high, fs=fs_fast, order=8)

    f, darm_psd = sig.welch(darm_copy, fs=fs_fast, nperseg=fs_fast * 16)
    _, clean_psd = sig.welch(clean, fs=fs_fast, nperseg=fs_fast * 16)
    darm_psd = darm_psd.reshape(len(f))
    clean_psd = clean_psd.reshape(len(f))

    fig, ax = plt.subplots(
        2,
        1,
        sharex=True,
        sharey=False,
        figsize=(10, 6),
        gridspec_kw={"height_ratios": [2, 1]},
    )

    ax[0].semilogy(f, darm_psd, label="CAL-DELTAL_EXTERNAL")
    ax[0].semilogy(f, clean_psd, label="NS Cleaned")
    ax[0].set_xlim([10, 90])
    ax[0].legend()
    ax[0].set_ylabel("PSD")
    ax[0].set_title("Nonstationary Subtraction: {}".format(fast_chans_list[ix]))
    ax[0].grid(True)

    ax[1].semilogy(f, np.ones_like(f), ls="--")
    ax[1].semilogy(f, clean_psd / darm_psd)
    ax[1].set_ylim([0.01, 10])
    ax[1].set_xlabel("Frequency [Hz]")
    ax[1].set_ylabel("Clean/Dirty ASD Ratio")
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig("ns/ns_subtraction_{}.png".format(ix))
    plt.close()

    cln = TimeSeries(clean, sample_rate=fs_fast, t0=st)
    specgram = cln.spectrogram(2, fftlength=1, overlap=0.9) ** (1 / 2.0)
    plot = specgram.imshow(norm="log")
    ax = plot.gca()
    ax.set_yscale("log")
    ax.set_ylim(10, 100)
    ax.set_title("Witness: {}".format(fast_chans_list[ix]))
    ax.colorbar(label=r"Gravitational-wave amplitude [strain/$\sqrt{\mathrm{Hz}}$]")
    plt.savefig("ns/ns_sub_spec_{}.png".format(fast_chans_list[ix]))
    plt.close()

dts = TimeSeries(darm_copy, sample_rate=fs_fast, t0=st)
specgram = dts.spectrogram(2, fftlength=1, overlap=0.9) ** (1 / 2.0)
plot = specgram.imshow(norm="log")
ax = plot.gca()
ax.set_yscale("log")
ax.set_ylim(10, 100)
ax.set_title("H1:CAL-DELTAL_EXTERNAL_DQ")
ax.colorbar(label=r"Gravitational-wave amplitude [strain/$\sqrt{\mathrm{Hz}}$]")
plt.savefig("ns/ns_darm_spec.png")
plt.close()
