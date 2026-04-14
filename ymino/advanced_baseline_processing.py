import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from resonator_tools import circuit
%matplotlib inline

# =========================================
# settings ここだけ主に編集
# =========================================

# baseline parameters
niter = 100
lam_amp = 1e6
lam_phase = 1e6
p_amp = 0.01 # 0.75
p_phase = 0.01 # 0.5
base_correct = 0.985

# fit type
use_notch_port = False   # True にすると notch_port, False で reflection_port

# ---- mask range の指定方法 ----
# method = "index" なら配列インデックスで指定
# method = "freq" なら周波数で指定
mask_method = "index"

# mask_method = "index" のとき使う
mask_start_idx = 900
mask_end_idx = 1100   # end は含まない

# mask_method = "freq" のとき使う
mask_fmin = 7.2727e9
mask_fmax = 7.2731e9

# ---- fit range の指定方法 ----
fit_method = "index"

# fit_method = "index" のとき使う
fit_start_idx = 700
fit_end_idx = 1300    # end は含まない

# fit_method = "freq" のとき使う
fit_fmin = 7.2715e9
fit_fmax = 7.2742e9


# =========================================
# load data
# =========================================
port_new = circuit.reflection_port()
port_new.add_froms2p(
    str(data_path("s2210dBmm.s1p")), 1, 2, "realimag", fdata_unit=1e9, delimiter=None
)
freq = port_new.f_data
S11 = port_new.z_data_raw

amp = np.abs(S11)
phase = np.unwrap(np.angle(S11))

port_raw = circuit.reflection_port()
port_raw.add_data(freq, S11)


# =========================================
# make mask
# =========================================
if mask_method == "index":
    mask = np.ones_like(freq, dtype=bool)
    mask[mask_start_idx:mask_end_idx] = False

elif mask_method == "freq":
    mask = (freq < mask_fmin) | (freq > mask_fmax)

else:
    raise ValueError("mask_method must be 'index' or 'freq'")


# =========================================
# baseline用データ作成
# =========================================
amp_for_base = amp.copy()
phase_for_base = phase.copy()

amp_for_base[~mask] = np.interp(freq[~mask], freq[mask], amp[mask])
phase_for_base[~mask] = np.interp(freq[~mask], freq[mask], phase[mask])


# =========================================
# baseline fitting
# =========================================
baseline_amp = port_raw.fit_baseline_amp(amp_for_base, lam_amp, p_amp, niter=niter)
baseline_phase = port_raw.fit_baseline_phase(phase_for_base, lam_phase, p_phase, niter=niter)

baseline_amp_corr = baseline_amp * base_correct

S11_corr = (amp / baseline_amp_corr) * np.exp(1j * (phase - baseline_phase))


# =========================================
# fit range
# =========================================
if fit_method == "index":
    fit_slice = slice(fit_start_idx, fit_end_idx)
    freq_fit = freq[fit_slice]
    S11_fit = S11_corr[fit_slice]

elif fit_method == "freq":
    fit_sel = (freq >= fit_fmin) & (freq <= fit_fmax)
    freq_fit = freq[fit_sel]
    S11_fit = S11_corr[fit_sel]

else:
    raise ValueError("fit_method must be 'index' or 'freq'")


# =========================================
# resonator fit
# =========================================
if use_notch_port:
    port_fit = circuit.notch_port()
else:
    port_fit = circuit.reflection_port()

port_fit.add_data(freq_fit, S11_fit)
port_fit.autofit()


# =========================================
# plot helper for shaded ranges
# =========================================
f_mask_min = freq[~mask].min()
f_mask_max = freq[~mask].max()
f_fit_min = freq_fit.min()
f_fit_max = freq_fit.max()


# =========================================
# show baseline plots
# =========================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# raw amplitude + baseline
axes[0, 0].plot(freq, amp, label="raw amplitude")
axes[0, 0].plot(freq, amp_for_base, "--", label="amp used for baseline")
axes[0, 0].plot(freq, baseline_amp, label="baseline amp")
axes[0, 0].plot(freq, baseline_amp_corr, label=f"baseline amp x {base_correct}")
axes[0, 0].axvspan(f_mask_min, f_mask_max, alpha=0.15, color="gray", label="masked region")
axes[0, 0].set_title("Amplitude baseline fit")
axes[0, 0].set_xlabel("Frequency (Hz)")
axes[0, 0].set_ylabel("Amplitude")
axes[0, 0].legend()

# raw phase + baseline
axes[0, 1].plot(freq, phase, label="raw phase (unwrap)")
axes[0, 1].plot(freq, phase_for_base, "--", label="phase used for baseline")
axes[0, 1].plot(freq, baseline_phase, label="baseline phase")
axes[0, 1].axvspan(f_mask_min, f_mask_max, alpha=0.15, color="gray", label="masked region")
axes[0, 1].set_title("Phase baseline fit")
axes[0, 1].set_xlabel("Frequency (Hz)")
axes[0, 1].set_ylabel("Phase (rad)")
axes[0, 1].legend()

# corrected amplitude
axes[1, 0].plot(freq, np.abs(S11_corr), label="corrected amplitude")
axes[1, 0].axvspan(f_mask_min, f_mask_max, alpha=0.15, color="gray", label="masked region")
axes[1, 0].axvspan(f_fit_min, f_fit_max, alpha=0.10, color="orange", label="fit range")
axes[1, 0].set_title("Amplitude after baseline correction")
axes[1, 0].set_xlabel("Frequency (Hz)")
axes[1, 0].set_ylabel("Amplitude")
axes[1, 0].legend()

# corrected phase
axes[1, 1].plot(freq, np.unwrap(np.angle(S11_corr)), label="corrected phase")
axes[1, 1].axvspan(f_mask_min, f_mask_max, alpha=0.15, color="gray", label="masked region")
axes[1, 1].axvspan(f_fit_min, f_fit_max, alpha=0.10, color="orange", label="fit range")
axes[1, 1].set_title("Phase after baseline correction")
axes[1, 1].set_xlabel("Frequency (Hz)")
axes[1, 1].set_ylabel("Phase (rad)")
axes[1, 1].legend()

plt.tight_layout()
plt.show()


# =========================================
# show fit result
# =========================================
port_fit.plotall()
print(port_fit.fitresults)


# =========================================
# print settings summary
# =========================================
print("===== settings summary =====")
print(f"path           = {path}")
print(f"use_notch_port = {use_notch_port}")
print(f"lam_amp        = {lam_amp}")
print(f"lam_phase      = {lam_phase}")
print(f"p_amp          = {p_amp}")
print(f"p_phase        = {p_phase}")
print(f"base_correct   = {base_correct}")

if mask_method == "index":
    print(f"mask_method     = index")
    print(f"mask_start_idx  = {mask_start_idx}")
    print(f"mask_end_idx    = {mask_end_idx}")
else:
    print(f"mask_method     = freq")
    print(f"mask_fmin       = {mask_fmin}")
    print(f"mask_fmax       = {mask_fmax}")

if fit_method == "index":
    print(f"fit_method      = index")
    print(f"fit_start_idx   = {fit_start_idx}")
    print(f"fit_end_idx     = {fit_end_idx}")
else:
    print(f"fit_method      = freq")
    print(f"fit_fmin        = {fit_fmin}")
    print(f"fit_fmax        = {fit_fmax}")