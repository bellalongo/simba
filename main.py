#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# main.py  — corrected
"""
Swept stepped-LO spectral reconstruction simulation (simba).

Key differences from testin3.py
---------------------------------
  testin3   : dual-comb ptychoscopy, correlation C_n(omega)
  this file : single stepped LO, per-step FFT + sweep DFT

Master equation (non-overlapping regime):
    P_s(omega_n^(c) + omega_k)
      = |(1/N_sw) sum_r exp(-2pi i s_n r/N_sw)  S_tilde_r(omega_k,qn)|^2
        / |E0|^2

where the twiddle is NEGATIVE because S_tilde_r carries
exp(+2pi i s_n r/N_sw) from the signal carrier phase.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import gcd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x

from simba import Signal
from simba.local_oscillator import SweptSteppedLO
from simba.mixing import build_interferogram_single_lo
from simba.extraction import extract_single_sweep, extract_multi_sweep


# ---------------------------------------------------------------------------
#  Analytic lineshapes (identical to testin3.py for direct comparison)
# ---------------------------------------------------------------------------

def Lfcn(fv, f0, As, Ac, FW, Nt, dt):
    """Cauchy (Lorentzian) PSD, per-tooth DFT normalisation."""
    num = 2 * np.pi * dt * FW
    den = (np.pi * dt * FW) ** 2 + ((fv - f0) * 2 * np.pi * dt) ** 2
    return (1.0 / Nt) * (num / den) * np.abs(As) ** 2 * np.abs(Ac) ** 2


def Lfcn2(fv, f0, As, Ac, FW, Nt, dt):
    """
    Exact discrete Lorentzian (accounts for DFT periodicity / aliasing).

    Nt must be the number of samples in the DFT window used for extraction:
      - single-sweep or multi-sweep:  Nt = Nt_per_step  (= Nps)
      After the sweep DFT, the amplitude equals one sweep's FFT value,
      so the normalisation is still per-step.

    Consistency check: at resonance fv == f0,
      peak = |As|^2 |Ac|^2 / (pi * dt * FW * Nt)  ~ signal PSD.
    """
    num  = np.sinh(np.pi * dt * FW)
    den  = np.cosh(np.pi * dt * FW) - np.cos((fv - f0) * 2 * np.pi * dt)
    gate = (np.abs(fv - f0) <= 1 / (2 * dt)).astype(float)
    return (1.0 / Nt) * (num / den) * np.abs(As) ** 2 * np.abs(Ac) ** 2 * gate


def boxcar_valid(y, N):
    if N <= 1: return y
    return np.convolve(y, np.ones(N) / N, mode='valid')


# ---------------------------------------------------------------------------
#  Main simulation
# ---------------------------------------------------------------------------

def run_swept_lo(
    seed          = 0,
    N_sw          = 100,
    NITER         = 10,
    SAVE_PATH     = None,
    SMOOTH_N      = 200,
    show_per_tooth= True,
):
    """
    Run the swept stepped-LO simulation.

    Assumptions
    -----------
    A1. Discrete time.  dt = 1/(2B).  N_tot = N_sw * Nt_sw.
    A2. LO: pure tone E0*exp(i2pi f_comb[p] tau_j) during step p.
        |E_LO| = E0 > 0 everywhere.
    A3. H(omega) = 1 (ideal detector).
    A4. Implicit DFT periodicity over T = N_tot * dt.
    A5. Non-overlapping regime: signal BW per tooth < fr1/2.
    A6. gcd(N_st, N_sw) = 1 required for unambiguous CRT phase mapping.
    A7. Numerical sanity: f_comb[p] * T_sw is an integer for all p.
        (Verified below so that exp(i2pi f_p r T_sw) = 1 and the
         inter-sweep phase is purely from E_s.)
    """

    rng = np.random.default_rng(seed)

    # ----------------------------------------------------------------
    # 1. LO grid — matches fc1 in testin3.py
    # ----------------------------------------------------------------
    fr1         = 10e9
    f_comb_full = np.arange(3.9e12, 5.1e12 + fr1 / 2, fr1)
    active_mask = (f_comb_full >= 4e12) & (f_comb_full <= 5e12)
    f_comb      = f_comb_full[active_mask]
    N_st        = len(f_comb)

    LO_PWR = 1e-3
    E0     = np.sqrt(LO_PWR)

    # ----------------------------------------------------------------
    # 2. Signal — identical to testin3.py
    # ----------------------------------------------------------------
    fs    = np.array([4.211, 4.558, 4.783, 4.803]) * 1e12
    As    = np.array([1.0,   1.5,   2.0,   2.5  ]) * 1e-5 * np.sqrt(LO_PWR)
    FWHMs = np.array([10e6,  10e6,  10e6,  10e6 ]) * 10 * 100   # 1 GHz each

    # ----------------------------------------------------------------
    # 3. Timing
    # ----------------------------------------------------------------
    B           = 10e9 * 5          # 50 GHz one-sided BW
    RBW         = 1e6  * 10         # 10 MHz RBW per tooth
    dt          = 1.0 / (2 * B)    # 10 ps
    Nt_per_step = int(np.round(1.0 / (RBW * dt)))   # 10 000 samples
    T_step      = Nt_per_step * dt
    Nt_sw       = N_st * Nt_per_step
    T_sw        = Nt_sw * dt
    N_tot       = N_sw * Nt_sw

    assert abs(RBW * T_step - 1.0) < 1e-9, "RBW*T_step != 1"

    # Assumption A7 check: f_comb[p] * T_sw must be an integer for all p.
    # This ensures exp(i2pi f_p r T_sw) = 1, so the inter-sweep phase
    # of S_tilde_r comes entirely from E_s (giving exp(+i2pi s_n r/N_sw)).
    for p, fp in enumerate(f_comb):
        val = fp * T_sw
        assert abs(val - round(val)) < 1e-3, \
            f"f_comb[{p}] * T_sw = {val} is not an integer. " \
            "Phase interleaving derivation breaks down."

    PSD_UNITS = Nt_per_step * dt * 1e18 * 1e6    # -> nW^2/MHz

    print(f"[setup] N_st={N_st}, Nt_per_step={Nt_per_step}, dt={dt*1e12:.1f} ps")
    print(f"        T_step={T_step*1e9:.1f} ns, T_sw={T_sw*1e6:.3f} µs, N_sw={N_sw}")
    print(f"        N_tot={N_tot:,}, fine resolution={1/(N_sw*T_sw)/1e3:.1f} kHz")

    # Coprimality check (Assumption A6)
    if gcd(N_st, N_sw) != 1:
        print(f"[WARNING] gcd(N_st={N_st}, N_sw={N_sw}) = {gcd(N_st,N_sw)} != 1. "
              "Some fine-grid bins will be degenerate.")

    lo = SweptSteppedLO(f_comb, E0, N_sw)

    # ----------------------------------------------------------------
    # 4. Simulation loop
    # ----------------------------------------------------------------
    Ps_recon_acc  = 0.0
    Ps_theory_acc = 0.0
    IF_psd_acc    = 0.0

    for it in tqdm(range(NITER), desc='Averaging'):

        # 4a. Signal envelopes: (N_tot, n_lines)
        sig = Signal(fs, As, FWHMs)
        sig.generate_envelopes(N_tot, dt, rng)

        # 4b. Interferogram: shape (Nt_sw, N_sw), column r = sweep r
        I = build_interferogram_single_lo(
            signal=sig, f_comb=f_comb, E0=E0,
            Nt_sw=Nt_sw, dt=dt, B=B, N_sw=N_sw,
        )

        # 4c. Spectral extraction
        # ----------------------------------------------------------------
        # For N_sw = 1: per-tooth FFT, no phase interleaving needed.
        # For N_sw > 1: sweep DFT with twiddle exp(-2pi i s_n r / N_sw).
        #
        # Both return:
        #   f_optical   — stitched optical freq axis (Hz), ~THz range
        #   Ps_stitched — |S_fine|^2 / |E0|^2, on f_optical axis
        # ----------------------------------------------------------------
        if N_sw == 1:
            result = extract_single_sweep(
                interferogram=I[:, 0], f_comb=f_comb, E0=E0,
                Nt_sw=Nt_sw, dt=dt, f_rep=fr1,
            )
        else:
            result = extract_multi_sweep(
                interferogram=I, f_comb=f_comb, E0=E0,
                Nt_sw=Nt_sw, dt=dt, f_rep=fr1, N_sw=N_sw,
            )

        f_opt    = result['f_optical']
        Ps_recon = result['Ps_stitched']

        # 4d. Analytic ground truth on the same frequency axis.
        #
        # Nt = Nt_per_step because:
        #   - Each tooth's spectrum comes from a per-step FFT of Nps=Nt_per_step samples.
        #   - After the sweep DFT, |S_fine| = |F_pr|  (coherent sum: N_sw * 1/N_sw = 1),
        #     so the amplitude equals one sweep's per-step FFT value.
        #   - Dividing by |E0|^2 removes the LO power.
        # Ac = 1.0 because |E0|^2 is already divided out.
        Ps_theory = np.zeros_like(f_opt)
        for k in range(len(fs)):
            Ps_theory += Lfcn2(
                f_opt, fs[k], As[k], 1.0, FWHMs[k], Nt_per_step, dt
            )

        # 4e. IF diagnostic (tooth nearest signal line 0, sweep 0)
        p_diag  = int(np.argmin(np.abs(fs[0] - f_comb)))
        I_diag  = I[p_diag * Nt_per_step : (p_diag + 1) * Nt_per_step, 0]
        F_if    = np.fft.fftshift(np.fft.fft(I_diag)) / Nt_per_step
        IF_psd_acc += np.abs(F_if) ** 2 / E0 ** 2

        # Accumulate — ONCE per iteration
        Ps_recon_acc  += Ps_recon
        Ps_theory_acc += Ps_theory

    # Average
    Ps_recon_mean  = Ps_recon_acc  / NITER
    Ps_theory_mean = Ps_theory_acc / NITER
    IF_psd_mean    = IF_psd_acc    / NITER

    # ----------------------------------------------------------------
    # 5. Consistency check
    # ----------------------------------------------------------------
    # At the analytic peak, recon should match theory within noise.
    peak_idx    = np.argmax(Ps_theory_mean)
    peak_theory = Ps_theory_mean[peak_idx] * PSD_UNITS * LO_PWR
    peak_recon  = Ps_recon_mean[peak_idx]  * PSD_UNITS * LO_PWR
    rel_err     = abs(peak_recon - peak_theory) / (peak_theory + 1e-300)
    print(f"[check] Peak theory = {peak_theory:.3e} nW²/MHz, "
          f"recon = {peak_recon:.3e} nW²/MHz, "
          f"relative error = {rel_err*100:.1f}%")

    # ----------------------------------------------------------------
    # 6. Plots
    # ----------------------------------------------------------------
    f_IF_diag = np.fft.fftshift(np.fft.fftfreq(Nt_per_step, d=dt))
    SMOOTH = int(SMOOTH_N)
    x  = boxcar_valid(f_opt,            SMOOTH)
    yr = boxcar_valid(Ps_recon_mean,  SMOOTH)
    ya = boxcar_valid(Ps_theory_mean, SMOOTH)

    fig = plt.figure(figsize=(6.5, 7.5))

    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(f_IF_diag / 1e9, IF_psd_mean * PSD_UNITS * LO_PWR,
             lw=0.8, label=r'$|\tilde{S}_r(\omega_k)|^2$ tooth 0')
    ax1.set_xlim([-5, 5])
    ax1.set_xlabel('Intermediate frequency (GHz)')
    ax1.set_ylabel('PSD (nW$^2$/MHz)')
    ax1.set_title('Raw IF PSD')
    ax1.legend(loc='upper right', fontsize=8)

    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(x / 1e12, (yr - ya) * PSD_UNITS * LO_PWR, lw=0.8)
    ax2.set_ylabel('Residual (nW$^2$/MHz)')
    ax2.set_xticklabels([])
    ax2.set_title('Reconstruction residual')

    ax3 = plt.subplot(3, 1, 3, sharex=ax2)
    ax3.plot(x / 1e12, yr * PSD_UNITS * LO_PWR, label='reconstructed')
    ax3.plot(x / 1e12, ya * PSD_UNITS * LO_PWR, label='analytic')
    ax3.set_xlabel('Optical frequency (THz)')
    ax3.set_ylabel('Optical PSD (nW$^2$/MHz)')
    ax3.set_xlim([4.0, 5.0])
    ax3.legend(loc='upper left', fontsize=8)
    ax3.set_title(f'Reconstructed broadband spectrum (stepped LO, N_sw={N_sw})')

    plt.tight_layout()
    if SAVE_PATH:
        plt.savefig(SAVE_PATH, dpi=180, bbox_inches='tight')
    plt.show()

    # ----------------------------------------------------------------
    # 7. Per-tooth mini-spectra
    # ----------------------------------------------------------------
    if show_per_tooth:
        mini_sp  = result['mini_spectra']
        f_IF_ax  = result['f_IF']
        guard    = np.abs(f_IF_ax) < fr1 / 2.0

        fig2, axs = plt.subplots(1, len(fs), figsize=(3.2 * len(fs), 2.8), sharey=True)
        if len(fs) == 1: axs = [axs]

        for k, fk in enumerate(fs):
            p_near = int(np.argmin(np.abs(fk - f_comb)))
            recon_mini  = mini_sp[p_near]
            theory_mini = Lfcn(
                f_IF_ax, fk - f_comb[p_near], As[k], 1.0,
                FWHMs[k], Nt_per_step, dt,
            )
            IF_offset = (fk - f_comb[p_near]) / 1e9
            axs[k].plot(f_IF_ax[guard] / 1e9, recon_mini[guard], lw=1, label='recon')
            axs[k].plot(f_IF_ax[guard] / 1e9, theory_mini[guard], lw=1, label='analytic')
            axs[k].set_title(f'tooth near {fk/1e12:.3f} THz\nIF = {IF_offset:.2f} GHz')
            axs[k].set_xlabel('IF (GHz)')
        axs[0].set_ylabel('mini-spectrum (a.u.)')
        axs[0].legend(fontsize=7)
        plt.suptitle('Per-tooth mini-spectra: recon vs analytic', y=1.02)
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    # Level 1: single sweep, H=1
    # Expected: four Lorentzian peaks on top of analytic, flat residual.
    run_swept_lo(seed=0, N_sw=1, NITER=1, SMOOTH_N=200, show_per_tooth=True)

    # Level 2: multi-sweep phase interleaving (N_sw=20)
    # Expected: same peaks, N_sw times denser frequency grid.
    # run_swept_lo(seed=0, N_sw=20, NITER=1, SMOOTH_N=200, show_per_tooth=True)