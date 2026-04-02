#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# main.py
"""
Swept stepped-LO spectral reconstruction simulation (simba).

Mirrors the structure of testin3.py (dual-comb ptychoscopy) but replaces
the two-comb + ptychoscopic-correlation pipeline with a SINGLE stepped LO
and the corrected V4.0 master-equation extraction:

    P_s(omega_n^(c)) = |(1/N_sw) sum_r  e^{-2*pi*i*s_n*r/N_sw}
                         (1/N_t^(sw)) sum_j  [S_r(tau_j) / E*_LO(t_{r,j})]
                         * e^{-i*omega_n^(c)*tau_j}|^2          ... (Eq. 6 / Eq. 7)

For N_sw = 1 this collapses to a direct per-tooth FFT (Case 1, Eq. 8).
For N_sw > 1 the outer sum is the sweep DFT that provides fine-frequency
resolution via phase interleaving (gcd(N_st, N_sw) = 1 required).

Parameters are chosen to MATCH testin3.py for side-by-side comparison:
    same signal lines  : 4.211 / 4.558 / 4.783 / 4.803 THz
    same amplitudes    : [1.0, 1.5, 2.0, 2.5] * 1e-5 * sqrt(1 mW)
    same linewidths    : 1 GHz FWHM each
    same LO spacing    : fr1 = 10 GHz
    same bandwidth     : B = 50 GHz  =>  dt = 10 ps
    same RBW per tooth : 10 MHz  =>  Nt_per_step = 10 000

Key differences from testin3.py
---------------------------------
  testin3      : dual-comb (fc1, fc2), ptychoscopic correlation C_n(omega)
  this file    : single stepped LO, time-domain division S / E*_LO -> DFT
  no sinc      : LO divides out exactly at every sample (see Corrected_Master_Equation)
  no overlap   : no multi-tooth interference in any IF bin (sequential LO)
  no |E0|^2 in result : amplitude cancels in noiseless case

Notation (Corrected_Master_Equation_Findings.pdf)
--------------------------------------------------
  r          : sweep index, r = 0,...,N_sw-1
  j          : fast-time index within sweep, j = 0,...,N_t^(sw)-1
  p          : LO step index within sweep, p = 0,...,N_st-1
  t_{r,j}   : global sample time = r*T_sw + tau_j
  tau_j      : local sweep time  = j * dt
  omega_n^(c): fine optical frequency = n * omega_rep, n = 0,...,N_tot-1
  q_n, s_n   : coarse/fine index decomp  n = q_n * N_sw + s_n
"""

import numpy as np
import matplotlib.pyplot as plt
from math import gcd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

# ------------------------------------------------------------------
# simba imports  (modules not re-exported via __init__, import directly)
# ------------------------------------------------------------------
from simba import Signal                                  # gen_signal
from simba.local_oscillator import SteppedLO, SweptSteppedLO
from simba.mixing import build_interferogram_single_lo
from simba.extraction import extract_single_sweep, extract_multi_sweep


# ==================================================================
#  Analytic lineshapes — identical to testin3.py for direct comparison
# ==================================================================

def Lfcn(fv, f0, As, Ac, FW, Nt, dt):
    """
    Lorentzian PSD in the IF domain (Cauchy form).

    Parameters
    ----------
    fv  : frequency axis (Hz)
    f0  : line centre in IF (Hz)
    As  : signal field amplitude (V/m)
    Ac  : LO field amplitude (V/m); pass 1.0 when extraction already
          normalises by |E0|^2
    FW  : FWHM linewidth (Hz)
    Nt  : dwell samples (= Nt_per_step for stepped LO)
    dt  : sampling interval (s)
    """
    num = 2 * np.pi * dt * FW
    den = (np.pi * dt * FW) ** 2 + ((fv - f0) * (2 * np.pi * dt)) ** 2
    return (1.0 / Nt) * (num / den) * (np.abs(As) ** 2) * (np.abs(Ac) ** 2)


def Lfcn2(fv, f0, As, Ac, FW, Nt, dt):
    """
    Exact discrete Lorentzian (accounts for DFT periodicity / aliasing).
    Use this for the full stitched-spectrum analytic ground truth.
    Gate: only contributions within the Nyquist band |f - f0| <= 1/(2*dt).
    """
    num  = np.sinh(np.pi * dt * FW)
    den  = np.cosh(np.pi * dt * FW) - np.cos((fv - f0) * (2 * np.pi * dt))
    gate = (np.abs(fv - f0) <= 1 / (2 * dt)).astype(float)
    return (1.0 / Nt) * (num / den) * (np.abs(As) ** 2) * (np.abs(Ac) ** 2) * gate


def boxcar_valid(y, N):
    """Uniform boxcar smoothing, 'valid' mode (trims N-1 edges)."""
    if N <= 1:
        return y
    return np.convolve(y, np.ones(N) / N, mode='valid')


# ==================================================================
#  Main simulation function
# ==================================================================

def run_swept_lo(
    seed         = 0,
    N_sw         = 1,        # number of sweeps (1 = "short" / Level-1 test)
    NITER        = 1,        # ensemble averages over phase realisations
    SAVE_PATH    = None,
    SMOOTH_N     = 200,
    show_per_tooth = True,
):
    """
    Run the swept stepped-LO simulation.

    Assumptions (stated explicitly per the PhD-student derivation rules)
    -------------------------------------------------------------------
    A1  Discrete time.  Sample index m = 0,...,N_tot-1, spacing dt = 1/(2B).
    A2  LO is sequential: during step p of sweep r the LO is a pure tone
            E_LO(t_{r,j}) = E0 * exp(i * 2*pi * f_comb[p] * tau_j)
        so |E_LO| = E0 > 0 everywhere -- no Wiener regularisation needed.
    A3  H(omega) = 1 (infinite bandwidth).  No deconvolution applied.
        Detector bandwidth is enforced only by discarding IF beats with
        |f_beat| > B during interferogram construction.
    A4  Implicit periodicity: the DFT treats the N_tot-sample record as
        periodic.  Fine resolution = 1/T = 1/(N_sw * T_sw).
    A5  Signal amplitudes already include sqrt(LO_PWR) power scaling
        (same convention as testin3.py).
    A6  gcd(N_st, N_sw) = 1 required for unambiguous CRT phase mapping.
    """

    rng = np.random.default_rng(seed)

    # ----------------------------------------------------------------
    # 1.  LO GRID   (matches fc1 in testin3.py)
    # ----------------------------------------------------------------
    # Assumption A2: LO steps through f_comb[p] = f_start + p * fr1.
    # Active teeth: 4 THz <= f_comb[p] <= 5 THz  (same as active_mask in testin3).
    # Index range: p = 0,...,N_st-1.

    fr1         = 10e9                                        # step spacing (Hz)
    f_comb_full = np.arange(3.9e12, 5.1e12 + fr1 / 2, fr1)   # full grid
    active_mask = (f_comb_full >= 4e12) & (f_comb_full <= 5e12)
    f_comb      = f_comb_full[active_mask]                    # shape (N_st,)
    N_st        = len(f_comb)

    LO_PWR = 1e-3                  # 1 mW per tooth
    E0     = np.sqrt(LO_PWR)       # LO field amplitude s.t. |E0|^2 = LO_PWR

    # ----------------------------------------------------------------
    # 2.  SIGNAL   (identical to testin3.py)
    # ----------------------------------------------------------------
    # Four Lorentzian lines, 1 GHz FWHM each, power-scaled by sqrt(LO_PWR).
    # Index k = 0,...,3.

    fs    = np.array([4.211, 4.558, 4.783, 4.803]) * 1e12    # carrier freq (Hz)
    As    = np.array([1.0,   1.5,   2.0,   2.5  ]) * 1e-5    # amplitudes (V/m)
    FWHMs = np.array([10e6,  10e6,  10e6,  10e6 ]) * 10 * 100 # 1 GHz each
    As    = As * np.sqrt(LO_PWR)  # power scaling (matches testin3 convention)

    # ----------------------------------------------------------------
    # 3.  TIMING   (mirrors testin3.py derivation)
    # ----------------------------------------------------------------
    # Assumption A1.
    # dt     = 1/(2B)         -- Nyquist interval
    # Nt_per_step = 1/(RBW*dt) -- samples for 10 MHz RBW per tooth
    # T_step = Nt_per_step * dt = 1/RBW = 100 ns per tooth
    # Nt_sw  = N_st * Nt_per_step -- samples per sweep
    # T_sw   = Nt_sw * dt         -- sweep duration
    # N_tot  = N_sw * Nt_sw        -- total samples

    B           = 10e9 * 5    # one-sided detector bandwidth = 50 GHz (Hz)
    RBW         = 1e6  * 10   # resolution bandwidth per tooth = 10 MHz (Hz)
    dt          = 1.0 / (2 * B)                    # 10 ps
    Nt_per_step = int(np.round(1.0 / (RBW * dt)))  # 10 000 samples/step
    T_step      = Nt_per_step * dt                 # 100 ns
    Nt_sw       = N_st * Nt_per_step               # samples per sweep
    T_sw        = Nt_sw * dt                       # sweep duration
    N_tot       = N_sw * Nt_sw                     # total samples

    # Consistency check (dimensional analysis):  RBW * T_step == 1 (matched filter)
    assert abs(RBW * T_step - 1.0) < 1e-9, \
        f"Consistency failure: RBW*T_step = {RBW*T_step}, expected 1."

    PSD_UNITS = Nt_per_step * dt * 1e18 * 1e6   # -> nW^2/MHz  (same as testin3)

    print(f"[setup] N_st={N_st}, Nt_per_step={Nt_per_step}, dt={dt*1e12:.1f} ps")
    print(f"        T_step={T_step*1e9:.1f} ns, T_sw={T_sw*1e6:.3f} µs, N_sw={N_sw}")
    print(f"        N_tot={N_tot:,}, RBW/tooth={RBW/1e6:.0f} MHz")
    print(f"        Fine resolution = 1/T = {1/(N_sw*T_sw)/1e3:.1f} kHz")

    # ----------------------------------------------------------------
    # 4.  LO OBJECT
    # ----------------------------------------------------------------
    # CRT coprimality check (Assumption A6):
    # gcd(N_st, N_sw) = 1 ensures the phase mapping
    #   n = q_n * N_sw + s_n  is a bijection onto {0,...,N_st*N_sw - 1}.
    # [Flag: modular arithmetic / periodicity]
    if gcd(N_st, N_sw) != 1:
        print(f"[WARNING] gcd(N_st={N_st}, N_sw={N_sw}) = {gcd(N_st,N_sw)} != 1.  "
              "Phase interleaving is degenerate -- some fine-grid bins will be missed.")

    lo = SweptSteppedLO(f_comb, E0, N_sw)

    # ----------------------------------------------------------------
    # 5.  SIMULATION LOOP
    # ----------------------------------------------------------------
    Ps_recon_acc  = 0.0
    Ps_theory_acc = 0.0
    IF_psd_acc    = 0.0

    for it in tqdm(range(NITER), desc='Averaging iterations'):

        # 5a.  Signal envelopes
        # env_k[m] = As[k] * exp(i * phi_k(m))
        # phi_k: Wiener process, Var(d phi_k) = 2*pi*FWHM_k*dt per step.
        # Shape: (N_tot, n_lines)
        sig = Signal(fs, As, FWHMs)
        sig.generate_envelopes(N_tot, dt, rng)

        # 5b.  Heterodyne interferogram
        # S_r(tau_j) = E0* * eps_r*(tau_j) * E_s(t_{r,j})
        # where eps_r(tau_j) = E0 * exp(i*2*pi*f_comb[p]*tau_j)  during step p.
        # Output shape: (Nt_sw, N_sw),  column r = sweep r.
        # [Flag: finite observation time T = N_sw * T_sw]
        I = build_interferogram_single_lo(
            signal = sig,
            f_comb = f_comb,
            E0     = E0,
            Nt_sw  = Nt_sw,
            dt     = dt,
            B      = B,
            N_sw   = N_sw,
        )

        # 5c.  Spectral extraction -- V4.0 master equation
        # -------------------------------------------------
        # Noiseless pipeline (Assumption A3):
        #   E_s(t_{r,j}) = S_r(tau_j) / E*_LO(t_{r,j})   [time-domain division]
        #   a_n = (1/N_tot) sum_{r,j} E_s(t_{r,j}) * exp(-i*omega_n^(c)*t_{r,j})
        #       = E_hat_s(omega_n^(c))                      [global DFT of E_s]
        #   P_s(omega_n^(c)) = |a_n|^2
        #
        # Cooley-Tukey factored form (Eq. 6):
        #   inner (j): per-step FFT evaluated at fine-grid frequency omega_n^(c)
        #              kernel e^{-i*omega_n^(c)*tau_j}  [includes twiddle factor]
        #   outer (r): sweep DFT  e^{-2*pi*i*s_n*r/N_sw}
        #
        # For N_sw=1: outer sum has one term, reduces to direct per-tooth FFT.
        # [Flag: implicit periodicity in DFT, finite window of length T_step]

        if N_sw == 1:
            result = extract_single_sweep(
                interferogram = I[:, 0],   # shape (Nt_sw,) for single sweep
                f_comb        = f_comb,
                E0            = E0,
                Nt_sw         = Nt_sw,
                dt            = dt,
                f_rep         = fr1,
                detector      = None,      # H=1 (Assumption A3)
            )
        else:
            result = extract_multi_sweep(
                interferogram = I,         # shape (Nt_sw, N_sw)
                f_comb        = f_comb,
                E0            = E0,
                Nt_sw         = Nt_sw,
                dt            = dt,
                f_rep         = fr1,
                N_sw          = N_sw,
                detector      = None,
            )

        f_opt    = result['f_optical']    # stitched optical freq axis (Hz)
        Ps_recon = result['Ps_stitched']  # |a_n|^2, already divided by |E0|^2

        # 5d.  Analytic ground truth
        # Lfcn2 with Ac=1.0 because extraction normalises by |E0|^2.
        # Nt = Nt_per_step (dwell window sets resolution).
        # [Consistency check: at resonance f_opt = fs[k],
        #   Lfcn2 peak = |As[k]|^2 / (pi*dt*FWHM*Nt_per_step) ~ signal PSD]
        Ps_theory = np.zeros_like(f_opt)
        for k in range(len(fs)):
            Ps_theory += Lfcn2(f_opt, fs[k], As[k], 1.0, FWHMs[k], Nt_per_step, dt)

        # Diagnostic: IF PSD of first tooth in first sweep (should show peaks)
        I_tooth0 = I[:Nt_per_step, 0]
        F_if     = np.fft.fftshift(np.fft.fft(I_tooth0)) / Nt_per_step
        IF_psd_acc   += np.abs(F_if) ** 2

        Ps_recon_acc  += Ps_recon
        Ps_theory_acc += Ps_theory

    # Average over realisations
    Ps_recon_mean  = Ps_recon_acc  / NITER
    Ps_theory_mean = Ps_theory_acc / NITER
    IF_psd_mean    = IF_psd_acc    / NITER

    f_IF_diag = np.fft.fftshift(np.fft.fftfreq(Nt_per_step, d=dt))

    # ----------------------------------------------------------------
    # 6.  SANITY CHECK  (mirrors testin3.py assertion)
    # ----------------------------------------------------------------
    # Find the tooth nearest signal line 0 and compare recon vs analytic.
    p_check = int(np.argmin(np.abs(fs[0] - f_comb)))
    mini_sp = result['mini_spectra'][p_check]
    f_if_ax = result['f_IF']
    theory_check = Lfcn(f_if_ax, fs[0] - f_comb[p_check], As[0], 1.0,
                        FWHMs[0], Nt_per_step, dt)

    if np.all(theory_check == 0) or np.all(mini_sp == 0):
        raise RuntimeError("Sanity check failed: zero vector in mini-spectrum.")
    assert np.dot(mini_sp, theory_check) > 0, \
        "Mini-spectrum not aligned with analytic theory (wrong sign / reshape?)."
    print("[sanity] Mini-spectrum dot-product check passed.")

    # ----------------------------------------------------------------
    # 7.  PLOTS   (3-panel, mirrors testin3.py figure layout)
    # ----------------------------------------------------------------
    SMOOTH = int(SMOOTH_N)
    x  = boxcar_valid(f_opt,          SMOOTH)
    yr = boxcar_valid(Ps_recon_mean,  SMOOTH)
    ya = boxcar_valid(Ps_theory_mean, SMOOTH)

    fig = plt.figure(figsize=(6.5, 7.5))

    # Panel 1: raw IF PSD (analog of "Raw power spectral density" in testin3)
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(f_IF_diag / 1e9, IF_psd_mean * PSD_UNITS * LO_PWR,
             lw=0.8, label=r'$|\tilde{S}_r(\omega_k)|^2$ tooth 0')
    ax1.set_xlim([-5, 5])
    ax1.set_xlabel('Intermediate frequency (GHz)')
    ax1.set_ylabel('PSD (nW$^2$/MHz)')
    ax1.set_title('Raw IF PSD')
    ax1.legend(loc='upper right', fontsize=8)

    # Panel 2: residual
    ax2 = plt.subplot(3, 1, 2)
    residual = (yr - ya) * PSD_UNITS * LO_PWR
    ax2.plot(x / 1e12, residual, lw=0.8)
    ax2.set_ylabel('Residual (nW$^2$/MHz)')
    ax2.set_xticklabels([])
    ax2.set_title('Reconstruction residual')

    # Panel 3: recon vs analytic (optical domain)
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
    # 8.  PER-TOOTH MINI-SPECTRA   (verification panel, mirrors testin3.py)
    # ----------------------------------------------------------------
    if show_per_tooth:
        mini_spectra = result['mini_spectra']
        f_IF_tooth   = result['f_IF']
        guard        = np.abs(f_IF_tooth) < fr1 / 2.0  # guard band ±fr1/2

        fig2, axs = plt.subplots(1, len(fs), figsize=(3.2 * len(fs), 2.8), sharey=True)
        if len(fs) == 1:
            axs = [axs]

        for k, fk in enumerate(fs):
            p_near = int(np.argmin(np.abs(fk - f_comb)))
            # extraction normalised by |E0|^2 already -> Ac=1
            recon_mini  = mini_spectra[p_near]
            theory_mini = Lfcn(
                f_IF_tooth,
                fk - f_comb[p_near],   # signal IF offset at this tooth
                As[k],
                1.0,                    # Ac=1: |E0|^2 already divided out
                FWHMs[k],
                Nt_per_step,
                dt,
            )
            axs[k].plot(f_IF_tooth[guard] / 1e9, recon_mini[guard],
                        lw=1, label='recon')
            axs[k].plot(f_IF_tooth[guard] / 1e9, theory_mini[guard],
                        lw=1, label='analytic')
            axs[k].set_title(f'tooth near {fk/1e12:.3f} THz\n'
                             f'IF = {(fk-f_comb[p_near])/1e9:.2f} GHz')
            axs[k].set_xlabel('IF (GHz)')

        axs[0].set_ylabel('mini-spectrum (a.u.)')
        axs[0].legend(fontsize=7)
        plt.suptitle('Per-tooth mini-spectra: recon vs analytic', y=1.02)
        plt.tight_layout()
        plt.show()


# ==================================================================
#  Entry point
# ==================================================================

if __name__ == '__main__':

    # ------------------------------------------------------------------
    # Level 1 sanity test  (single sweep, H=1, N_sw=1)
    # ------------------------------------------------------------------
    # What to expect:
    #   - Panel 1 (IF PSD): peaks at the beat frequencies of signal lines
    #     with tooth 0 (f_beat = fs - f_comb[0]).  Most lines will NOT
    #     appear here because their IF falls outside tooth 0's guard band.
    #   - Panel 3 (reconstructed): four Lorentzian peaks at 4.211, 4.558,
    #     4.783, 4.803 THz overlapping the analytic ground truth.
    #   - Panel 2 (residual): small and structureless.
    #   - Per-tooth plots: each signal line matches its Lfcn prediction.
    # ------------------------------------------------------------------
    run_swept_lo(
        seed          = 0,
        N_sw          = 1,
        NITER         = 1,
        SAVE_PATH     = None,
        SMOOTH_N      = 200,
        show_per_tooth= True,
    )

    # ------------------------------------------------------------------
    # Level 1 averaged  (uncomment after single-iteration looks good)
    # More averages -> smoother spectrum, residual -> noise floor
    # ------------------------------------------------------------------
    # run_swept_lo(seed=0, N_sw=1, NITER=10,
    #              SAVE_PATH='swept_lo_avg.png', SMOOTH_N=100,
    #              show_per_tooth=False)

    # ------------------------------------------------------------------
    # Level 2: multi-sweep phase interleaving
    # N_sw=3 is coprime with N_st~100, gives 3x finer resolution.
    # Expectation: same peaks, but the fine-frequency axis is denser
    # and narrow linewidths (1 GHz) should be resolved more cleanly.
    # ------------------------------------------------------------------
    # run_swept_lo(seed=0, N_sw=3, NITER=1,
    #              SAVE_PATH='swept_lo_3sweep.png', SMOOTH_N=200,
    #              show_per_tooth=True)