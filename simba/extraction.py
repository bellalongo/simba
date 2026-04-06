# extraction.py  — corrected
"""
Spectral extraction for the swept-LO simulation.

Two approaches:

(A) extract_single_sweep  — per-tooth FFT, single sweep, H=1.
    f_optical = f_comb[p] + f_IF  (stitching).
    Analytic: Lfcn2(f_opt, fs[k], As[k], 1.0, FWHM, Nt_per_step, dt)

(B) extract_multi_sweep  — sweep DFT with phase interleaving.
    Same stitching; each coarse bin is split into N_sw fine-grid offsets.

    SIGN CONVENTION (derivation summary, discrete identities used):
    ---------------------------------------------------------------
    The interferogram during step p of sweep r carries the factor

        exp(+i 2pi s_n r / N_sw)          ... (inter-sweep phase from E_s)

    from the signal carrier. To coherently combine N_sw sweeps and
    isolate offset s_n, the sweep DFT multiplies by the conjugate:

        (1/N_sw) sum_r exp(-i 2pi s_n r / N_sw) * exp(+i 2pi s_n r / N_sw)
            = (1/N_sw) sum_r 1  =  1

    DFT orthogonality (discrete, exact):
        (1/N_sw) sum_{r=0}^{N_sw-1} exp(i 2pi (s - s') r / N_sw)
            = delta_{s, s'}    for 0 <= s, s' < N_sw

    The sign in the twiddle is NEGATIVE:  exp(-2pi i s_n r / N_sw).

    Analytic: Lfcn2 with Nt = Nt_per_step (= Nps = samples per step).
    The amplitude after the sweep DFT equals a SINGLE-SWEEP FFT value
    (coherent sum: N_sw terms each of value 1/N_sw, product = 1),
    so the amplitude calibration is identical to the single-sweep case.
"""
import numpy as np


# ---------------------------------------------------------------------------
# (A) Single-sweep extraction
# ---------------------------------------------------------------------------

def extract_single_sweep(
    interferogram: np.ndarray,
    f_comb: np.ndarray,
    E0: float,
    Nt_sw: int,
    dt: float,
    f_rep: float,
    detector=None,
):
    """
    Per-tooth FFT of a single-sweep stepped-LO interferogram.

    Assumptions
    -----------
    A1. H(omega) = 1 (no detector correction).
    A2. Non-overlapping: signal bandwidth per tooth < f_rep/2.
    A3. interferogram[j] = E0* * env_m * exp(i 2pi df t_abs[j])
        stored at absolute times t_abs = j * dt.
    A4. Implicit periodicity: DFT treats each dwell window as periodic.

    Returns
    -------
    dict with keys:
        'f_optical'  : stitched optical frequency axis (Hz)
        'Ps_stitched': stitched power spectrum |Ê_s|² / |E0|²
        'f_IF'       : per-tooth IF axis (Hz)
        'mini_spectra': dict {tooth_index -> mini-spectrum array}
    """
    I = interferogram.ravel()
    N_st = len(f_comb)
    Nps  = Nt_sw // N_st          # samples per step (= Nt_per_step)

    f_IF = np.fft.fftshift(np.fft.fftfreq(Nps, d=dt))
    guard = np.abs(f_IF) < f_rep / 2.0

    f_opt_list, Ps_list, mini_spectra = [], [], {}

    for p in range(N_st):
        j0, j1 = p * Nps, (p + 1) * Nps
        I_p = I[j0:j1]
        if detector is not None:
            I_p = detector.deconvolve(I_p, dt)

        # Per-step FFT  (1/Nps normalisation)
        F_p  = np.fft.fftshift(np.fft.fft(I_p)) / Nps
        P_p  = np.abs(F_p) ** 2 / np.abs(E0) ** 2     # divide by |E0|^2

        mini_spectra[p] = P_p
        f_opt_list.append(f_comb[p] + f_IF[guard])
        Ps_list.append(P_p[guard])

    f_optical  = np.concatenate(f_opt_list)
    Ps_stitched = np.concatenate(Ps_list)
    idx = np.argsort(f_optical)
    return {
        'f_optical':   f_optical[idx],
        'Ps_stitched': Ps_stitched[idx],
        'f_IF':        f_IF,
        'mini_spectra': mini_spectra,
    }


# ---------------------------------------------------------------------------
# (B) Multi-sweep extraction with corrected phase-interleaving sign
# ---------------------------------------------------------------------------

def extract_multi_sweep(
    interferogram: np.ndarray,
    f_comb: np.ndarray,
    E0: float,
    Nt_sw: int,
    dt: float,
    f_rep: float,
    N_sw: int,
    detector=None,
):
    """
    Sweep-DFT extraction using N_sw sweeps (phase interleaving).

    Twiddle sign derivation
    -----------------------
    The per-step FFT of sweep r carries the r-dependent phase

        phi_r = 2pi * s_n * r / N_sw   (positive, from E_s carrier)

    [Flag: this relies on f_comb[p] * T_sw being an integer, which holds
     when fr1 * T_sw and f_start * T_sw are both integers. Check before use.]

    To select offset s_n by DFT orthogonality:
        (1/N_sw) sum_r exp(-i phi_r) * exp(+i phi_r) = 1   (s matches)
        (1/N_sw) sum_r exp(-i phi_r) * exp(+i phi_{r,wrong}) = 0  (mismatch)

    Twiddle:  exp(-2 pi i s_n r / N_sw)  — NEGATIVE sign.

    Parameters
    ----------
    interferogram : (Nt_sw, N_sw) array
        Heterodyne signal.  Column r = sweep r.
    f_comb : (N_st,) array
    E0     : LO field amplitude
    Nt_sw  : samples per sweep
    dt     : sampling interval
    f_rep  : coarse LO step spacing (Hz)
    N_sw   : number of sweeps

    Returns
    -------
    dict with same keys as extract_single_sweep.
    """
    N_st = len(f_comb)
    Nps  = Nt_sw // N_st          # samples per step
    T_sw = Nt_sw * dt

    f_IF  = np.fft.fftshift(np.fft.fftfreq(Nps, d=dt))
    guard = np.abs(f_IF) < f_rep / 2.0

    f_rep_fine = 1.0 / (N_sw * T_sw)   # fine-grid spacing = 1 / T_total
    r_arr      = np.arange(N_sw)

    f_opt_list, Ps_list = [], []
    mini_spectra = {}

    for p in range(N_st):
        j0, j1 = p * Nps, (p + 1) * Nps

        # Per-step FFT for each sweep r
        F_sweeps = np.zeros((N_sw, Nps), dtype=np.complex128)
        for r in range(N_sw):
            I_pr = interferogram[j0:j1, r]
            if detector is not None:
                I_pr = detector.deconvolve(I_pr, dt)
            F_sweeps[r] = np.fft.fftshift(np.fft.fft(I_pr)) / Nps

        # Sweep DFT for each fine-grid offset s_n = 0,...,N_sw-1
        for s_n in range(N_sw):
            # NEGATIVE sign: cancels the +s_n phase carried by F_sweeps
            # DFT orthogonality:
            #   (1/N_sw) sum_r exp(-i2pi s_n r/N_sw) exp(+i2pi s_n r/N_sw) = 1
            #   (1/N_sw) sum_r exp(-i2pi s_n r/N_sw) exp(+i2pi s' r/N_sw) = 0  (s'!=s_n)
            twiddle = np.exp(-2j * np.pi * s_n * r_arr / N_sw)   # ← CORRECTED sign
            S_fine  = (1.0 / N_sw) * (twiddle @ F_sweeps)        # shape (Nps,)
            P_fine  = np.abs(S_fine) ** 2 / np.abs(E0) ** 2

            # Optical frequency = comb tooth + IF + fine-grid offset
            f_opt = f_comb[p] + f_IF[guard] + s_n * f_rep_fine
            f_opt_list.append(f_opt)
            Ps_list.append(P_fine[guard])

        # Coarse mini-spectrum for this tooth (s_n=0 only, for display)
        twiddle0       = np.ones(N_sw)
        S_coarse       = (1.0 / N_sw) * (twiddle0 @ F_sweeps)
        mini_spectra[p] = np.abs(S_coarse) ** 2 / np.abs(E0) ** 2

    f_optical  = np.concatenate(f_opt_list)
    Ps_stitched = np.concatenate(Ps_list)
    idx = np.argsort(f_optical)
    return {
        'f_optical':   f_optical[idx],
        'Ps_stitched': Ps_stitched[idx],
        'f_IF':        f_IF,
        'mini_spectra': mini_spectra,
    }