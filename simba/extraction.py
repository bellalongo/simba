# extraction.py
"""
Spectral extraction for the swept-LO simulation.

This module implements two extraction approaches:

(A) Direct mini-spectrum extraction (simple, per-tooth):
    For each comb tooth, take the FFT over its dwell time and
    map IF -> optical.  This gives coarse resolution (1/T_step)
    but is the most transparent check of the heterodyne physics.

(B) V3.0 / V4.0 master-equation extraction (sweep DFT):
    Per-sweep FFT of the heterodyne signal, followed by a DFT
    over sweep indices with phase-interleaving twiddle factors.
    Gives fine resolution 1/(N_sw * T_sw).

    The V3.0 formula (non-overlapping regime):
        P_s(omega_n^(c) + omega_k)
          = T_sw^2 |sum_r W^{s_n r} S_tilde_r(omega_{k,q_n})|^2
                 / |sum_r W^{s_n r} eps_tilde_r(omega_n^(c))|^2

    where W = exp(-2*pi*i / N_sw).

    The V4.0 formula (universal, time-domain division):
        P_s(omega_n^(c)) = |(1/N_sw) sum_r exp(-2*pi*i*s_n*r/N_sw)
                              E_hat_s^(r)(omega_{k,q_n})|^2

    Both are equivalent in the non-overlapping regime.

See V4.0 Eqs. (165)-(181) for the full pipeline.
"""
import numpy as np


# ---------------------------------------------------------------
#  (A) Direct per-tooth mini-spectrum extraction
# ---------------------------------------------------------------
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
    Extract the stitched optical power spectrum from a single-sweep
    stepped-LO interferogram.

    For each comb tooth p, the mini-spectrum is the FFT of the
    data recorded DURING that tooth's dwell time.  The mini-spectrum
    lives in the IF domain; stitching maps it to optical frequency:

        f_optical = f_comb[p] + f_IF

    Arguments
    ---------
    interferogram : np.ndarray, shape (Nt_sw,) or (Nt_sw, 1)
        Time-domain heterodyne signal for one sweep.
    f_comb : np.ndarray, shape (N_st,)
        Comb tooth frequencies (Hz).
    E0 : float
        LO field amplitude.
    Nt_sw : int
        Total samples in one sweep.
    dt : float
        Sampling interval (s).
    f_rep : float
        Comb spacing (Hz) = f_comb[1] - f_comb[0].
    detector : Detector or None
        If provided, deconvolves detector bandwidth from each
        mini-spectrum FFT before computing power.

    Returns
    -------
    result : dict
        'f_optical'     : stitched optical frequency axis (Hz)
        'Ps_stitched'   : stitched power spectrum (a.u.)
        'f_IF'          : per-tooth IF axis (Hz)
        'mini_spectra'  : dict mapping tooth index -> mini-spectrum array
    """
    I = interferogram.ravel()
    N_st = len(f_comb)
    Nps = Nt_sw // N_st  # samples per step

    # Guard band: only keep IF bins within f_rep / 2
    f_IF_step = np.fft.fftshift(np.fft.fftfreq(Nps, d=dt))
    guard_mask = np.abs(f_IF_step) < f_rep / 2.0

    f_optical_list = []
    Ps_list = []
    mini_spectra = {}

    for p in range(N_st):
        j0 = p * Nps
        j1 = (p + 1) * Nps

        # Data during this dwell
        I_p = I[j0:j1]

        # Apply detector deconvolution if requested
        if detector is not None:
            I_p = detector.deconvolve(I_p, dt)

        # Per-step FFT  (1/N normalised)
        F_p = np.fft.fftshift(np.fft.fft(I_p)) / Nps

        # Mini-spectrum: power normalised by |E0|^2
        P_p = np.abs(F_p) ** 2 / (np.abs(E0) ** 2)

        mini_spectra[p] = P_p

        # Stitch: keep only bins within the guard band
        f_opt_p = f_comb[p] + f_IF_step[guard_mask]
        Ps_p = P_p[guard_mask]

        f_optical_list.append(f_opt_p)
        Ps_list.append(Ps_p)

    f_optical = np.concatenate(f_optical_list)
    Ps_stitched = np.concatenate(Ps_list)

    # Sort by frequency
    sort_idx = np.argsort(f_optical)

    return {
        'f_optical': f_optical[sort_idx],
        'Ps_stitched': Ps_stitched[sort_idx],
        'f_IF': f_IF_step,
        'mini_spectra': mini_spectra,
    }


# ---------------------------------------------------------------
#  (B) Multi-sweep extraction with phase interleaving
# ---------------------------------------------------------------
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
    Extract the stitched optical power spectrum using N_sw sweeps
    with phase-interleaving (sweep DFT).

    Each sweep executes the same staircase LO pattern.  The signal's
    inter-sweep phase accumulation provides fine-grid discrimination:

        E_s(r*T_sw + tau) ~ sum_n a_n exp(i*omega_n^(c) * (r*T_sw + tau))

    The factor  exp(i * omega_n^(c) * r * T_sw) = exp(2*pi*i * s_n * r / N_sw)
    is extracted by a DFT over sweep index r  (Eq. 176 of V4.0).

    Pipeline (per comb tooth, per IF bin):
        1. Per-step FFT for each sweep r:  F_p^(r)(f_k)
        2. Sweep DFT with twiddle factor:
            S_n^(fine)(f_k) = (1/N_sw) sum_r exp(+2*pi*i*s_n*r/N_sw) F_p^(r)(f_k)
        3. Power:  P_s = |S_n^(fine)|^2 / |E_LO(omega_n^(c))|^2

    Note: the twiddle sign is +2*pi*i because we operate on the
    heterodyne signal S = eps* . E_s (the LO conjugation contributes
    a negative sign, compensated by the positive twiddle).
    In the V4.0 formula operating on E_s directly, the sign flips to
    -2*pi*i.  Both give the same power spectrum.  See V4.0 Table 1.

    Arguments
    ---------
    interferogram : np.ndarray, shape (Nt_sw, N_sw)
        Heterodyne signal.  Column r = sweep r.
    f_comb : np.ndarray, shape (N_st,)
        Comb tooth frequencies (Hz).
    E0 : float
        LO field amplitude.
    Nt_sw : int
        Samples per sweep.
    dt : float
        Sampling interval (s).
    f_rep : float
        Comb spacing (Hz).
    N_sw : int
        Number of sweeps.
    detector : Detector or None
        Optional detector for deconvolution.

    Returns
    -------
    result : dict
        'f_optical'     : stitched optical frequency axis (Hz)
        'Ps_stitched'   : stitched power spectrum
        'f_IF'          : per-tooth IF axis (Hz)
        'mini_spectra'  : per-tooth phase-interleaved mini-spectra
    """
    N_st   = len(f_comb)
    Nps    = Nt_sw // N_st
    T_sw   = Nt_sw * dt
    T_tot  = N_sw * T_sw
    omega_rep = 2.0 * np.pi / T_tot          # fine-grid spacing

    f_IF_step  = np.fft.fftshift(np.fft.fftfreq(Nps, d=dt))
    guard_mask = np.abs(f_IF_step) < f_rep / 2.0
    r_arr = np.arange(N_sw)

    f_optical_list, Ps_list, mini_spectra = [], [], {}

    for p in range(N_st):
        j0  = p * Nps
        j1  = (p + 1) * Nps
        tau = dt * np.arange(Nps)

        # Per-sweep FFT of the raw (baseband) interferogram
        # ---------------------------------------------------
        # The carrier exp(i*2*pi*f_p*tau) was already removed by mixing.
        # DO NOT divide by exp(-i*2*pi*f_p*tau) here — that would
        # up-convert the IF signal to optical frequency and alias it
        # outside the guard band.
        F_sweeps = np.zeros((N_sw, Nps), dtype=np.complex128)
        for r in range(N_sw):
            I_pr = interferogram[j0:j1, r].copy()
            if detector is not None:
                I_pr = detector.deconvolve(I_pr, dt)
            F_sweeps[r] = np.fft.fftshift(np.fft.fft(I_pr)) / Nps

        for s_n in range(N_sw):
            # Inner Cooley-Tukey twiddle (V4.0 correction):
            # centers the inner DFT at the fine-grid frequency
            # omega_n^(c) = omega_k + s_n * omega_rep.
            # For baseband signal at IF f_IF = k0/T_step + s_n/T_tot,
            # this removes the sub-bin phase from the inner sum.
            fine_twiddle = np.exp(-1j * s_n * omega_rep * tau)  # shape (Nps,)

            F_tw = np.zeros((N_sw, Nps), dtype=np.complex128)
            for r in range(N_sw):
                F_tw[r] = np.fft.fftshift(
                    np.fft.fft(
                        (interferogram[j0:j1, r] * fine_twiddle)
                    )
                ) / Nps

            # Outer sweep DFT — negative twiddle (corrected sign)
            # Signal inter-sweep phase: exp(+2*pi*i*s_n*r/N_sw)
            # Extraction twiddle must be conjugate: exp(-2*pi*i*s_n*r/N_sw)
            sweep_tw = np.exp(-2j * np.pi * s_n * r_arr / N_sw)
            S_fine   = (1.0 / N_sw) * np.einsum('r,rk->k', sweep_tw, F_tw)
            P_fine   = np.abs(S_fine) ** 2 / (np.abs(E0) ** 2)

            f_opt = f_comb[p] + f_IF_step[guard_mask] + s_n / T_tot
            f_optical_list.append(f_opt)
            Ps_list.append(P_fine[guard_mask])

        # Diagnostic mini-spectrum (s_n = 0, mean power over sweeps)
        mini_spectra[p] = np.mean(np.abs(F_sweeps) ** 2, axis=0) / (np.abs(E0) ** 2)

    f_optical   = np.concatenate(f_optical_list)
    Ps_stitched = np.concatenate(Ps_list)
    idx         = np.argsort(f_optical)
    return {
        'f_optical'   : f_optical[idx],
        'Ps_stitched' : Ps_stitched[idx],
        'f_IF'        : f_IF_step,
        'mini_spectra': mini_spectra,
    }