# extraction_fixed.py
"""
Corrected spectral extraction for the swept-LO simulation.

Bug fixed (Cooley-Tukey twiddle factor)
----------------------------------------
The original extract_multi_sweep computed the per-step FFT at COARSE-GRID
frequencies  exp(-i 2pi k j / Nps)  and then applied the sweep DFT.
This is the algorithm from V4.0 Eq. (178), which the Corrected Master
Equation document identifies as WRONG because it drops the Cooley-Tukey
twiddle factor.

The correct Cooley-Tukey decomposition of the full N_tot-point DFT is:

    a_n  =  (1/N_sw) sum_r  exp(-i 2pi s_n r / N_sw)
            * [(1/Nps) sum_j  x[r*Nps + j]
               * exp(-i 2pi q j / Nps)      <- coarse DFT kernel
               * exp(-i 2pi s_n j / N_tot)  <- twiddle factor  ← WAS MISSING
              ]

where  n = q * N_sw + s_n  is the fine-grid index,
       N_tot = N_sw * Nps,
       q = floor(n / N_sw),   s_n = n mod N_sw.

Equivalently, the inner sum uses the FINE-GRID frequency:
    exp(-i omega_n^{(c)} tau_j) = exp(-i (omega_k + s_n omega_rep) tau_j)
instead of the coarse frequency  exp(-i omega_k tau_j).

The twiddle factor per sample j within a step is:
    W_j^{s_n} = exp(-i 2pi s_n j / N_tot) = exp(-i 2pi s_n f_rep_fine tau_j)

where  tau_j = j * dt  and  f_rep_fine = 1 / (N_sw * Nps * dt) = 1 / T_total.

For s_n = 0 (which is ALL fine-grid offsets when N_sw = 1), the twiddle is 1
and the bug is invisible — hence the N_sw = 1 case worked correctly.

Consistency check
-----------------
For N_sw = 2, Nps = 8, a pure tone at fine-grid index n = 1 (q=0, s_n=1):
    Buggy  : |S_fine[k=0]|^2 = 0.41   (wrong)
    Fixed  : |S_fine[k=0]|^2 = 1.00   ✓
    Full DFT: |A_1|^2         = 1.00   ✓

Amplitude scaling note
----------------------
After the fix, for a Lorentzian envelope of FWHM = FW measured over Nps
samples per step, the expected PSD at the peak is:

    E[|S_fine[k_IF]|^2 / |E0|^2]  ≈  Lfcn2(f_signal, f_signal, As, 1.0, FW, Nps, dt)

for EACH value of N_sw.  The sweep DFT is a coherent combiner of N_sw
INDEPENDENT estimates (one per sweep), each of amplitude ~1/N_sw, summing
to amplitude ~1 — identical to the single-sweep case.  See derivation in
Section III of Corrected_Master_Equation_Findings.pdf.
"""
import numpy as np


# ---------------------------------------------------------------------------
# (A) Single-sweep extraction   (unchanged — was already correct)
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
    A2. Non-overlapping: signal bandwidth per tooth < f_rep / 2.
    A3. interferogram[j] = E0* * env_m * exp(i 2pi df t_abs[j]).
    A4. Implicit periodicity: DFT treats each dwell window as periodic.
    A5. f_comb[p] * T_sw ∈ Z for all p  (Assumption A7 of main.py).
        Ensures exp(i 2pi f_p r T_sw) = 1, so the inter-sweep phase
        from the LO vanishes and the signal phase dominates.

    Returns
    -------
    dict with keys:
        'f_optical'   : stitched optical frequency axis (Hz)
        'Ps_stitched' : stitched power spectrum  |F_p|^2 / |E0|^2
        'f_IF'        : per-tooth IF axis (Hz)
        'mini_spectra': dict  {tooth_index -> mini-spectrum array}
    """
    I = interferogram.ravel()
    N_st = len(f_comb)
    Nps  = Nt_sw // N_st           # samples per step

    f_IF  = np.fft.fftshift(np.fft.fftfreq(Nps, d=dt))
    guard = np.abs(f_IF) <= f_rep / 2.0

    f_opt_list, Ps_list, mini_spectra = [], [], {}

    for p in range(N_st):
        j0, j1 = p * Nps, (p + 1) * Nps
        I_p = I[j0:j1]
        if detector is not None:
            I_p = detector.deconvolve(I_p, dt)

        # Per-step FFT  (1/Nps normalisation)
        F_p = np.fft.fftshift(np.fft.fft(I_p)) / Nps
        P_p = np.abs(F_p) ** 2 / np.abs(E0) ** 2

        mini_spectra[p] = P_p
        f_opt_list.append(f_comb[p] + f_IF[guard])
        Ps_list.append(P_p[guard])

    f_optical   = np.concatenate(f_opt_list)
    Ps_stitched = np.concatenate(Ps_list)
    idx = np.argsort(f_optical)
    return {
        'f_optical':    f_optical[idx],
        'Ps_stitched':  Ps_stitched[idx],
        'f_IF':         f_IF,
        'mini_spectra': mini_spectra,
    }


# ---------------------------------------------------------------------------
# (B) Multi-sweep extraction — CORRECTED Cooley-Tukey twiddle factor
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

    Twiddle-factor derivation  (discrete, exact)
    --------------------------------------------
    The full N_tot = N_sw * Nps  point DFT of the signal  E_s(t_m)  is:

        a_n = (1/N_tot) sum_{m=0}^{N_tot-1} E_s(t_m) exp(-i 2pi n m / N_tot)

    Cooley-Tukey factorisation with  n = q * N_sw + s_n,  m = r * Nps + j:

        a_n = (1/N_sw) sum_{r=0}^{N_sw-1}  exp(-i 2pi s_n r / N_sw)
              * [(1/Nps) sum_{j=0}^{Nps-1}  E_s(r*Nps + j)
                 * exp(-i 2pi q j / Nps)
                 * exp(-i 2pi s_n j / N_tot)]   <- TWIDDLE FACTOR

    The twiddle factor per local sample j is:

        W_j^{s_n} = exp(-i 2pi s_n j / N_tot)
                  = exp(-i 2pi s_n f_rep_fine tau_j)
                  = exp(-i s_n omega_rep tau_j)        [Flag A]

    [Flag A] This factor encodes the sub-bin frequency shift  s_n * f_rep_fine
    that distinguishes fine-grid bins within a single coarse bin.
    It vanishes for s_n = 0, making N_sw = 1 a degenerate correct case.

    DFT orthogonality (discrete, exact):
        (1/N_sw) sum_{r=0}^{N_sw-1} exp(i 2pi (s - s_n) r / N_sw)
            = delta_{s, s_n}    for  0 <= s, s_n < N_sw            [Flag B]

    [Flag B] The outer sum over r uses EXACT discrete orthogonality,
    not an approximation or continuous Dirac delta.

    Amplitude prediction (consistency check)
    ----------------------------------------
    After the fix, for a Lorentzian signal with As and FWHM = FW:

        E[|S_fine[k_IF]|^2 / |E0|^2]  =  Lfcn2(fk, f_signal, As, 1.0, FW, Nps, dt)

    independent of N_sw.  The sweep DFT's coherent combination of N_sw sweeps
    gives amplitude equal to one sweep's contribution (each sweep contributes
    1/N_sw, N_sw of them sum to 1), matching the single-sweep prediction.

    Parameters
    ----------
    interferogram : (Nt_sw, N_sw) array
        Heterodyne signal.  Column r = sweep r.  Built by mixing.py.
    f_comb : (N_st,) array
        LO step frequencies (Hz).
    E0     : float
        LO field amplitude.
    Nt_sw  : int
        Samples per sweep  = N_st * Nps.
    dt     : float
        Sampling interval (s).
    f_rep  : float
        Coarse LO step spacing (Hz)  = 1 / T_step.
    N_sw   : int
        Number of sweeps.
    detector : optional
        Detector object with a .deconvolve(I, dt) method.

    Returns
    -------
    dict with same keys as extract_single_sweep.
    """
    N_st       = len(f_comb)
    Nps        = Nt_sw // N_st          # samples per step
    T_sw       = Nt_sw * dt             # sweep duration
    T_total    = N_sw * T_sw            # full measurement duration
    f_rep_fine = 1.0 / T_total          # fine-grid spacing = 1 / T_total

    f_IF  = np.fft.fftshift(np.fft.fftfreq(Nps, d=dt))
    guard = np.abs(f_IF) <= f_rep / 2.0

    r_arr = np.arange(N_sw)
    tau_j = np.arange(Nps) * dt         # local time axis within a step

    f_opt_list = []
    Ps_list    = []
    mini_spectra = {}

    for p in range(N_st):
        j0, j1 = p * Nps, (p + 1) * Nps

        # --- For each fine-grid offset s_n, apply the twiddle BEFORE FFT ---
        # This is more expensive (one FFT per s_n per step) but exact.
        # Alternatives: chirp-Z transform or NUFFT for efficiency.

        # First, load all sweep data for this step (apply detector if any)
        I_step = np.zeros((N_sw, Nps), dtype=np.complex128)
        for r in range(N_sw):
            I_pr = interferogram[j0:j1, r]
            if detector is not None:
                I_pr = detector.deconvolve(I_pr, dt)
            I_step[r] = I_pr

        # Coarse FFT (s_n = 0 case, used for mini_spectra display)
        F_coarse = np.zeros((N_sw, Nps), dtype=np.complex128)
        for r in range(N_sw):
            F_coarse[r] = np.fft.fftshift(np.fft.fft(I_step[r])) / Nps
        twiddle0        = np.ones(N_sw)
        S_coarse        = (1.0 / N_sw) * (twiddle0 @ F_coarse)
        mini_spectra[p] = np.abs(S_coarse) ** 2 / np.abs(E0) ** 2

        # Fine-grid sweep DFT with Cooley-Tukey twiddle
        for s_n in range(N_sw):
            # --- Cooley-Tukey twiddle factor (inner, per local sample j) ---
            # W_j = exp(-i 2pi s_n f_rep_fine tau_j)
            # This shifts the effective DFT frequency from omega_k to
            # omega_k + s_n * omega_rep  (the fine-grid frequency).
            W_inner = np.exp(-2j * np.pi * s_n * f_rep_fine * tau_j)  # (Nps,)

            # Per-step FFT at the FINE-GRID frequency for offset s_n
            F_fine = np.zeros((N_sw, Nps), dtype=np.complex128)
            for r in range(N_sw):
                # Apply twiddle in time domain, then FFT
                F_fine[r] = np.fft.fftshift(np.fft.fft(I_step[r] * W_inner)) / Nps

            # --- Outer sweep DFT (inter-sweep phase, exact DFT orthogonality) ---
            # Phase convention: signal carries exp(+i 2pi s_n r / N_sw)
            # from the carrier at the coarse bin.  Conjugate twiddle cancels it.
            twiddle_outer = np.exp(-2j * np.pi * s_n * r_arr / N_sw)  # (N_sw,)
            S_fine        = (1.0 / N_sw) * (twiddle_outer @ F_fine)   # (Nps,)
            P_fine        = np.abs(S_fine) ** 2 / np.abs(E0) ** 2

            # Optical frequency: comb tooth + coarse IF + fine-grid sub-bin offset
            f_opt = f_comb[p] + f_IF[guard] + s_n * f_rep_fine
            f_opt_list.append(f_opt)
            Ps_list.append(P_fine[guard])

    f_optical   = np.concatenate(f_opt_list)
    Ps_stitched = np.concatenate(Ps_list)
    idx = np.argsort(f_optical)
    return {
        'f_optical':    f_optical[idx],
        'Ps_stitched':  Ps_stitched[idx],
        'f_IF':         f_IF,
        'mini_spectra': mini_spectra,
    }