# mixing.py
"""
Heterodyne mixing engine for the swept-LO simulation.

Builds the time-domain interferogram:
    S_r(tau_j) = epsilon_r*(tau_j) . E_s(r * T_sw + tau_j)

For a STEPPED LO, the LO is a single tone during each step p.
The beat frequency between signal line m and step p is:

    f_beat = f_signal_m  -  f_comb[p]

Only pairs with  |f_beat| <= B  contribute to the detected signal
(the detector bandwidth clips the rest).

Implementation note
-------------------
Like testin3.py, the simulation stores signal envelopes in baseband
and applies the carrier phase analytically:

    S[j] = E0* * sum_m  env_m[j_abs] * exp(i * 2*pi * df_m * t_abs)

where  df_m = f_signal_m - f_LO(tau_j)  is the IF beat frequency,
and  t_abs = r * T_sw + j * dt  is the absolute measurement time.

See V4.0 Eq. (163):
    S_opt(tau_j) = eps_r*(tau_j) . E_s(r*T_sw + tau_j)
"""
import numpy as np


def build_interferogram_single_lo(
    signal,
    f_comb: np.ndarray,
    E0: float,
    Nt_sw: int,
    dt: float,
    B: float,
    N_sw: int = 1,
):
    """
    Build the heterodyne interferogram for a stepped LO.

    For each sweep r and each sample j within the sweep, the LO dwells
    at ONE frequency (the step that covers sample j).  The interferogram
    is the sum of all (signal-line, LO-step) beats that fall within
    the detector bandwidth.

    Arguments
    ---------
    signal : Signal
        Signal object with .envelopes (n_total, n_lines) already generated,
        and .frequencies giving the carrier frequencies.
    f_comb : np.ndarray, shape (N_st,)
        LO step frequencies (Hz).
    E0 : float
        LO field amplitude.
    Nt_sw : int
        Number of samples per sweep.
    dt : float
        Sampling interval (s).
    B : float
        One-sided detector bandwidth (Hz).
    N_sw : int
        Number of sweeps (default 1).

    Returns
    -------
    I : np.ndarray, shape (Nt_sw, N_sw), complex128
        Heterodyne interferogram.  Column r = sweep r.
        Follows the testin3 convention  (fast_time, slow_time).
    """
    N_st = len(f_comb)
    N_s = signal.n_lines
    n_total = Nt_sw * N_sw

    # --- time axes ---
    # Absolute time for all samples
    t_abs = dt * np.arange(n_total)  # shape (n_total,)

    # --- allocate output ---
    I_flat = np.zeros(n_total, dtype=np.complex128)

    # --- which step is each sample in? ---
    # For a stepped LO, the LO steps through N_st teeth per sweep.
    # Samples-per-step in one sweep:
    Nt_per_step = Nt_sw // N_st

    # --- loop over signal lines and comb teeth ---
    for m in range(N_s):
        f_m = signal.frequencies[m]
        env_m = signal.envelopes[:, m]  # shape (n_total,)

        for p in range(N_st):
            f_p = f_comb[p]
            df = f_m - f_p  # IF beat frequency

            # Bandwidth gate: only include if beat is within detector range
            if np.abs(df) > B:
                continue

            # This step p is active during samples [p*Nt_per_step, (p+1)*Nt_per_step)
            # within EACH sweep.
            for r in range(N_sw):
                # Global sample indices for step p of sweep r
                sweep_offset = r * Nt_sw
                j0 = sweep_offset + p * Nt_per_step
                j1 = sweep_offset + min((p + 1) * Nt_per_step, Nt_sw)
                idx = slice(j0, j1)

                # Contribution:  E0* . env_m . exp(i 2 pi df t_abs)
                I_flat[idx] += (
                    np.conj(E0) * env_m[idx]
                    * np.exp(1j * 2 * np.pi * df * t_abs[idx])
                )

    # Reshape to (Nt_sw, N_sw) using Fortran order so that
    # column r contains sweep r (same convention as testin3)
    I = I_flat.reshape(Nt_sw, N_sw, order='F')

    return I


def build_interferogram_comb_lo(
    signal,
    f_comb: np.ndarray,
    Ec: np.ndarray,
    Nt_sw: int,
    dt: float,
    B: float,
    N_sw: int = 1,
):
    """
    Build the heterodyne interferogram for a simultaneous-comb LO.

    All comb teeth are present at every time sample (like ptychoscopy
    comb 1).  This is used for comparison / validation against the
    ptychoscopy result.

    Arguments
    ---------
    signal : Signal
        Signal object with envelopes already generated.
    f_comb : np.ndarray, shape (N_c,)
        Comb tooth frequencies (Hz).
    Ec : np.ndarray, shape (N_c,), complex128
        Complex comb tooth amplitudes.
    Nt_sw : int
        Samples per batch/sweep.
    dt : float
        Sampling interval (s).
    B : float
        Detector bandwidth (Hz).
    N_sw : int
        Number of batches/sweeps.

    Returns
    -------
    I : np.ndarray, shape (Nt_sw, N_sw), complex128
    """
    N_c = len(f_comb)
    N_s = signal.n_lines
    n_total = Nt_sw * N_sw

    t_abs = dt * np.arange(n_total)
    I_flat = np.zeros(n_total, dtype=np.complex128)

    # All comb teeth beat with all signal lines at every sample
    for n in range(N_c):
        for m in range(N_s):
            df = f_comb[n] - signal.frequencies[m]
            if np.abs(df) > B:
                continue

            I_flat += (
                Ec[n] * np.conj(signal.envelopes[:, m])
                * np.exp(1j * 2 * np.pi * df * t_abs)
            )

    I = I_flat.reshape(Nt_sw, N_sw, order='F')
    return I