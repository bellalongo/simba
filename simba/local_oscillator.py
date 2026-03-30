# local_oscillator.py
"""
Local oscillator models for heterodyne spectroscopy simulation.

Two LO types are implemented:

1. SteppedLO  --  a single-sweep staircase LO with N_st frequency steps.
   During step p the LO is a pure tone:
       epsilon(tau) = E0 * exp(i * 2*pi * f_p * tau)
   where f_p = f_start + p * f_step,  tau in [0, T_step).

2. SweptSteppedLO  --  N_sw identical sweeps of the stepped LO.
   Each sweep executes the SAME staircase.  The resolution enhancement
   comes from the signal's inter-sweep phase accumulation (phase
   interleaving), not from any LO variation between sweeps.

Both classes provide the LO field epsilon_r(tau_j) evaluated on
the fast-time sampling grid, which is needed for:
    (a) building the heterodyne interferogram  S = eps* . E_s, and
    (b) time-domain division  E_s = S_opt / eps*  (V4.0 pipeline).

See V1.3 Section 2 for the stepped-LO notation:
    f_p = f_0^(LO) + p * f_step,   p = 0, ..., N_st - 1
"""
import numpy as np
from .error_check import check_lo_inputs


class SteppedLO:
    """
    Single-sweep stepped local oscillator.

    Attributes
    ----------
    f_comb : np.ndarray, shape (N_st,)
        Comb tooth / step frequencies [Hz].
    E0 : float
        Constant LO field amplitude.
    N_st : int
        Number of frequency steps per sweep.
    """

    def __init__(self, f_comb: np.ndarray, E0: float):
        """
        Arguments:
            f_comb: array of LO step frequencies (Hz), length N_st
            E0:     LO field amplitude (V/m)
        """
        check_lo_inputs(f_comb, E0)
        self.f_comb = np.asarray(f_comb, dtype=np.float64)
        self.E0 = E0
        self.N_st = len(f_comb)

    def field(self, tau: np.ndarray, Nt_per_step: int) -> np.ndarray:
        """
        Evaluate the LO field epsilon(tau_j) over one sweep.

        During step p (samples [p*Nt_per_step, (p+1)*Nt_per_step)):
            epsilon(tau_j) = E0 * exp(i * 2*pi * f_p * tau_j)

        Arguments:
            tau:          local time axis for one sweep (s), shape (Nt_sw,)
            Nt_per_step:  samples per frequency step

        Returns:
            eps: complex LO field, shape (Nt_sw,)
        """
        Nt_sw = len(tau)
        eps = np.zeros(Nt_sw, dtype=np.complex128)

        for p in range(self.N_st):
            j0 = p * Nt_per_step
            j1 = min((p + 1) * Nt_per_step, Nt_sw)
            f_p = self.f_comb[p]
            eps[j0:j1] = self.E0 * np.exp(1j * 2 * np.pi * f_p * tau[j0:j1])

        return eps

    def step_index(self, j: int, Nt_per_step: int) -> int:
        """
        Return which step index p a sample j belongs to.
        """
        return min(j // Nt_per_step, self.N_st - 1)


class SweptSteppedLO:
    """
    Multi-sweep stepped local oscillator.

    Every sweep executes the same staircase pattern.  The fine-grid
    resolution arises from the signal's sweep-to-sweep phase
    accumulation  exp(i * omega_n^(c) * r * T_sw), NOT from any LO
    variation.

    This class wraps SteppedLO and adds sweep bookkeeping.

    Attributes
    ----------
    stepped : SteppedLO
        Underlying single-sweep LO.
    N_sw : int
        Number of sweeps.
    """

    def __init__(self, f_comb: np.ndarray, E0: float, N_sw: int):
        """
        Arguments:
            f_comb: step frequencies (Hz)
            E0:     LO field amplitude (V/m)
            N_sw:   number of sweeps
        """
        self.stepped = SteppedLO(f_comb, E0)
        self.N_sw = N_sw

    @property
    def f_comb(self):
        return self.stepped.f_comb

    @property
    def E0(self):
        return self.stepped.E0

    @property
    def N_st(self):
        return self.stepped.N_st

    def field_sweep(self, r: int, tau: np.ndarray,
                    Nt_per_step: int) -> np.ndarray:
        """
        LO field during sweep r.  Identical for all r.

        Arguments:
            r:            sweep index (unused; every sweep is the same)
            tau:          local time axis (s)
            Nt_per_step:  samples per step

        Returns:
            eps_r: complex field, shape (Nt_sw,)
        """
        return self.stepped.field(tau, Nt_per_step)