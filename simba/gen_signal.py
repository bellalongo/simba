# gen_signal.py
import numpy as np

from .error_check import check_signal_inputs
from .helpers import complex_random_walk


class Signal:
    """
    Optical signal as a sum of Lorentzian-broadened spectral lines.

    Physical model
    --------------
    E_s(t) = sum_k  A_k  exp(i * 2*pi * f_k * t  +  i * phi_k(t))

    where phi_k(t) is a discrete Wiener process (cumulative Gaussian
    random walk) with per-step variance  2*pi * FWHM_k * dt.  This
    produces a Lorentzian power spectral density with linewidth FWHM_k.

    Implementation note
    -------------------
    Because the carrier frequencies f_k are in the THz range while the
    sampling rate is set by the detector bandwidth (~GHz), we CANNOT
    sample E_s(t) directly.  Instead we store:

        * self.frequencies  — carrier frequencies f_k  (Hz)
        * self.envelopes    — baseband envelopes  A_k * exp(i * phi_k(t))

    The carrier phase exp(i * 2*pi * f_k * t) is applied analytically
    when building the heterodyne interferogram (see mixing.py).
    This is identical to the approach used in Burghoff's ptychoscopy
    simulation (testin3.py).

    Arguments
    ---------
    frequencies : array or None
        Carrier frequencies (Hz).  If None, uses default test lines
        matching the ptychoscopy code.
    amplitudes : array or None
        Field amplitudes (V/m).
    fwhms : array or None
        Lorentzian linewidths (Hz, FWHM).
    """

    def __init__(
        self,
        frequencies: np.ndarray = None,
        amplitudes: np.ndarray = None,
        fwhms: np.ndarray = None
    ):
        if frequencies is not None:
            check_signal_inputs(frequencies, amplitudes, fwhms)
            self.frequencies = np.asarray(frequencies, dtype=np.float64)
            self.amplitudes = np.asarray(amplitudes, dtype=np.float64)
            self.fwhms = np.asarray(fwhms, dtype=np.float64)
        else:
            # Default: same lines as ptychoscopy test case
            self.frequencies = np.array([4.211e12, 4.558e12,
                                         4.783e12, 4.803e12])
            self.amplitudes = np.array([1.0, 1.5, 2.0, 2.5]) * 1e-5
            self.fwhms = np.array([10e6, 10e6, 10e6, 10e6]) * 1000

        self.n_lines = len(self.frequencies)
        self.envelopes = None
        self.phase_walks = None

    def generate_envelopes(self, n_total: int, dt: float,
                           rng: np.random.Generator):
        """
        Generate baseband envelopes for all signal lines.

        Produces  env_k[m] = A_k * exp(i * phi_k(m))  where phi_k is
        a random-walk phase with  <(Delta phi)^2> = 2*pi*FWHM_k*dt
        per step.

        Arguments:
            n_total: total number of time samples across all sweeps
            dt:      sampling interval (s)
            rng:     numpy random number generator

        Sets:
            self.envelopes   — shape (n_total, n_lines), complex128
            self.phase_walks — shape (n_total, n_lines), float64
        """
        self.phase_walks = np.zeros((n_total, self.n_lines))
        self.envelopes = np.zeros((n_total, self.n_lines),
                                  dtype=np.complex128)

        for k in range(self.n_lines):
            std = np.sqrt(2 * np.pi * dt * self.fwhms[k])
            self.phase_walks[:, k] = complex_random_walk(n_total, std, rng)
            self.envelopes[:, k] = (
                self.amplitudes[k] * np.exp(1j * self.phase_walks[:, k])
            )

    def scale_amplitudes(self, factor: float):
        """
        Multiply all amplitudes by a common factor (e.g. sqrt(power)).

        Also updates envelopes if they have already been generated.
        """
        self.amplitudes = self.amplitudes * factor
        if self.envelopes is not None:
            self.envelopes = self.envelopes * factor