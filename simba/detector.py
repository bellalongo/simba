# detector.py
import numpy as np

from .error_check import check_detector_inputs


class Detector:
    """
    Bandwidth-limited photodetector with transfer function H(omega).

    Physical model
    --------------
    The bare optical beating is  S_opt(t) = E_LO*(t) * E_s(t).
    The detector output (ADC voltage) is the convolution:

        V(t) = (h * S_opt)(t)

    In the frequency domain:

        V_hat(w_k) = H(w_k) * S_hat_opt(w_k)

    where H(w) = FT[h(t)] is the detector transfer function.

    Supported models
    ----------------
    'ideal'  : H(f) = 1  for |f| < BW,  0 otherwise
    'rc'     : H(f) = 1 / (1 + i*f/f_3dB)     (first-order RC)
    'butter' : H(f) = 1 / sqrt(1 + (f/f_3dB)^{2n})  (Butterworth)

    See Eq. (155)-(156) of V4.0 for the general detector model.

    Arguments
    ---------
    bandwidth : float
        Half-bandwidth (Hz).  |f| < bandwidth defines the passband.
    model : str
        Transfer function model: 'ideal', 'rc', or 'butter'.
    order : int
        Butterworth filter order (only used when model='butter').
    """

    def __init__(self, bandwidth: float, model: str = 'ideal', order: int = 4):
        check_detector_inputs(bandwidth)
        self.bandwidth = bandwidth
        self.model = model
        self.order = order

    def transfer_function(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Evaluate H(f) at the given IF frequencies.

        Arguments:
            frequencies: IF frequency array (Hz)

        Returns:
            H: complex transfer function values, same shape as input
        """
        f = np.asarray(frequencies, dtype=np.float64)

        if self.model == 'ideal':
            return (np.abs(f) <= self.bandwidth).astype(np.complex128)

        elif self.model == 'rc':
            return 1.0 / (1.0 + 1j * f / self.bandwidth)

        elif self.model == 'butter':
            n = self.order
            # Magnitude-only Butterworth (zero phase)
            mag = 1.0 / np.sqrt(1.0 + (f / self.bandwidth) ** (2 * n))
            return mag.astype(np.complex128)

        else:
            raise ValueError(f"Unknown detector model: '{self.model}'")

    def apply_bandwidth(self, signal_td: np.ndarray,
                        dt: float) -> np.ndarray:
        """
        Apply detector bandwidth to a time-domain signal.

        Computes  V(t) = IFFT[ H(f) * FFT[S_opt(t)] ]

        This is Eq. (158)-(160) of V4.0:
            S_det(w) = H(w) * S_opt(w)

        Arguments:
            signal_td: bare optical beating S_opt(tau_j), complex array
            dt:        sampling interval (s)

        Returns:
            V: detected (bandwidth-limited) signal, same shape
        """
        N = len(signal_td)
        freqs = np.fft.fftfreq(N, d=dt)
        H = self.transfer_function(freqs)

        S_hat = np.fft.fft(signal_td)
        V = np.fft.ifft(H * S_hat)

        return V

    def deconvolve(self, measured_td: np.ndarray, dt: float,
                   regularization: float = 0.0) -> np.ndarray:
        """
        Deconvolve the detector response from measured time-domain data.

        Recovers the bare optical beating:
            S_opt(tau_j) = IFFT[ V_hat(w_k) / H(w_k) ]

        This is Eq. (166) of V4.0.

        For bins where |H(w_k)| < threshold (beyond the bandwidth),
        those bins are zeroed (they carry no signal information).
        Optionally applies Wiener regularisation (Eq. 15 of the
        Corrected Master Equation document):

            V_hat / H  -->  H* * V_hat / (|H|^2 + lambda)

        Arguments:
            measured_td:    measured ADC voltage V_r[j], complex array
            dt:             sampling interval (s)
            regularization: Wiener regularisation parameter lambda

        Returns:
            S_opt: deconvolved bare optical beating, same shape
        """
        N = len(measured_td)
        freqs = np.fft.fftfreq(N, d=dt)
        H = self.transfer_function(freqs)

        V_hat = np.fft.fft(measured_td)

        if regularization > 0:
            # Wiener regularisation
            H_inv = np.conj(H) / (np.abs(H) ** 2 + regularization)
        else:
            # Hard threshold: zero bins with negligible H
            threshold = 1e-10
            H_inv = np.where(
                np.abs(H) > threshold,
                1.0 / H,
                0.0 + 0j
            )

        S_opt = np.fft.ifft(H_inv * V_hat)

        return S_opt