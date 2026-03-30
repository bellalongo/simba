# helpers.py
import numpy as np


# ---------- Random walk for phase-noise modelling ----------
def complex_random_walk(N: int, std: float, rng: np.random.Generator):
    """
    Cumulative sum of Gaussian increments (discrete Wiener process).

    For a signal with Lorentzian linewidth FWHM, the per-step
    standard deviation is  std = sqrt(2 * pi * dt * FWHM).

    Arguments:
        N:   number of time steps
        std: standard deviation of each Gaussian increment (rad)
        rng: numpy random generator instance

    Returns:
        phi: array of shape (N,), cumulative phase (rad)
    """
    return np.cumsum(rng.normal(0.0, std, size=N))


# ---------- Analytical lineshapes (same forms as Burghoff MATLAB) ----------
def lorentzian_psd(fv, f0, As, Ac, FW, Nt, dt):
    """
    Analytical PSD of a Lorentzian-broadened line in the DFT spectrum.

    This is the *non-periodic* (infinite-bandwidth) form. Appropriate
    when the linewidth is much smaller than the Nyquist frequency 1/(2*dt).

    Derivation: The DFT of A * exp(i*phi(t)), where phi is a Wiener
    process with diffusion constant D = pi * FW, yields a Lorentzian
    PSD normalised by 1/Nt (the DFT forward prefactor squared).

    Arguments:
        fv: frequency axis (Hz), array
        f0: line centre frequency (Hz)
        As: signal field amplitude
        Ac: LO (or comb) field amplitude (for heterodyne normalisation)
        FW: full-width at half-maximum (Hz)
        Nt: number of time samples in the DFT window
        dt: sampling interval (s)

    Returns:
        L(fv): PSD values, same shape as fv
    """
    num = 2 * np.pi * dt * FW
    den = (np.pi * dt * FW) ** 2 + ((fv - f0) * (2 * np.pi * dt)) ** 2
    return (1.0 / Nt) * (num / den) * (np.abs(As) ** 2) * (np.abs(Ac) ** 2)


def lorentzian_psd_periodic(fv, f0, As, Ac, FW, Nt, dt):
    """
    Periodic (aliased) Lorentzian PSD for the discrete DFT.

    Uses the sinh/cosh closed form that accounts for spectral
    aliasing when the line is broad relative to the Nyquist band.
    Includes a hard gate at |f - f0| <= 1/(2*dt) (first Brillouin zone).

    Arguments:
        fv, f0, As, Ac, FW, Nt, dt: same as lorentzian_psd

    Returns:
        L(fv): PSD values with periodic aliasing correction
    """
    num = np.sinh(np.pi * dt * FW)
    den = np.cosh(np.pi * dt * FW) - np.cos((fv - f0) * (2 * np.pi * dt))
    gate = (np.abs(fv - f0) <= 1.0 / (2 * dt)).astype(float)
    return (1.0 / Nt) * (num / den) * (np.abs(As) ** 2) * (np.abs(Ac) ** 2) * gate


# ---------- Smoothing ----------
def boxcar_smooth(y, N):
    """
    Boxcar (moving-average) smoothing filter.

    Arguments:
        y: input array
        N: window width (integer >= 1)

    Returns:
        Smoothed array of length len(y) - N + 1 (valid convolution).
    """
    if N <= 1:
        return y
    kernel = np.ones(N) / N
    return np.convolve(y, kernel, mode='valid')