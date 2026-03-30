# error_check.py
import numpy as np


def check_signal_inputs(
    frequencies: np.ndarray,
    amplitudes: np.ndarray,
    fwhms: np.ndarray
):
    """
    Validate that signal inputs are consistent.

    Arguments:
        frequencies: signal carrier frequencies (Hz)
        amplitudes: signal field amplitudes (V/m)
        fwhms: Lorentzian linewidths (Hz, full-width at half-maximum)

    Raises:
        AssertionError if lengths mismatch or values are unphysical.
    """
    assert len(frequencies) == len(amplitudes) == len(fwhms), (
        f"Input length mismatch: frequencies({len(frequencies)}), "
        f"amplitudes({len(amplitudes)}), fwhms({len(fwhms)})"
    )
    assert np.all(np.array(fwhms) >= 0), "FWHMs must be non-negative."
    assert np.all(np.array(amplitudes) >= 0), "Amplitudes must be non-negative."


def check_lo_inputs(frequencies: np.ndarray, amplitude: float):
    """
    Validate LO parameters.

    Arguments:
        frequencies: LO frequencies for each sweep (Hz)
        amplitude: LO field amplitude (V/m), must be nonzero

    Raises:
        AssertionError if amplitude is zero or frequencies are empty.
    """
    assert amplitude > 0, "LO amplitude must be positive (nonzero field)."
    assert len(frequencies) > 0, "LO must have at least one frequency."


def check_detector_inputs(bandwidth: float):
    """
    Validate detector parameters.

    Arguments:
        bandwidth: detector half-bandwidth (Hz)

    Raises:
        AssertionError if bandwidth is non-positive.
    """
    assert bandwidth > 0, "Detector bandwidth must be positive."