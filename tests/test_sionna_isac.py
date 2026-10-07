"""
Tests for the processing functions built on top of Sionna's ISAC/OFDM utilities
(``steering_vectors``, ``ofdm_to_time_channel``).

The reference implementations below are the previous pure-NumPy ones, kept here as oracles.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.signal.windows import get_window

sys.path.insert(0, str(Path(__file__).parent))
import cases  # noqa: E402
from cissir.beamforming import steer_vec, array_factor  # noqa: E402
from cissir.sigproc import MatchedFilter  # noqa: E402


# ---------------------------------------------------------------------------------------------
# Reference implementations (previous NumPy versions)
# ---------------------------------------------------------------------------------------------

def np_steer_vec(n_elements, thetas_rad, electric_length=0.5, centered=False):
    phase_delta = electric_length * 2 * np.pi * np.sin(thetas_rad)
    el_array = np.arange(n_elements, dtype=float)
    if centered:
        el_array -= (n_elements - 1) / 2
    phase = el_array[:, np.newaxis] * np.expand_dims(phase_delta, axis=0)
    return np.exp(1j * phase)


def np_array_factor(antenna_weights, thetas_radian, n_antennas=None, transmit=True):
    if n_antennas is None:
        n_antennas = len(antenna_weights)
    d_sign = 1 if transmit else -1
    return np.array([np_steer_vec(n_antennas, d_sign * d).reshape(1, n_antennas) @ antenna_weights
                     for d in thetas_radian]).squeeze()


def np_fft(x, axis=-1):
    return np.fft.fftshift(np.fft.fft(x, axis=axis), axes=axis)


def np_ifft(x, axis=-1):
    return np.fft.ifft(np.fft.ifftshift(x, axes=axis), axis=axis)


def np_matched_filter(signal, reference, window_vals, time_input, time_output, axis=-1, return_filter=False):
    if time_input:
        signal = np_fft(signal, axis=axis)
        reference = np_fft(reference, axis=axis)
    x_win = np.conj(reference) * window_vals
    h_freq = signal * x_win
    if time_output:
        h_freq = np_ifft(h_freq, axis=axis)
        if return_filter:
            x_win = np_ifft(x_win, axis=axis)
    return (h_freq, x_win) if return_filter else h_freq


# ---------------------------------------------------------------------------------------------
# Steering vectors
# ---------------------------------------------------------------------------------------------

ANGLES = {
    "scalar": 0.3,
    "scalar_neg": -0.7,
    "array": np.deg2rad(np.linspace(-90, 90, 13)),
    "wide": np.deg2rad(np.array([-170.0, -120.0, 0.0, 95.0, 180.0])),
}


@pytest.mark.parametrize("angles", ANGLES)
@pytest.mark.parametrize("n_elements", [1, 4, 8])
@pytest.mark.parametrize("electric_length", [0.5, 0.37])
@pytest.mark.parametrize("centered", [False, True])
def test_steer_vec(angles, n_elements, electric_length, centered):
    theta = ANGLES[angles]
    out = steer_vec(n_elements, theta, electric_length=electric_length, centered=centered)
    expected = np_steer_vec(n_elements, theta, electric_length=electric_length, centered=centered)
    assert out.shape == expected.shape == (n_elements, np.size(theta))
    assert out.dtype == np.complex128
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("transmit", [True, False])
@pytest.mark.parametrize("n_antennas", [4, 8])
def test_array_factor(transmit, n_antennas):
    weights = cases.rand_complex((n_antennas,), seed=800 + n_antennas, dtype=np.complex128)
    thetas = np.linspace(-np.pi, np.pi, 1000)
    out = array_factor(weights, thetas, transmit=transmit)
    expected = np_array_factor(weights, thetas, transmit=transmit)
    assert out.shape == expected.shape == (1000,)
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-11)


def test_array_factor_beam_peak():
    """Matched weights have their maximum array factor in the steered direction, with unit gain"""
    n = 8
    theta0 = np.deg2rad(20.0)
    weights = steer_vec(n, theta0).conj().squeeze() / n
    thetas = np.deg2rad(np.linspace(-90, 90, 3601))
    af = np.abs(array_factor(weights, thetas, transmit=True))
    assert abs(thetas[np.argmax(af)] - theta0) < np.deg2rad(0.1)
    assert af.max() == pytest.approx(1.0, rel=1e-9)


# ---------------------------------------------------------------------------------------------
# Matched filter
# ---------------------------------------------------------------------------------------------

NUM_SC, GUARD, BATCH = 60, 10, (3, 2)
FFT = NUM_SC + 2 * GUARD

WINDOWS = {"none": (None, ()), "hann": ("hann", ()), "kaiser": ("kaiser", (6.0,))}


def window_values(name):
    window, win_args = WINDOWS[name]
    if window is None:
        return 1.0
    win_arg = window if len(win_args) == 0 else (window,) + win_args
    return np.pad(get_window(win_arg, NUM_SC, fftbins=True), GUARD, constant_values=0)


@pytest.mark.parametrize("dtype, tol", [(np.complex64, 5e-6), (np.complex128, 1e-12)])
@pytest.mark.parametrize("return_filter", [False, True])
@pytest.mark.parametrize("time_output", [False, True])
@pytest.mark.parametrize("time_input", [False, True])
@pytest.mark.parametrize("window", WINDOWS)
def test_matched_filter(window, time_input, time_output, return_filter, dtype, tol):
    signal = cases.rand_complex(BATCH + (FFT,), seed=810, dtype=dtype)
    reference = cases.rand_complex(BATCH + (FFT,), seed=811, dtype=dtype)
    win_name, win_args = WINDOWS[window]
    mf = MatchedFilter(NUM_SC, GUARD, *win_args, window=win_name, time_input=time_input, time_output=time_output)

    out = mf.filter(torch.from_numpy(signal), torch.from_numpy(reference), return_filter=return_filter)
    expected = np_matched_filter(signal.astype(np.complex128), reference.astype(np.complex128), window_values(window),
                                 time_input, time_output, return_filter=return_filter)
    out = out if return_filter else (out,)
    expected = expected if return_filter else (expected,)
    for o, e in zip(out, expected):
        assert isinstance(o, torch.Tensor) and o.dtype == torch.from_numpy(signal).dtype
        assert tuple(o.shape) == e.shape
        assert np.abs(o.numpy() - e).max() <= tol * np.abs(e).max()


def test_matched_filter_numpy_inputs_and_axis():
    """NumPy arrays are accepted and any axis can be filtered (the window is applied along that axis)"""
    signal = cases.rand_complex(BATCH + (FFT,), seed=812)
    reference = cases.rand_complex(BATCH + (FFT,), seed=813)
    mf = MatchedFilter(NUM_SC, GUARD, window="hann", time_input=False, time_output=True)
    ref_out = mf.filter(signal, reference)
    assert isinstance(ref_out, torch.Tensor)
    for axis in (0, -2):
        out = mf.filter(np.moveaxis(signal, -1, axis), np.moveaxis(reference, -1, axis), axis=axis)
        np.testing.assert_allclose(np.moveaxis(out.numpy(), axis, -1), ref_out.numpy(), rtol=0, atol=1e-6)


def test_matched_filter_recovers_delay():
    """Matched filtering a delayed copy of the reference peaks at the delay, with the reference energy as gain"""
    delay, n_sc, fft_size = 17, 400, 512
    guard = (fft_size - n_sc) // 2
    freqs = np.fft.fftshift(np.fft.fftfreq(fft_size))                    # centered normalized frequencies
    reference = np.zeros(fft_size, dtype=np.complex64)
    reference[guard:guard + n_sc] = cases.rand_complex((n_sc,), seed=814)
    signal = reference * np.exp(-2j * np.pi * freqs * delay)             # delay in the frequency domain
    mf = MatchedFilter(n_sc, guard, time_input=False, time_output=True)
    profile = mf.filter(torch.from_numpy(signal), torch.from_numpy(reference)).abs().numpy()
    assert profile.argmax() == delay
    assert profile.max() == pytest.approx(np.sum(np.abs(reference) ** 2) / fft_size, rel=1e-4)
