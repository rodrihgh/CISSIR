"""
Signal Processing Module
"""

import numpy as np
from scipy.signal.windows import get_window
import torch

from sionna.phy.channel import ofdm_to_time_channel

from cissir.utils import axes_tuple


class MatchedFilter:
    """
    Provides functionality for matched filtering in RF systems.

    The MatchedFilter class is used to perform matched filtering operations on PyTorch tensors.
    This implementation includes options for applying a specified window
    function, operating in the time domain or frequency domain, and returning
    the applied filter along with the filtered signal.

    The frequency-domain signals follow Sionna's OFDM convention, i.e., centered subcarrier ordering, and
    the transformation to the time domain is carried out with
    :func:`sionna.phy.channel.ofdm_to_time_channel`.

    Attributes:
        time_input (bool): Indicates whether the input signals are in the time domain.
        time_output (bool): Indicates whether the output signals should be in the time domain.
    """
    def __init__(self, num_subcarriers: int, num_guard_carriers: int,
                 *win_args, window=None, time_input=True, time_output=True):
        self.time_input = time_input
        self.time_output = time_output
        self._window = self._build_window(num_subcarriers, num_guard_carriers, window, *win_args)

    @staticmethod
    def _build_window(window_length, zero_pad, window, *win_args):
        if window is None:
            return 1.0
        win_arg = window if len(win_args) == 0 else (window,) + win_args
        win_vals = get_window(win_arg, window_length, fftbins=True)
        return np.pad(win_vals, zero_pad, constant_values=0)

    def filter(self, signal, reference, axis=-1, return_filter=False):
        """
        Filters a signal using a matched filter technique in the frequency domain.

        This method applies a matched filter to the input `signal` using the `reference`
        signal, with an optional `axis` parameter to specify which axis the operation
        should be performed on. The method can operate in the frequency or time domain
        depending on the object's configuration. Optionally, the method can also return
        the applied filter.

        Parameters:
        signal: The input signal to be filtered (tensor or array-like).
        reference: The reference signal used for filtering (tensor or array-like).
        axis: int, optional
            The axis along which filtering is applied, which is also the one the window is applied to.
            Default is -1.
        return_filter: bool, optional
            If True, the applied filter is returned alongside the filtered signal. Default is False.

        Returns:
            The filtered signal as a tensor. If `return_filter` is True, returns a tuple containing
            the filtered signal and the filter used.
        """
        signal = torch.movedim(torch.as_tensor(signal), axis, -1)
        reference = torch.movedim(torch.as_tensor(reference), axis, -1)

        if self.time_input:
            signal = _fft_centered(signal)
            reference = _fft_centered(reference)

        # Frequency-domain matched filter
        window = torch.as_tensor(self._window, dtype=reference.real.dtype, device=reference.device)
        x_win = reference.conj() * window
        h_freq = signal * x_win

        if self.time_output:
            h_freq = ofdm_to_time_channel(h_freq)
            if return_filter:
                x_win = ofdm_to_time_channel(x_win)

        h_freq = torch.movedim(h_freq, -1, axis)
        if return_filter:
            return h_freq, torch.movedim(x_win, -1, axis)
        else:
            return h_freq


def _fft_centered(x):
    """DFT along the last axis, with the zero frequency in the center"""
    return torch.fft.fftshift(torch.fft.fft(x, dim=-1), dim=-1)


def signal_power(x: torch.Tensor, axis=None, keepdims=False, average=False):
    """
    Power of a real or complex signal (or channel), computed as the squared magnitude :math:`|x|^2`
    aggregated over the given axes.

    By default, the values are *summed* (total power or energy over the axes).
    With ``average=True`` they are *averaged* instead, which gives the average power
    :math:`E[|x|^2]`. This is equal to the variance of the signal plus the squared magnitude of
    its mean, and thus coincides with the variance for zero-mean signals.

    :param x: Signal tensor
    :param axis: Axis or axes on which to aggregate. If ``None``, all axes are used
    :param keepdims: Whether to keep the aggregated dimensions
    :param average: If ``True``, return the average power (mean of :math:`|x|^2`),
        otherwise return the total power (sum of :math:`|x|^2`)
    :return: Real-valued power over the specified ``axis``
    """
    x = torch.as_tensor(x)
    aggregate = torch.mean if average else torch.sum
    return aggregate(torch.abs(x) ** 2, dim=axes_tuple(axis, x.ndim), keepdim=keepdims)
