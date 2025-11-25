"""
Signal Processing Module
"""

import numpy as np
from numpy import fft as npft
from scipy.signal.windows import get_window


class MatchedFilter:
    """
    Provides functionality for matched filtering in RF systems.

    The MatchedFilter class is used to perform matched filtering operations.
    This implementation includes options for applying a specified window
    function, operating in the time domain or frequency domain, and returning
    the applied filter along with the filtered signal.

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
        signal: The input signal to be filtered.
        reference: The reference signal used for filtering.
        axis: int, optional
            The axis along which filtering is applied. Default is -1.
        return_filter: bool, optional
            If True, the applied filter is returned alongside the filtered signal. Default is False.

        Returns:
            The filtered signal. If `return_filter` is True, returns a tuple containing
            the filtered signal and the filter used.
        """

        if self.time_input:
            signal = fft(signal, axis=axis)
            reference = fft(reference, axis=axis)

        # Frequency-domain matched filter
        x_win = np.conj(reference) * self._window
        h_freq = signal * x_win

        if self.time_output:
            h_freq = ifft(h_freq, axis=axis)
            if return_filter:
                x_win = ifft(x_win, axis=axis)

        if return_filter:
            return h_freq, x_win
        else:
            return h_freq


def fft(x, n=None, axis=-1, **kwargs):
    return npft.fftshift(npft.fft(x, n, axis, **kwargs), axes=axis)


def ifft(x, n=None, axis=-1, **kwargs):
    return npft.ifft(npft.ifftshift(x, axes=axis), n, axis, **kwargs)
