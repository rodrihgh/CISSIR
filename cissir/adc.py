"""
Functions for ADC quantization noise
author: Danial Dehghani, Rodrigo Hernangomez
"""

import numpy as np
import pandas as pd
import torch
from cissir.physics import db2mag, db2power, c0
from cissir.utils import axes_tuple


def abs_complex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the absolute values of complex numbers in each column of a DataFrame.

    Arguments:
    - df: DataFrame where each cell is a string representing a complex number.

    Returns:
    - DataFrame with the same structure as input, where each cell is the absolute value of the original complex number.

    This function iterates through each column and applies the absolute value function to each element after converting
    it to a complex number.
    """
    abs_df = pd.DataFrame()

    for col in df.columns:
        abs_values = df[col].apply(lambda x: abs(complex(x)))
        abs_df[col] = abs_values

    return abs_df


# %%

def complex_max(complex_input):
    complex_input = torch.as_tensor(complex_input)
    return torch.maximum(complex_input.real.max(), complex_input.imag.max())


def max_abs_complex(complex_input, axis=None, keepdims=False):
    complex_input = torch.as_tensor(complex_input)
    return torch.amax(torch.abs(complex_input), dim=axes_tuple(axis, complex_input.ndim), keepdim=keepdims)


def quantize_signal(signal: torch.Tensor, quantization_bits: int, max_value: float = None) -> torch.Tensor:
    """
    Quantizes a complex signal tensor into discrete levels.

    Arguments:
    - signal: PyTorch Tensor of complex numbers.
    - max_value: The maximum value the real or imaginary part of the signal can take.
      If None, use the input signal to determine it
    - quantization_bits: The number of quantization bits for every signal component (I and Q).

    Returns:
    - Quantized PyTorch Tensor of complex numbers.

    This function separately quantizes the real and imaginary parts of the input complex signal and combines them back
    into a complex tensor.
    """
    signal = torch.as_tensor(signal)
    real_part = signal.real
    imaginary_part = signal.imag

    if max_value is None:
        max_value = complex_max(signal)

    quantization_level = 2 ** quantization_bits

    def quantize(values):
        delta = (2 * max_value) / quantization_level
        scaled_values = torch.floor(values / delta) + 0.5
        quantized_values = scaled_values * delta
        return quantized_values

    quantized_real = quantize(real_part)
    quantized_imaginary = quantize(imaginary_part)
    quantized_signal = torch.complex(quantized_real, quantized_imaginary)

    return quantized_signal


def papr(sig, axis=None, keepdims=False,):
    sig = torch.as_tensor(sig)
    dims = axes_tuple(axis, sig.ndim)
    # Population variance (correction=0), as in TensorFlow's ``reduce_variance``
    avg_pow = torch.var(sig, dim=dims, correction=0, keepdim=keepdims)
    max_pow = torch.amax(torch.abs(sig), dim=dims, keepdim=keepdims)**2
    return max_pow/avg_pow


def snr_thermal(p_t, channel_gain, noise_power):
    return p_t * channel_gain / noise_power


def sqnr_bound(si2target_db, target1norm, target2norm, num_bits,
               signal_papr, target_power_ratio, bandwidth, snr=None):
    adc_term = (3/2) * 2 ** (2*num_bits)
    signal_term = target_power_ratio/(signal_papr * bandwidth)
    channel_term = ((target2norm/target1norm)/(db2mag(si2target_db)+1)) ** 2
    sqnr = adc_term * signal_term * channel_term
    if snr is not None:
        sqnr = 1/(1/sqnr + 1/snr)
    return sqnr


def crlb_range(snr_db, bandwidth_hz):
    """Compute the Cramer-Rao Lower Bound for range estimation as per
    Chapter 18.4.3 in (Richards et al., 2010).
    :param snr_db: Signal-to-Noise Ratio in dB
    :param bandwidth_hz: Bandwidth in Hz
    :return: CRLB in square meters
    """
    snr = db2power(snr_db)
    bw_rms = bandwidth_hz * np.pi/np.sqrt(3)
    return ((c0/bw_rms)**2)/(4*snr)
