"""
Functions for ADC quantization noise
author: Danial Dehghani, Rodrigo Hernangomez
"""

import pandas as pd
import tensorflow as tf
from cissir.physics import db2mag


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
    real_part = tf.math.real(complex_input)
    imaginary_part = tf.math.imag(complex_input)
    return tf.maximum(tf.reduce_max(real_part), tf.reduce_max(imaginary_part))


def max_abs_complex(complex_input, axis=None, keepdims=False):
    return tf.reduce_max(tf.abs(complex_input), axis=axis, keepdims=keepdims)


def quantize_signal(signal: tf.Tensor, quantization_bits: int, max_value: float = None) -> tf.Tensor:
    """
    Quantizes a complex signal tensor into discrete levels.

    Arguments:
    - signal: TensorFlow Tensor of complex numbers.
    - max_value: The maximum value the real or imaginary part of the signal can take.
      If None, use the input signal to determine it
    - quantization_bits: The number of quantization bits for every signal component (I and Q).

    Returns:
    - Quantized TensorFlow Tensor of complex numbers.

    This function separately quantizes the real and imaginary parts of the input complex signal and combines them back
    into a complex tensor.
    """
    real_part = tf.math.real(signal)
    imaginary_part = tf.math.imag(signal)

    if max_value is None:
        max_value = complex_max(signal)

    quantization_level = 2 ** quantization_bits

    def quantize(values):
        delta = (2 * max_value) / quantization_level
        scaled_values = tf.math.floor(values / delta) + 0.5
        quantized_values = scaled_values * delta
        return quantized_values

    quantized_real = quantize(real_part)
    quantized_imaginary = quantize(imaginary_part)
    quantized_signal = tf.complex(quantized_real, quantized_imaginary)

    return quantized_signal


def papr(sig, axis=None, keepdims=False,):
    avg_pow = tf.math.reduce_variance(sig, axis=axis, keepdims=keepdims)
    max_pow = tf.reduce_max(tf.abs(sig), axis=axis, keepdims=keepdims)**2
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
