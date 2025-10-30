import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cissir.physics import pow2db, db2mag, mag2db

import tensorflow as tf
from tensorflow.keras.ops import numpy as tnp

from sionna import Block
from sionna.phy.utils import expand_to_rank, flatten_last_dims, split_dim

rng = np.random.default_rng()

# 6 + 6 bits
num_q_phases = 64
num_q_amps = 64
db_step = 0.5

beam_oversampling = 4


def steer_vec(n_elements, thetas_rad, electric_length=0.5, centered=False):
    """
    Steering vector of uniform linear array (ULA) for one or more angles
    :param n_elements: Number of elements in the ULA
    :param thetas_rad: Incidence angle(s) in radians. It can be a singleton or an array
    :param electric_length: physical length over wavelength. Defaults to lambda half
    :param centered: If ``True``, the reference element is at the center of the ULA
    :return: Steering vector/matrix with shape (n_elements, len(thetas_rad))
    """

    phase_delta = electric_length * 2 * np.pi * np.sin(thetas_rad)
    el_array = np.arange(n_elements, dtype=float)
    if centered:
        el_array -= (n_elements-1)/2
    phase = el_array[:, np.newaxis] * np.expand_dims(phase_delta, axis=0)
    return np.exp(1j * phase)


def quantize_codebook(beam_codebook, max_amp=None, num_phases=num_q_phases, num_amps=num_q_amps):

    cb_amps = np.abs(beam_codebook)
    cb_phases = np.angle(beam_codebook, deg=False)

    if max_amp is None:
        max_amp = np.max(cb_amps)

    phase_step = 2 * np.pi / num_phases
    min_db, max_db = -num_amps * db_step, 0.0

    cb_amps_norm = cb_amps / max_amp
    cb_amps_db = mag2db(cb_amps_norm)
    cb_amps_db_q = np.round(np.clip(cb_amps_db, min_db, max_db)/db_step)*db_step
    cb_amps_q = max_amp * db2mag(cb_amps_db_q)

    cb_phases_q = np.round(cb_phases/phase_step)*phase_step

    return cb_amps_q * np.exp(1j*cb_phases_q)


def array_factor(antenna_weights, thetas_radian, n_antennas=None, transmit=True):
    if n_antennas is None:
        n_antennas = len(antenna_weights)

    d_sign = 1 if transmit else -1
    af = np.array([steer_vec(n_antennas, d_sign * d).reshape(1, n_antennas) @ antenna_weights
                   for d in thetas_radian]).squeeze()
    return af


def plot_beamforming_polar(bf_vectors, transmit=True, element_pattern=None, color_palette=None,
                           r_lim=(-30, None), theta_lim=None, axis=None, aggregate=False, fill=False,  **kwargs):
    if axis is None:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    else:
        ax = axis
        fig = ax.get_figure()
    plot_f = ax.plot if not fill else ax.fill
    ax.set_theta_zero_location("N")

    r_min, r_max = r_lim

    if theta_lim is None:
        theta_min, theta_max = -np.pi, np.pi
    else:
        theta_min, theta_max = theta_lim
        ax.set_thetalim(theta_min, theta_max)
    ax.set_xlabel("Antenna gain [dBi]")
    thetas_eval = np.linspace(theta_min, theta_max, 1000)

    if element_pattern is None:
        el_pat_db = np.zeros_like(thetas_eval)
    else:
        el_pat_db = pow2db([element_pattern(d) for d in thetas_eval])

    r_max_bf = []
    agg = []
    colors = None if color_palette is None else sns.color_palette(color_palette, n_colors=bf_vectors.shape[-1])
    for n, bf_vec in enumerate(bf_vectors.T):
        if colors is not None:
            kwargs["c"] = colors[n]
        bf = array_factor(bf_vec, thetas_eval, transmit=transmit)
        bf_db = pow2db(np.abs(bf)**2) + el_pat_db
        r_max_bf.append(max(bf_db))
        if aggregate:
            agg.append(bf_db)
        else:
            plot_f(thetas_eval, bf_db, **kwargs)
    if aggregate:
        bf_db = np.max(np.stack(agg, axis=-1), axis=-1)
        plot_f(thetas_eval, bf_db, **kwargs)

    if r_max is None:
        r_max = np.max(r_max_bf) + 1
        r_min += r_max

    ax.set_rlim(r_min, r_max)
    if theta_lim is None:
        ax.set_rlabel_position(-120)
    if axis is None:
        plt.show()
    return fig, ax


def dft_beamforming(steer_index, n_antennas, oversampling=4, transmit=True, normalize=True):
    phases = (2 * np.pi * np.reshape(steer_index, (1, -1)) *
              np.arange(n_antennas)[:, None] / (n_antennas * oversampling))
    if transmit:
        phases *= -1
    vec = np.exp(1j * phases)
    if normalize:
        vec /= np.sqrt(n_antennas)
    return vec


def dft_codebook(L_max: int, N1: int, O1=beam_oversampling, az_min=-60.0, az_max=60.0,
                 transmit=True, full_grid=False, random_angles=False):
    """
    Generate codebook according to 3GPP DFT beamforming
    :param L_max: Number of codewords (matrix columns). Ignored if ``full_grid=True``
    :param N1: Number of antennas
    :param O1: Oversampling factor
    :param az_min: Minimum azimuth angle to cover
    :param az_max: Maximum azimuth angle to cover
    :param transmit: if ``True``, a TX codebook will be computed, otherwise a RX codebook will be computed
    :param full_grid: If ``True``, span the full DFT grid between ``az_min`` and ``az_max``
    :param random_angles: if ``True``, the codebook angles will be randomly sampled,
    otherwise they will be linearly sampled. Ignored if ``full_grid=True``
    :return: The codebook matrix with shape (N1, L_max), together with the beam angles in degrees
    """
    beam_degs: np.ndarray
    if random_angles and not full_grid:
        beam_degs = rng.uniform(low=az_min, high=az_max, size=L_max)
    else:
        worst_beam_width = np.rad2deg(2 / N1)  # (N1 * np.cos(np.deg2rad(az_max))))
        beam_degs = np.linspace(az_min + worst_beam_width / 2, az_max - worst_beam_width / 2, L_max)
    beam_ang_freq = np.pi * np.sin(np.deg2rad(beam_degs))
    l_indices = np.round(beam_ang_freq * N1 * O1 / (2 * np.pi)).astype(int)
    if full_grid:
        l_indices = np.arange(l_indices[0], l_indices[-1] + 1)
        beam_degs = np.rad2deg(np.arcsin(2 * l_indices / (N1 * O1)))
    dft_matrix: np.ndarray = dft_beamforming(l_indices, N1, oversampling=O1, transmit=transmit)
    return dft_matrix, beam_degs


def channel_power(h_channel: tf.Tensor, axis=None, keepdims=False, name=None):
    """
    Compute the total power of a wireless channel
    :param h_channel: Channel tensor
    :param axis: Axes on which to compute the power
    :param keepdims: Whether to keep the power dimensions
    :param name: Tensorflow variable name
    :return: Channel power over specified ``axis``
    """
    return tf.reduce_sum(tf.math.pow(tf.abs(h_channel), 2),
                         axis=axis, keepdims=keepdims, name=name)


class BeamSelection(Block):
    """
    Select ``num_beams`` beams with the strongest channel gain for hybrid/analog transmission.
    If ``num_rx>1``, then the strongest ``num_beams//num_rx`` beams from each user will be selected.
    ``num_beams`` must be a multiple of ``num_rx``.

    - Input: ``h_channel``
    - Output ``h_beams``, with ``num_beams`` in the ``beam_axis`` (default axis: -3)

    This class inherits from the Keras `Layer` class and can be used as layer in
    a Keras model.
    """

    _channel_ndims = 7

    def __init__(self, num_beams, oversampling=beam_oversampling, orthogonal=True,
                 normalize=True, beam_axis=None, power_axes=None, **kwargs):

        super().__init__(**kwargs)

        self._num_beams = num_beams
        self._o1 = oversampling
        self._normalize = normalize

        self._power_axes = _power_axes if power_axes is None else power_axes

        self._beam_axis = sionna_mimo_axes("ofdm")[-1] if beam_axis is None else beam_axis

        self._orthogonal = orthogonal and num_beams > 1

        self.channel_shape = None
        self.codebook_size = None

    def build(self, *args):
        channel_shape = tf.unstack(args[0])
        self.codebook_size = channel_shape[self._beam_axis]

        channel_shape[self._beam_axis] = channel_shape[-1]
        channel_shape[-1] = self._num_beams
        self.channel_shape = tf.stack(channel_shape)

    def _simple_selection(self, beam_power: tf.Tensor):
        return tf.math.top_k(beam_power, k=self._num_beams).indices

    def _ortho_selection(self, beam_power: tf.Tensor):
        """
        Orthogonal beam selection based on Algorithm 1 in
        'A Tutorial on Downlink Precoder Selection Strategies for 3GPP MIMO Codebooks' (Fu et al., 2023)
        :param beam_power: Power for each beam, aggregated over users and resources
        :return: orthogonal indices with max power
        """

        cb_size = self.codebook_size
        pad_size = self._o1 * tf.cast(tf.math.ceil(cb_size / self._o1), dtype=cb_size.dtype) - cb_size
        paddings = tf.concat([tf.zeros([beam_power.ndim-1, 2], dtype=pad_size.dtype),
                              tf.reshape([0, pad_size], [1, 2])], axis=0)
        beam_pow_pad = tf.pad(beam_power, paddings, constant_values=-1)

        beam_max = tf.math.top_k(beam_power, k=1)
        beam_q = tf.math.mod(beam_max.indices, self._o1)
        ortho_beams = tf.gather(beam_pow_pad, beam_q + tf.range(cb_size, delta=self._o1), batch_dims=-1)
        return self._o1 * tf.math.top_k(ortho_beams, k=self._num_beams).indices + beam_q

    def call(self, h_channel: tf.Tensor):

        h_power = channel_power(h_channel, axis=self._power_axes, keepdims=True)

        h_channel = tnp.swapaxes(h_channel, self._beam_axis, -1)
        h_power = tnp.swapaxes(h_power, self._beam_axis, -1)

        beam_indices = self._ortho_selection(h_power) if self._orthogonal else self._simple_selection(h_power)
        output = tf.gather(h_channel, tf.broadcast_to(beam_indices, self.channel_shape), batch_dims=-1)
        output = tnp.swapaxes(output, self._beam_axis, -1)

        if self._normalize:
            output /= tf.sqrt(tf.cast(self._num_beams, dtype=output.dtype))

        return output


class Beamspace(Block):
    r"""

    Convert MIMO channel from antenna space to beam space using Tx beam codebook
    :math:`\mathbf{W}=[w_{n,b}]` of size `[num_tx_ant, num_tx_beam]` and Rx beam codebook
    :math:`\mathbf{C}=[c_{m,b'}]` of size `[num_rx_ant, num_rx_beam]`.

    The elements of the matrix tensor :math:`\mathbf{H}` are transformed as:

    :math:`h_{\ldots,b',b}=\sum_{m}\sum_{n}c_{m,b'}^*h_{\ldots,m,n}w_{n,b}`

    This class inherits from the Keras `Layer` class and can be used as layer in
    a Keras model.

    """

    def __init__(self, transmit_axis=None, receive_axis=None, **kwargs):
        if transmit_axis is None and receive_axis is None:
            raise ValueError('At least one axis must be set')
        super().__init__(**kwargs)

        self._transmit_axis = transmit_axis
        self._receive_axis = receive_axis
        self._rank = self.h_shape = None

    def build(self, *args):
        self.h_shape = args[0]
        self._rank = len(self.h_shape)

    def call(self, *inputs):

        tx_axis = self._transmit_axis
        rx_axis = self._receive_axis
        h = tx_beams = rx_beams = None
        tx_kwargs = dict(transpose_a=True)
        rx_kwargs = dict(adjoint_a=True)

        if tx_axis is not None and rx_axis is not None:
            h, rx_beams, tx_beams = inputs
        elif tx_axis is not None:
            h, tx_beams = inputs
        elif rx_axis is not None:
            h, rx_beams = inputs

        h_rank = self._rank
        for axis, beams, mm_kwargs in ((tx_axis, tx_beams, tx_kwargs), (rx_axis, rx_beams, rx_kwargs)):
            if axis is None:
                continue
            if axis >= 0:
                axis -= h_rank
            if axis < - 2:
                flattened_dims = -(axis + 1)
                last_dims = self.h_shape[-flattened_dims:]
                h = flatten_last_dims(h, flattened_dims)
                beams = expand_to_rank(beams, h_rank, axis=0)
                h = tf.matmul(beams, h, **mm_kwargs)
                h = split_dim(h, last_dims, axis=h_rank - flattened_dims)
            elif axis == -2:
                beams = expand_to_rank(beams, h_rank, axis=0)
                h = tf.matmul(beams, h, **mm_kwargs)
            elif axis == -1:
                beams = expand_to_rank(beams, h_rank, axis=0)
                h = tf.linalg.matvec(beams, h, **mm_kwargs)

        return h


def sionna_mimo_axes(channel: str):
    mimo_dict = {"cir":  (-5, -3),
                 "ofdm": (-5, -3),
                 "time": (-5, -3)}

    return mimo_dict[channel]


# full channel dimensions: [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
# [num_rx, num_rx_ant, num_ofdm_symbols, fft_size]
_power_axes = (1, 2, 5, 6)
