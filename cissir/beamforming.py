import numpy as np
import matplotlib.pyplot as plt
from cissir.physics import pow2db, db2mag, mag2db

import tensorflow as tf
from tensorflow.keras.layers import Layer
from sionna.utils import expand_to_rank, flatten_last_dims, split_dim

# 6 + 6 bits
num_q_phases = 64
num_q_amps = 64
db_step = 0.5


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


def plot_beamforming_polar(bf_vectors, transmit=True, element_pattern=None,
                           r_lim=(-30, None), theta_lim=None, axis=None, **kwargs):
    if axis is None:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    else:
        ax = axis
        fig = ax.get_figure()
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
    for bf_vec in bf_vectors.T:
        bf = array_factor(bf_vec, thetas_eval, transmit=transmit)
        bf_db = pow2db(np.abs(bf)**2) + el_pat_db
        r_max_bf.append(max(bf_db))
        ax.plot(thetas_eval, bf_db, **kwargs)

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
    phases = 2 * np.pi * steer_index * np.arange(n_antennas) / (n_antennas * oversampling)
    if transmit:
        phases *= -1
    vec = np.exp(1j * phases)
    if normalize:
        vec /= np.sqrt(n_antennas)
    return vec


def dft_codebook(L_max, N1, O1=4, az_min=-60, az_max=60, transmit=True):
    worst_beam_width = np.rad2deg(2 / (N1 * np.cos(np.deg2rad(az_max))))
    beam_degs = np.linspace(az_min + worst_beam_width / 2, az_max - worst_beam_width / 2, L_max)
    beam_ang_freq = np.pi * np.sin(np.deg2rad(beam_degs))
    l_indices = np.round(beam_ang_freq * N1 * O1 / (2 * np.pi)).astype(int)
    dft_vectors = [dft_beamforming(l_i, N1, oversampling=O1, transmit=transmit) for l_i in l_indices]
    return np.stack(dft_vectors, axis=-1), beam_degs


class Beamspace(Layer):
    r"""

    Convert MIMO channel from antenna space to beam space using Tx beam codebook
    :math:`\mathbf{W}=[w_{n,b}]` of size `[num_tx_ant, num_tx_beam]` and Rx beam codebook
    :math:`\mathbf{C}=[c_{m,b'}]` of size `[num_rx_ant, num_rx_beam]`.

    The elements of the matrix tensor :math:`\mathbf{H}` are transformed as:

    :math:`h_{\ldots,b',b}=\sum_{m}\sum_{n}c_{m,b'}^*h_{\ldots,m,n}w_{n,b}`

    This class inherits from the Keras `Layer` class and can be used as layer in
    a Keras model.

    """

    def __init__(self, transmit_axis=None, receive_axis=None, rank=None, **kwargs):
        if transmit_axis is None and receive_axis is None:
            raise ValueError('At least one axis must be set')
        super().__init__(**kwargs)

        self._transmit_axis = transmit_axis
        self._receive_axis = receive_axis
        self._rank = rank

    def call(self, inputs):

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

        h_rank = int(tf.rank(h) if self._rank is None else self._rank)
        for axis, beams, mm_kwargs in ((tx_axis, tx_beams, tx_kwargs), (rx_axis, rx_beams, rx_kwargs)):
            if axis is None:
                continue
            if axis >= 0:
                axis -= h_rank
            if axis < - 2:
                flattened_dims = -(axis + 1)
                last_dims = h.shape[-flattened_dims:]
                h = flatten_last_dims(h, flattened_dims)
                beams = expand_to_rank(beams, tf.rank(h), axis=0)
                h = tf.matmul(beams, h, **mm_kwargs)
                h = split_dim(h, last_dims, axis=h_rank - flattened_dims)
            elif axis == -2:
                beams = expand_to_rank(beams, tf.rank(h), axis=0)
                h = tf.matmul(beams, h, **mm_kwargs)
            elif axis == -1:
                beams = expand_to_rank(beams, tf.rank(h), axis=0)
                h = tf.matvec(beams, h, **mm_kwargs)

        return h


def sionna_mimo_axes(channel: str):
    mimo_dict = {"cir":  (-5, -3),
                 "ofdm": (-5, -3),
                 "time": (-5, -3)}

    return mimo_dict[channel]
