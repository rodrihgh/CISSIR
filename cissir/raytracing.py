"""
Utilities for Sionna's ray tracer
author: Danial Dehghani, Rodrigo Hernangomez
"""

import numpy as np
from typing import List, Optional, Dict
import tensorflow as tf

from sionna import rt

from pathlib import Path

scene_fname = "scene.xml"
rt_path = Path(__file__).parents[1].absolute()/"rt"
scene_path = str(rt_path/scene_fname)

_paths_no_dim = ['min_tau', 'normalize_delays', 'reverse_direction', 'sources', 'targets']
_paths_prelast_dim = ['a', 'vertices']
_paths_last_dim = ['doppler', 'mask', 'objects', 'phi_r', 'phi_t',
                   'targets_sources_mask', 'tau', 'theta_r', 'theta_t', 'types']


def set_scattering(scattering_coefficients: dict, scene: rt.Scene) -> None:
    for rm in scene.radio_materials.values():
        if rm.name in scattering_coefficients:
            rm.scattering_coefficient = scattering_coefficients[rm.name]


def remove_zero_paths(path_dict):
    # Remove null paths
    paths_a = path_dict['a']
    path_mask = tf.reduce_any(tf.abs(paths_a) > 0.0,
                              axis=[n for n in range(paths_a.ndim)
                                    if n != paths_a.ndim - 2])

    new_dict = {}
    for k, v in path_dict.items():
        if k in _paths_no_dim:
            new_dict[k] = v   
        elif k in _paths_prelast_dim:
            new_dict[k] = tf.boolean_mask(v, path_mask, axis=v.ndim-2)    
        elif k in _paths_last_dim:
            new_dict[k] = tf.boolean_mask(v, path_mask, axis=v.ndim-1)  
        else:
            raise KeyError(f"Unexpected path key '{k}'")

    return new_dict


def concatenate_paths(path_dicts: List[Dict], zero_paths=False):
    if not zero_paths:
        path_dicts = [remove_zero_paths(p) for p in path_dicts]
    out_dict = {'tau_min': min(p['min_tau'] for p in path_dicts)}
    p0 = path_dicts[0]
    eq_path_opts = [k for k in _paths_no_dim if k != 'min_tau']
    for k in eq_path_opts:
        v = p0[k]
        assert all(tf.reduce_all(tf.equal(v, p[k]))
                   for p in path_dicts), f"Incompatible paths, unmatched parameter '{k}'"
        out_dict[k] = v
    for k in _paths_prelast_dim:
        out_dict[k] = tf.concat([p[k] for p in path_dicts], axis=p0[k].ndim-2)
    for k in _paths_last_dim:
        out_dict[k] = tf.concat([p[k] for p in path_dicts], axis=p0[k].ndim-1)
    
    return out_dict       


def rearrange_paths(h, tau):
    # Group non-negative delays under same paths
    t_shape = tau.shape
    h_shape = h.shape
    n_tsteps = h_shape[-1]
    num_paths = t_shape[-1]
    tau_tab = tf.reshape(tau, shape=(-1, num_paths))
    h_tab = tf.reshape(h, shape=(-1, num_paths, n_tsteps))

    t_collector = []
    h_collector = []
    for i in range(tau_tab.shape[0]):
        tau_row = tau_tab[i, :]
        h_row = h_tab[i, :, :]
        bool_mask = tf.greater_equal(tau_row, 0.0)
        t_collector.append(tf.concat([tf.boolean_mask(tau_row, bool_mask),
                                      tf.boolean_mask(tau_row, tf.logical_not(bool_mask))],
                                     axis=0))
        bool_mask_h = tf.stack([bool_mask for _ in range(n_tsteps)], axis=-1)
        h_collector.append(tf.concat([tf.boolean_mask(h_row, bool_mask_h),
                                      tf.boolean_mask(h_row, tf.logical_not(bool_mask_h))],
                                     axis=0))
    tau_new = tf.reshape(tf.stack(t_collector, axis=0), shape=t_shape)
    h_new = tf.reshape(tf.stack(h_collector, axis=0), shape=h_shape)
    return h_new, tau_new


def cylindrical_to_cartesian(coordinates: List[float]) -> Optional[np.ndarray]:
    """
    Converts cylindrical coordinates to Cartesian coordinates.

    Arguments:
    - coordinates: List of three floats [r, theta, z] representing cylindrical coordinates.

    Returns:
    - NumPy array representing Cartesian coordinates [x, y, z].

    This function performs the following conversions:
    1. Calculates the x-coordinate using r*cos(theta).
    2. Calculates the y-coordinate using r*sin(theta).
    3. Preserves the z-coordinate as is.
    """
    r, theta, z = coordinates

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return np.array([x, y, z])


def tx_rx_positions(N_tx, N_rx, wavelength_m, min_dist_el, transceiver_position=(0, 0, .94)):

    tx_rx_dist_el = (N_tx - 1) / 2 + (N_rx - 1) / 2 + min_dist_el
    tx_position = list(transceiver_position)
    rx_position = list(transceiver_position)
    tx_position[1] += (tx_rx_dist_el * wavelength_m)/2
    rx_position[1] -= (tx_rx_dist_el * wavelength_m)/2

    return tx_position, rx_position
