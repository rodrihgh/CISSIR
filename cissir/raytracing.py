"""
Utilities for Sionna's ray tracer
author: Danial Dehghani, Rodrigo Hernangomez
"""

import numpy as np
from typing import List, Optional
import torch

from sionna import rt

from cissir.utils import base_path, res_path
from cissir.optimization import codebook_si

cir_path = res_path/"channel_impulse_responses.npz"
si_mat_path = res_path/"si_mimo.npz"

scene_fname = "scene.xml"
rt_path = base_path/"rt"
scene_path = str(rt_path/scene_fname)


def set_scattering(scattering_coefficients: dict, scene: rt.Scene) -> None:
    for rm in scene.radio_materials.values():
        if rm.name in scattering_coefficients:
            rm.scattering_coefficient = scattering_coefficients[rm.name]


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


def si_paths2cir(si_paths, axis_path=0, transpose=(2, 0, 1)):

    if axis_path is None:
        sum_paths = si_paths
    else:
        sum_paths = np.sum(si_paths, axis=axis_path)
    if transpose is None:
        result = sum_paths
    else:
        result = np.squeeze(sum_paths).transpose(*transpose)

    return result


def _to_numpy(x):
    """Convert a tensor (on any device) or array-like to a NumPy array"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def save_cir(ht_si, ht_tgt, t_channel_s, fname=None):
    fname = cir_path if fname is None else fname
    np.savez(fname, ht_si=_to_numpy(ht_si), ht_tgt=_to_numpy(ht_tgt), t_channel_s=_to_numpy(t_channel_s))


def save_si_matrix(ht_si, t_channel_s, fname=None):
    """
    Saves the MIMO taps of the SI channels.
    :param ht_si: CIR of each SI path
    :param t_channel_s: time support of ``ht_si`` in seconds
    :param fname: Filename to save the matrix
    :return: mean path delays in seconds
    """
    fname = si_mat_path if fname is None else fname
    ht_squeeze = torch.squeeze(torch.as_tensor(ht_si))
    ht_si_abs = torch.abs(ht_squeeze)
    t_channel_s = np.asarray(t_channel_s)
    t_tensor = torch.as_tensor(t_channel_s, dtype=ht_si_abs.dtype, device=ht_si_abs.device)
    mean_t = torch.sum(ht_si_abs * t_tensor, dim=-1) / torch.sum(ht_si_abs, dim=-1)
    t_si_matrix = torch.mean(mean_t, dim=(1, 2), keepdim=True)
    si_indices = [np.argmin(np.abs(t_channel_s - t))
                  for t in t_si_matrix.reshape(-1).cpu().numpy()]
    h_si_matrix = torch.stack([ht_squeeze[n, ..., i] for n, i in enumerate(si_indices)], dim=0)
    np.savez(fname, h_si_matrix=h_si_matrix.cpu().numpy(), mean_delay=t_si_matrix.cpu().numpy())

    return t_si_matrix


def load_data(*args, fname=cir_path, torch_type=None, **kwargs):
    """
    Load arrays from a NumPy ``npz`` file
    :param args: Names of the arrays to load
    :param fname: Path of the ``npz`` file
    :param torch_type: If given, convert the arrays to PyTorch tensors of this dtype
    :param kwargs: Keyword arguments passed to ``torch.as_tensor``, e.g. ``device``
    :return: A single array if one name is given, otherwise a list of arrays
    """
    with np.load(fname) as rt_data:
        data = [rt_data[arg] for arg in args]
    if torch_type is not None:
        data = [torch.as_tensor(d, dtype=torch_type, **kwargs) for d in data]
    if len(data) == 1:
        return data[0]
    elif len(data) > 1:
        return data
    else:
        raise ValueError("At least one positional argument should be given")


def load_cir(*args, fname=cir_path, torch_type=torch.complex64, **kwargs):
    return load_data(*args, fname=fname, torch_type=torch_type, **kwargs)


def load_si_paths(num_taps, fname=None):
    if num_taps == "full":
        fname = cir_path if fname is None else fname
        h_si, t_rt = load_data("ht_si", "t_channel_s", fname=fname)
    elif isinstance(num_taps, int):
        fname = si_mat_path if fname is None else fname
        h_si, t_rt = [data[:num_taps] for data in
                      load_data("h_si_matrix", "mean_delay", fname=fname)]
    else:
        raise ValueError("num_taps must be either 'full' or an integer")

    return h_si, t_rt


def normalize_si_taps(h_si_taps, h_si_full, tx_codebook, rx_codebook, num_taps=None):
    if num_taps is None:
        num_taps = h_si_taps.shape[0]
    h_si_cir = si_paths2cir(h_si_full[:num_taps])
    si_taps_mag = codebook_si(tx_codebook, rx_codebook, h_si_taps).max()
    si_ref_mag = codebook_si(tx_codebook, rx_codebook, h_si_cir).max()

    return h_si_taps * (si_ref_mag / si_taps_mag), si_ref_mag
