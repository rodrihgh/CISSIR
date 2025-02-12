"""
Utilities for physical units and radar physics
"""

import numpy as np
from scipy.stats import hmean
from cissir import params


c0 = params.c
"""
Speed of light in m/s
"""


def db2power(db_value):
    return 10 ** (db_value/10)


def db2mag(db_value):
    return 10 ** (db_value/20)


def mag2db(mag_value):
    return 20 * np.log10(mag_value)


def pow2db(pow_value):
    return 10 * np.log10(pow_value)


def avg_snr_noise(snr_db, axis=None):
    """
    Average noise for SNR values in dB with fixed signal level.
    It computes a linear harmonic mean of SNR values.
    :param axis: The axis along which to apply the harmonic mean.
    :param snr_db: SNR values in dB
    :return: SNR for average noise in dB
    """
    return pow2db(hmean(db2power(snr_db), axis=axis))


def delay2distance(delay_s, two_way=True):
    dist_m = delay_s * c0
    if two_way:
        dist_m /= 2
    return dist_m


def distance2delay(distance_m, two_way=True):
    delay_s = distance_m/c0
    if two_way:
        delay_s *= 2
    return delay_s


def radar_rng_eq(rcs_sigma, wavelength_m, tx_distance_m, rx_distance_m=None):
    """
    Radar range equation
    :param rcs_sigma: scatterer's radar cross section, in meters
    :param wavelength_m: Propagation wavelength in meters
    :param tx_distance_m: Distance from radar transmitter to scatterer in meters
    :param rx_distance_m: Distance from radar receiver to scatterer in meters.
    If not given, monostatic radar is inferred.
    :return: Magnitude attenuation according to the range radar equation
    """

    s = rcs_sigma
    wl = wavelength_m
    r1 = tx_distance_m
    r2 = r1 if rx_distance_m is None else rx_distance_m

    power_att = ((wl ** 2) * s)/(((4 * np.pi) ** 3) * (r1 ** 2) * (r2 ** 2))

    return np.sqrt(power_att)


def estimate_rcs(power_profile, distance, wavelength, g_tx, g_rx):

    assert power_profile.shape == distance.shape, "Distance and power profile must have the same shape"
    
    rcs = ((4*np.pi) ** 3)/(g_tx * g_rx * (wavelength ** 2)) * np.sum((distance ** 4) * power_profile)
    return rcs
