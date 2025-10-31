from sionna.phy import constants

# CONSTANTS
c = constants.SPEED_OF_LIGHT
k_boltzmann = 1.380649e-23

T_noise_K = 300.0
n0 = k_boltzmann * T_noise_K * 1e3  # mW/Hz

subcarriers_per_rb = 12 

# RF parameters
fc_hz = 28e9
wavelength_m = c / fc_hz
array_electrical_spacing = 0.5  # Electrical spacing (in lambdas) between array elements

n_rx = 8
n_tx = 8
num_beams = 8

antenna_pattern = "tr38901"
antenna_polarization = "V"

# Waveform parameters

subcarrier_spacing_khz = 120
num_prb = 140  # Around 200 MHz bandwidth, frac. bandwidth < 10%
