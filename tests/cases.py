"""
Deterministic, framework-free test inputs shared by the TensorFlow reference generator
(``gen_tf_reference.py``) and the PyTorch parity tests (``test_torch_parity.py``).

Only NumPy is used here, so that exactly the same inputs are produced in both environments.
"""

import numpy as np

from cissir.beamforming import dft_codebook

SEED = 20251005


def rng(offset=0):
    return np.random.default_rng(SEED + offset)


def rand_complex(shape, seed, dtype=np.complex64):
    g = rng(seed)
    return (g.standard_normal(shape) + 1j * g.standard_normal(shape)).astype(dtype)


def tied_complex(shape, seed, dtype=np.complex64):
    """Complex values with only a few distinct magnitudes, so that many powers are exactly tied."""
    g = rng(seed)
    mag = g.choice([0.0, 1.0, 2.0], size=shape)
    phase = g.choice([1, -1, 1j, -1j], size=shape)
    return (mag * phase).astype(dtype)


# full channel dimensions: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
CH_SHAPE = (3, 2, 2, 2, 5, 3)  # without the beam (tx_ant) dimension, which is inserted at axis 4


def dup_complex(shape, seed, beam_dim=4, dtype=np.complex64):
    """Beams come in pairs (2m, 2m+1) with exactly equal power but different phase.
    A different tie-breaking rule will therefore select different beams and show in the output."""
    shape = list(shape)
    n_beams = shape[beam_dim]
    shape[beam_dim] = (n_beams + 1) // 2
    base = rand_complex(tuple(shape), seed, dtype=dtype)
    pair = np.stack([base, base * dtype(1j)], axis=beam_dim + 1)
    new_shape = list(shape)
    new_shape[beam_dim] = 2 * shape[beam_dim]
    pair = pair.reshape(new_shape)
    return np.take(pair, np.arange(n_beams), axis=beam_dim)


def channel_shape(num_beams_cb):
    b, rx, rxa, tx, sym, fft = CH_SHAPE
    return (b, rx, rxa, tx, num_beams_cb, sym, fft)


# ---------------------------------------------------------------------------------------------
# channel_power
# ---------------------------------------------------------------------------------------------

CHANNEL_POWER_KWARGS = {
    "all": dict(axis=None, keepdims=False),
    "power_axes": dict(axis=(1, 2, 5, 6), keepdims=True),
    "last": dict(axis=-1, keepdims=False),
    "tuple_nokeep": dict(axis=(0, 3), keepdims=False),
}


def channel_power_input():
    return rand_complex(channel_shape(8), seed=1)


# ---------------------------------------------------------------------------------------------
# BeamSelection
# ---------------------------------------------------------------------------------------------

# name -> (input generator args, constructor kwargs)
# codebook size 30 is not a multiple of the oversampling factor 4 -> exercises the padding branch
BEAM_SELECTION_CASES = {}
_case_id = 0
for _L in (32, 30, 17):
    for _gen in ("rand", "tied", "dup"):
        for _ortho in (True, False):
            for _k in (1, 2, 4):
                for _norm in (True, False):
                    _case_id += 1
                    name = f"L{_L}_{_gen}_{'ortho' if _ortho else 'simple'}_k{_k}_{'norm' if _norm else 'raw'}"
                    BEAM_SELECTION_CASES[name] = dict(
                        L=_L, gen=_gen, seed=100 + _case_id,
                        kwargs=dict(num_beams=_k, orthogonal=_ortho, normalize=_norm, oversampling=4))

# Other oversampling factors and non-default axes
BEAM_SELECTION_CASES["L24_rand_ortho_o1_3_k3"] = dict(
    L=24, gen="rand", seed=900, kwargs=dict(num_beams=3, orthogonal=True, oversampling=3))
BEAM_SELECTION_CASES["L16_rand_ortho_o1_2_k4"] = dict(
    L=16, gen="rand", seed=901, kwargs=dict(num_beams=4, orthogonal=True, oversampling=2))
# Beams on the rx antenna axis (-5), power aggregated over everything else but batch
BEAM_SELECTION_CASES["rxaxis_L16_simple_k2"] = dict(
    L=16, gen="rand", seed=902, beam_dim=2,
    kwargs=dict(num_beams=2, orthogonal=False, beam_axis=-5, power_axes=(1, 3, 4, 5, 6)))
BEAM_SELECTION_CASES["rxaxis_L16_ortho_k2"] = dict(
    L=16, gen="rand", seed=903, beam_dim=2,
    kwargs=dict(num_beams=2, orthogonal=True, beam_axis=-5, power_axes=(1, 3, 4, 5, 6)))


def beam_selection_input(case):
    shape = list(channel_shape(case["L"]))
    if case.get("beam_dim", 4) != 4:  # move the beam dimension from tx_ant to another axis
        shape[4], shape[case["beam_dim"]] = shape[case["beam_dim"]], shape[4]
    if case["gen"] == "dup":
        return dup_complex(tuple(shape), seed=case["seed"], beam_dim=case.get("beam_dim", 4))
    gen = rand_complex if case["gen"] == "rand" else tied_complex
    return gen(tuple(shape), seed=case["seed"])


# ---------------------------------------------------------------------------------------------
# Beamspace
# ---------------------------------------------------------------------------------------------

def codebook(n_ant, n_beams, transmit, seed=0):
    """DFT codebook of shape (n_ant, n_beams), complex64"""
    cb, _ = dft_codebook(n_beams, n_ant, transmit=transmit)
    return cb.astype(np.complex64)


def beamspace_cases():
    """Random-channel cases. Each entry: dict(h=..., rx=..., tx=..., rx_axis=..., tx_axis=...)"""
    cases = {}
    h = rand_complex((3, 2, 4, 2, 6, 3, 5), seed=300)
    rx_cb = rand_complex((4, 7), seed=301)
    tx_cb = rand_complex((6, 9), seed=302)
    cases["both_matrix"] = dict(h=h, rx=rx_cb, tx=tx_cb, rx_axis=-5, tx_axis=-3)
    cases["tx_only"] = dict(h=h, rx=None, tx=tx_cb, rx_axis=None, tx_axis=-3)
    cases["rx_only"] = dict(h=h, rx=rx_cb, tx=None, rx_axis=-5, tx_axis=None)
    cases["both_vector"] = dict(h=h, rx=rx_cb[:, :1], tx=tx_cb[:, :1], rx_axis=-5, tx_axis=-3)
    # positive axes, as in the `_power_axes` conventions
    cases["both_posaxes"] = dict(h=h, rx=rx_cb, tx=tx_cb, rx_axis=2, tx_axis=4)
    cases["lastaxis_tx"] = dict(h=h, rx=None, tx=rand_complex((5, 4), seed=303), rx_axis=None, tx_axis=-1)
    # Double precision, to assess bit-level agreement
    cases["both_matrix_c128"] = dict(h=h.astype(np.complex128), rx=rx_cb.astype(np.complex128),
                                     tx=tx_cb.astype(np.complex128), rx_axis=-5, tx_axis=-3)
    return cases


def real_channel(res_path):
    """Realistic SI + target channel from the repo ray-tracing results: (3, 1, 8, 1, 8, 1, 2176)"""
    with np.load(res_path/"channel_impulse_responses.npz") as d:
        return np.concatenate([d["ht_si"], d["ht_tgt"]], axis=0)


def real_codebooks(n_ant=8):
    rx = codebook(n_ant, 32, transmit=False)
    tx = codebook(n_ant, 32, transmit=True)
    return rx, tx


# ---------------------------------------------------------------------------------------------
# ADC
# ---------------------------------------------------------------------------------------------

def adc_signal():
    return rand_complex((4, 3, 257), seed=500) * np.float32(2.5)


ADC_AXES = {
    "none": dict(axis=None, keepdims=False),
    "last_keep": dict(axis=-1, keepdims=True),
    "tuple_keep": dict(axis=(1, 2), keepdims=True),
    "first": dict(axis=0, keepdims=False),
}
ADC_BITS = (1, 3, 8)
