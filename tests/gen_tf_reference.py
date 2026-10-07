"""
Generate TensorFlow reference outputs for the PyTorch parity tests.

Run this with the *old* environment (TensorFlow 2.16 + Sionna 1.x), e.g.:

    conda activate cissir
    python tests/gen_tf_reference.py            # uses the last TF-based commit

The pre-migration ``cissir`` package is extracted from git into a temporary directory, so the
script does not depend on the state of the working tree. Output: ``tests/data/tf_reference.npz``.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

repo = Path(__file__).parents[1]
DEFAULT_REV = "fa6a72f"  # last commit before the PyTorch migration

parser = argparse.ArgumentParser()
parser.add_argument("--rev", default=DEFAULT_REV, help="git revision with the TensorFlow-based cissir package")
parser.add_argument("--out", default=str(repo/"tests"/"data"/"tf_reference.npz"))
parser.add_argument("--refresh-fixture", action="store_true",
                    help="re-create tests/data/real_channel.npz from results/channel_impulse_responses.npz")
args = parser.parse_args()

# Extract the TF-based package into a temp dir and make it take precedence over the working tree
tmp = tempfile.TemporaryDirectory()
archive = subprocess.run(["git", "archive", args.rev, "cissir"], cwd=repo, check=True, capture_output=True).stdout
subprocess.run(["tar", "-x", "-C", tmp.name], input=archive, check=True)
sys.path.insert(0, tmp.name)
sys.path.insert(1, str(repo/"tests"))

import tensorflow as tf  # noqa: E402
import cases  # noqa: E402
from cissir import adc, beamforming as bf, raytracing as rtr  # noqa: E402

assert Path(bf.__file__).is_relative_to(tmp.name), "Not using the extracted TF package"
print("TensorFlow", tf.__version__, "| cissir from", args.rev)

out = {}
res_path = repo/"results"

# The tests use a frozen copy of the ray-tracing channel (see cases.py)
if args.refresh_fixture or not cases.FIXTURE_PATH.exists():
    cases.FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Created fixture", cases.make_real_channel_fixture(res_path))

# --- channel_power --------------------------------------------------------------------------
h = cases.channel_power_input()
for name, kw in cases.CHANNEL_POWER_KWARGS.items():
    out[f"channel_power/{name}"] = bf.channel_power(tf.constant(h), **kw).numpy()

# --- BeamSelection --------------------------------------------------------------------------
for name, case in cases.BEAM_SELECTION_CASES.items():
    h = cases.beam_selection_input(case)
    out[f"beam_selection/{name}"] = bf.BeamSelection(**case["kwargs"])(tf.constant(h)).numpy()

# --- Beamspace ------------------------------------------------------------------------------
for name, case in cases.beamspace_cases().items():
    # Sionna blocks cast inputs to their precision (single by default), so double needs to be explicit
    double = case["h"].dtype == np.complex128
    layer = bf.Beamspace(transmit_axis=case["tx_axis"], receive_axis=case["rx_axis"],
                         precision="double" if double else None)
    inputs = [tf.constant(case["h"])]
    inputs += [tf.constant(case[k]) for k in ("rx", "tx") if case[k] is not None]
    out[f"beamspace/{name}"] = layer(*inputs).numpy()

def store_long(name, arr, stride=32):
    """Large time-domain outputs: keep a decimated slice plus full-length reductions over time"""
    out[f"{name}/slice"] = arr[..., ::stride]
    out[f"{name}/energy"] = np.sum(np.abs(arr.astype(np.complex128))**2, axis=-1)
    out[f"{name}/sum"] = np.sum(arr.astype(np.complex128), axis=-1)


# --- Realistic pipeline: Beamspace -> BeamSelection on ray-tracing channels ---------------------
h_real = cases.real_channel()
rx_cb, tx_cb = cases.real_codebooks()
rx_axis, tx_axis = bf.sionna_mimo_axes("ofdm")
h_bs = bf.Beamspace(transmit_axis=tx_axis, receive_axis=rx_axis)(
    tf.constant(h_real), tf.constant(rx_cb), tf.constant(tx_cb))
store_long("real/beamspace_full", h_bs.numpy())
# single-beam (SISO) beamspace, as used in the sensing notebooks
h_siso = bf.Beamspace(transmit_axis=tx_axis, receive_axis=rx_axis)(
    tf.constant(h_real), tf.constant(rx_cb[:, :1]), tf.constant(tx_cb[:, :1]))
store_long("real/beamspace_siso", h_siso.numpy())
# transmit-only beamspace followed by beam selection, as used in the communication notebook
h_tx = bf.Beamspace(transmit_axis=tx_axis)(tf.constant(h_real), tf.constant(tx_cb))
store_long("real/beamspace_tx", h_tx.numpy())
for ortho in (True, False):
    for k in (1, 4):
        sel = bf.BeamSelection(k, orthogonal=ortho, normalize=True)
        store_long(f"real/select_{'ortho' if ortho else 'simple'}_k{k}", sel(h_tx).numpy())

# --- ADC ------------------------------------------------------------------------------------
sig = cases.adc_signal()
out["adc/complex_max"] = adc.complex_max(tf.constant(sig)).numpy()
for name, kw in cases.ADC_AXES.items():
    out[f"adc/max_abs_complex/{name}"] = adc.max_abs_complex(tf.constant(sig), **kw).numpy()
    out[f"adc/papr/{name}"] = adc.papr(tf.constant(sig), **kw).numpy()
    out[f"adc/variance/{name}"] = tf.math.reduce_variance(
        tf.constant(sig), axis=kw["axis"], keepdims=kw["keepdims"]).numpy()
for bits in cases.ADC_BITS:
    out[f"adc/quantize/{bits}/auto"] = adc.quantize_signal(tf.constant(sig), bits).numpy()
    out[f"adc/quantize/{bits}/fixed"] = adc.quantize_signal(tf.constant(sig), bits, max_value=4.0).numpy()

# --- raytracing -----------------------------------------------------------------------------
ht_si, t_channel_s = cases.real_si_data()
with tempfile.TemporaryDirectory() as tdir:
    fname = Path(tdir)/"si_mimo.npz"
    t_si = rtr.save_si_matrix(tf.constant(ht_si), t_channel_s, fname=fname)
    with np.load(fname) as d:
        out["rtr/si_matrix/h_si_matrix"] = d["h_si_matrix"]
        out["rtr/si_matrix/mean_delay"] = d["mean_delay"]
    out["rtr/si_matrix/returned"] = t_si.numpy()

out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out_path, **out)
print(f"Saved {len(out)} reference arrays to {out_path} ({out_path.stat().st_size/1e6:.2f} MB)")
