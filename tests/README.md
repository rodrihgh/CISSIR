# Tests

Run the tests from the project root with the `cissir3` environment (see the [main README](../README.md)):

```shell
pytest
```

[pytest.ini](../pytest.ini) already adds the project root to the Python path.

## Test files

| File | What it checks | Needs data files |
| :--- | :------------- | :--------------- |
| [test_sionna_isac.py](test_sionna_isac.py) | `MatchedFilter` (built on `ofdm_to_time_channel`) and `steer_vec`/`array_factor` (built on `steering_vectors`) against the previous NumPy implementations. | No |
| [test_torch_parity.py](test_torch_parity.py) | Numerical parity of the PyTorch code (`adc`, `beamforming`, `raytracing`, `signal_power`) with the former TensorFlow implementation. | Yes, for most tests |

`test_torch_parity.py` compares against outputs of the **TensorFlow-based code** from before the PyTorch migration.
That reference is not part of the repository, but it can be generated as described below.
If the data files are missing, the tests that need them are **skipped** (with a pointer to this file) instead of failing; `pytest -rs` lists the reasons.
To run only the data-free tests:

```shell
pytest tests/test_sionna_isac.py
```

## Data files

Both files live in `tests/data/` (git-ignored) and are only used by `test_torch_parity.py`:

| File | Size | Content |
| :--- | :--- | :------ |
| `real_channel.npz` | ~1.2 MB | Frozen copy of the self-interference and target channels (first 640 taps of `results/channel_impulse_responses.npz`). It keeps the tests independent of the contents of `results/`, which change whenever the ray tracing is re-run. |
| `tf_reference.npz` | ~2 MB | Outputs of the TensorFlow implementation for the seeded test inputs of [cases.py](cases.py) and for `real_channel.npz`. |

`tf_reference.npz` is computed **from** `real_channel.npz`, so whenever you refresh the channel you must also regenerate the reference.

### 1. Channel fixture

It only needs `results/channel_impulse_responses.npz` (run [channel.ipynb](../notebooks/simulation/channel.ipynb) first if it is missing).
In the `cissir3` environment, from the project root:

```shell
python -c "import sys; sys.path.insert(0, 'tests'); import cases; cases.make_real_channel_fixture('results')"
```

### 2. TensorFlow reference

This step needs the **old environment**, since it runs the TensorFlow implementation of the last commit before the migration
(`fa6a72f`), which is extracted from the git history. Create the environment with the requirements of that commit
(Python 3.11, TensorFlow 2.16.1, Sionna 1.2.1):

```shell
git show fa6a72f:cluster/requirements.txt > requirements_tf.txt
conda create -n cissir -c conda-forge python=3.11
conda activate cissir
pip install -r requirements_tf.txt
```

Then, from the project root and with this environment active:

```shell
python tests/gen_tf_reference.py
```

The script creates `tests/data/real_channel.npz` if it is missing and writes `tests/data/tf_reference.npz`.
Useful options:

- `--refresh-fixture`: rebuild `real_channel.npz` from the current `results/channel_impulse_responses.npz`
(e.g., after re-running the ray tracing), and regenerate the reference accordingly.
- `--rev`: git revision holding the TensorFlow-based `cissir` package (default: `fa6a72f`).
- `--out`: output path of the reference.

The script reads that revision with `git archive`, so the git history must be complete (use `git fetch --unshallow` if the repository was cloned with `--depth`).
No GPU is needed.

## Adding tests

- Inputs should be seeded and framework-free (see [cases.py](cases.py)), so that they are identical in both environments.
- Prefer self-contained tests like those in `test_sionna_isac.py`, with a short reference implementation inside the test file,
over new data files.
