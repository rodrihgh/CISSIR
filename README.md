# CISSIR <img src="pics/cissir-logo.svg" height="120" align="right">

[![dependency - sionna](https://img.shields.io/badge/sionna->=0.17.0-green)][sionna]
[![dependency - cvxpy](https://img.shields.io/badge/cvxpy-1.5.2-blue)][cvxpy]
[![CC BY 4.0][cc-by-shield]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

[sionna]: https://nvlabs.github.io/sionna/
[cvxpy]: https://www.cvxpy.org/

Software repository for **Codebooks with Integral-Split Self-Interference Reduction**
(**CISSIR**, pronounced like "scissor").

Paper submission in progress.

![CISSIR diagram](pics/cissir_diagram.svg)

## Requirements

Full Python dependencies in the [requirements.txt](cluster/requirements.txt) file, including:

- [Sionna][sionna] >=0.17.0
- [Tensorflow](https://www.tensorflow.org/)
- [CVXPY][cvxpy]
- [Pandas](https://pandas.pydata.org/)
- [Seaborn](https://seaborn.pydata.org/)

### Installation

1. Install the requirements with [conda](https://docs.conda.io/projects/conda/en/stable/commands/install.html)
or [pip](https://pip.pypa.io/en/stable/cli/pip_install/). We recommend using a virtual environment for this.
2. To execute the [notebooks](notebooks), we recommend adding the project root to the `PYTHONPATH` so that the
[cissir](cissir) module is accessible everywhere. You can simply create an ad-hoc ipykernel with the proper
`PYTHONPATH` by running the following in the project root:
* In Unix:
```shell
python -m ipykernel install --sys-prefix --name cissir --display-name "CISSIR (python 3)" --env PYTHONPATH "${PYTHONPATH}:${PWD}"
```     
* In [Git Bash](https://gitforwindows.org/) for Windows:
```shell
python -m ipykernel install --sys-prefix --name cissir --display-name "CISSIR (python 3)" --env PYTHONPATH "${PYTHONPATH};$(pwd -W)"
```     

Afterwards, you can just execute `jupyter lab` and run the notebooks with the "CISSIR" kernel.

## Structure

- [cissir](cissir) - Local Python modules.
- [rt](rt) - [Blender](https://www.blender.org/) scene files for
[Sionna's Ray Tracer](https://nvlabs.github.io/sionna/api/rt.html). 
  - Developed with the help of
  [Danial Dehghani](https://www.linkedin.com/in/danial-dehghani/).
- [cluster](cluster) - Requirements, scripts and
[Apptainer](https://apptainer.org/) recipe for (optional)
execution on an [HPC GPU Cluster](https://www.nvidia.com/en-us/glossary/high-performance-computing/).
- [notebooks](notebooks) - [Jupyter](https://jupyter.org/) notebooks demonstrating the code.
- [results](results) - Channel and performance results are stored here.
- [plots](plots) - Generated graphs are stored here.

## Usage

The code can be directly used through [Jupyter notebooks](notebooks).
The following order is recommended:

0. Adjust the simulation parameters in [cissir/params.py](cissir/params.py).
1. Simulate the channel in [simulation/channel.ipynb](notebooks/simulation/channel.ipynb).
2. Run any of the following [optimization notebooks](notebooks/simulation):
   1. [tapered_cissir.ipynb](notebooks/optimization/tapered_cissir.ipynb)
   2. [phased_cissir.ipynb](notebooks/optimization/phased_cissir.ipynb)
   3. [lonestar.ipynb](notebooks/optimization/lonestar.ipynb)
3. Run the simulations:
   4. [snr_sensing.ipynb](notebooks/simulation/snr_sensing.ipynb).
   5. [si_codebook_comm.ipynb](notebooks/simulation/si_codebook_comm.ipynb).
4. Analyze and plot the results with the [analysis notebooks](notebooks/analysis):
   6. [link_level.ipynb](notebooks/analysis/link_level.ipynb)
   7. [trade_off.ipynb](notebooks/analysis/trade_off.ipynb)

## Citation

_TODO submission in progress_
   
---
