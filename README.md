# CISSIR

Software repository for **Codebooks with Integral-Split Self-Interference Reduction**
(**CISSIR**, pronounced like "scissor").

Paper submission in progress.

## Requirements

Full Python dependencies in the [requirements.txt](cluster/requirements.txt) file, including:

- [Sionna](https://nvlabs.github.io/sionna/) >=0.17.0
- [Tensorflow](https://www.tensorflow.org/)
- [CVXPY](https://www.cvxpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Seaborn](https://seaborn.pydata.org/)

## Structure

- [cissir](cissir) - Local Python modules.
- [rt](rt) - Blender scene files for
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

0. Adjust the simulation parameters in [params.py](cissir/params.py).
1. Run the [channel_simulator.ipynb](notebooks/channel_simulator.ipynb).
2. Run any of the following optimization notebooks:
   1. [lagrange_codebook_optimization.ipynb](notebooks/lagrange_codebook_optimization.ipynb)
   2. [sdr_codebook_optimization.ipynb](notebooks/sdr_codebook_optimization.ipynb)
   3. [socp_codebook_optimization.ipynb](notebooks/socp_codebook_optimization.ipynb)
   4. [lonestar.ipynb](notebooks/lonestar.ipynb)
3. Run the sensing evaluation [sqnr_bound.ipynb](notebooks/sqnr_bound.ipynb).
4. Run the link-level simulations [mimo_ofdm_comm.ipynb](notebooks/mimo_ofdm_comm.ipynb).
5. Plot the curves:
   1. [plot_tradeoffs.ipynb](notebooks/plot_tradeoffs.ipynb)
   2. [plot_comm.ipynb](notebooks/plot_comm.ipynb)
