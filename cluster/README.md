# HPC cluster

The files in this directory allow to run the code on a GPU-based 
[High-Performance Computing (HPC) cluster](https://www.nvidia.com/en-us/glossary/high-performance-computing/).
Although not mandatory, GPU acceleration is encouraged in order to speed up
[Sionna](https://nvlabs.github.io/sionna/)'s ray-tracing and link-level simulations.

## Requirements

- [CUDA](https://developer.nvidia.com/cuda-zone)-capable HPC cluster.
- [Singularity/Apptainer](https://apptainer.org/) container support.
- [Slurm](https://slurm.schedmd.com/) scheduling.

## Setup

Navigate to the project's root directory in the command line and run:

```shell
singularity build --fakeroot --force cluster/sionna.sif cluster/sionna.def
```

[More information on Singularity container creation](https://docs.sylabs.io/guides/3.0/user-guide/build_a_container.html)

## Usage

After building the container, you can submit a job to the HPC cluster with [sbatch](https://slurm.schedmd.com/sbatch.html):

```shell
sbatch cluster/submit_job.sh
```

Once submitted, you can run the code on your browser
via [JupyterLab](https://jupyterlab.readthedocs.io/).
JupyterLab's URL will be logged and written into the file _jupyter.url_ in the project's root.

[More information on Slurm commands](https://slurm.schedmd.com/quickstart.html#commands)
