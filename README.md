# Distributed Training on the mlcloud

This repository demonstrates and explains how one can train neural networks in a distributed manner with torch and horovod on the mlcloud. Its contribution is multifold:

1. It acts as documentation and tutorial for various aspects of distributed training, especially with the topology of the mlcloud in mind.
2. It provides various utility scripts for slurm and a script to initialize a conda environment that is ready for distributed training.
3. It already provides a fleshed out training skeleton that one can either use directly or borrow from.

## Conda Setup

This repository already provides a script that installs a conda environment with full cuda and horovod support, including support for NCCL and MPI. In particular, the script provides an environment that does not suffer from any version mismatchs (as of 07.05.2023).

### Step 1: Setting your conda location

It is strongly recommended to move conda environments to either `$WORK` or `$WORK2` and not locate them in your home directory (default). If you have already done so, you can skip this step.

1. Copy `environment/.condarc` to your home directory.
2. Create two folders that will hold your environments and the actual dowloaded packages respectively:
    - `mkdir -p $WORK2/conda/envs $WORK2/conda/pkgs`

### Step 2: Install mamba

To significantly speed up the installation process, the installer script uses mamba instead of conda. Mamba is a native, i.e. compiled, implementation of conda, but exposes the same CLI.

```
conda create --name mamba
conda install -n mamba -c conda-forge mamba
```

### Step 3: Create the environment

1. Make sure that your working directory is `./environment`.
2. Make sure that mamba is available: `conda activate mamba`
3. Execute `./create_env.sh YOUR_NAME`, where *YOUR_NAME* is the name of your new environment.

## Features of the Training Skeleton

1. Distributed training
2. Mixed precision support (optional)
3. Wandb support
4. Checkpointing and support for preemptable partitions
5. Logging of various useful diagnostic informations during startup
6. NaN checks for debugging purposes (optional)
7. A pre-defined structure for defining configs and experiments that still allows for a lot of flexibility

## Using the Training Skeleton

Have a look at `dmlcloud/experiments/mnist.py` for a simple example on how to create a new experiment. In general, the following steps need to be done:

1. Sub-class `BaseTrainer` and provide functions which create your model, dataset, loss, etc.
2. Write a `create_trainer` function that creates a new instance of your Trainer given parsed command line arguments from `argparse`.
3. Write a `add_parser` function that adds a new subparser for your experiments and registers all command line options with argparse.
4. Optionally, define new subconfigs for options specific to your experiment or model
5. Import and register your experiment in the cli script `dmlcloud/cli/train.py`.

Notice that step 2-5 are only required if you want to use the already provided cli training script at `dmlcloud/cli/train.py`.

The main motivation behind the config system is to separate where configs come from, e.g. argparse, from their actual usage. Due to this you could for instance also opt to disregard these steps 2-5 and instead read configs from a json file.
