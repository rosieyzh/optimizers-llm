#!/bin/bash
#SBATCH --job-name=optimizers-llm
#SBATCH --output=logs/%A_%a.log
#SBATCH --nodes=1              
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1     
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --mem=250GB		
#SBATCH --constraint=h100

# Custom environment
source ~/.bashrc
conda deactivate
conda activate opt-olmo

# Accept model config as first argument
export CONFIG=configs/base.yaml+${1}

# Accept sweep config as second argument
export SWEEP_CONFIG=$2

# Accept job index as argument if there is a third argument
if [ -z "$3" ]
then
    echo $SLURM_ARRAY_TASK_ID
else
    export SLURM_ARRAY_TASK_ID=$3
fi

# Set default path for checkpoints if not set
if [ -z "$CHECKPOINTS_PATH" ]
then
    export CHECKPOINTS_PATH=ckpts
fi

# Set ntasks if not set
if [ -z "$SLURM_NTASKS_PER_NODE" ]
then
    export SLURM_NTASKS_PER_NODE=1
fi


# Boilerplate environment variables
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MPICH_GPU_SUPPORT_ENABLED=1
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}

export PYTHONPATH=.:${PYTHONPATH}

export PYTORCH_KERNEL_CACHE_PATH=/tmp/pytorch_kernel_cache/
mkdir -p $PYTORCH_KERNEL_CACHE_PATH

python scripts/run_sweep.py config=${CONFIG} sweep_config=${SWEEP_CONFIG}