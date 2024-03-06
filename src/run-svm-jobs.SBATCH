#!/bin/bash

#SBATCH --job-name=SVM_Array
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --time=96:00:00
#SBATCH --mem=20GB
#SBATCH --array=0-9

module purge


configs=(./configs/Atezo_ALL.yaml,./configs/Atezo_COSMIC.yaml,./configs/Atezo_LINCS.yaml,\
    ./configs/Nivo_ALL.yaml,./configs/Nivo_COSMIC.yaml,./configs/Nivo_LINCS.yaml,
    ./configs/Pembro_ALL.yaml,./configs/Pembro_COSMIC.yaml,./configs/Pembro_LINCS.yaml)

singularity exec --nv \
            --overlay /scratch/jjb509/ORCKernels/src/my_overlay.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
            /bin/bash -c "source /ext3/env.sh; python run_svms.py -config ${configs[$SLURM_ARRAY_TASK_ID]}"
