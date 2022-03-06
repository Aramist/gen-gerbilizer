#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH -c 2
#SBATCH --gpus=1
#SBATCH --mem=32768mb
#SBATCH --time=08:00:00
#SBATCH --output=cluster_logs/train_job_%j.log

if [ -z $1 ]
then
    echo "Job id should be provided as a command line argument"
    exit 1
fi

CFG_PATH=$2
if [ -z $2 ]
then
    echo "No config file provided, using cluster.json..."
    CFG_PATH=configs/cluster.json
fi

source ~/.bashrc
conda activate vox
srun python generative/train.py \
    $1 \
    CFG_PATH
