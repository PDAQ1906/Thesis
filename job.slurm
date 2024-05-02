#!/bin/bash
#SBATCH --job-name=biomedclip        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=03:45:00          # total run time limit (HH:MM:SS)

module purge
module load anaconda3/2023.3
conda activate tf2-gpu

python myscript.py