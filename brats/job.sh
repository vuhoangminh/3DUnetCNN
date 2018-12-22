#!/bin/bash
#SBATCH -A SNIC2018-3-406
#SBATCH -n 1
##SBATCH --exclusive
#SBATCH --time=04-00:00:00
#SBATCH --error=%J_error.out
#SBATCH --output=%J_output.out
#SBATCH --gres=gpu:k80:1

ml GCC/6.4.0-2.28  CUDA/9.0.176  OpenMPI/2.1.1

export KERAS_BACKEND="tensorflow"

export command="python brats/train.py -t "0" -o "0" -n "01" -de "median" -hi "0" -ba 2"
echo "$command"
srun $command

wait
