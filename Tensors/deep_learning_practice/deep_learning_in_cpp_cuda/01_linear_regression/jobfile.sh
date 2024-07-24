#!/bin/bash -l                 
#
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
                                 
#unset SLURM_EXPORT_ENV             # enable export of environment from this script to srun
            
module load cuda/11.8.0 cudnn/8.8.0.121-11.8

export TORCH_PATH=$HOME/libtorch
export LD_LIBRARY_PATH=$TORCH_PATH/lib:$LD_LIBRARY_PATH

cd $HOME/deep_learning_in_cpp/MLP/build
ls -l

./linear_regression