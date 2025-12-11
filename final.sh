#!/bin/bash
#SBATCH --account=PAS3162 
#SBATCH --job-name=test_osc_cv_final_drink36
#SBATCH --time=15:00:00
#SBATCH --cluster=ascend
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1

conda activate cv_env
python /users/PAS2119/drink36/AV-Deepfake1M/examples/xception/new_train.py  --data_root /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus   --model xception   --batch_size 1024   --precision bf16-mixed --gpus 1 --num_train 50000 --num_val 5000 --max_epochs 10