#!/bin/bash
#SBATCH --account=PAS3162 
#SBATCH --job-name=test_osc_cv_final_drink36
#SBATCH --time=12:00:00
#SBATCH --cluster=ascend
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1

conda activate cv_env
python /users/PAS2119/drink36/deepfake_project/models/trainv2.py --data_root /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus --train_metadata /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/train_metadata_filtered.json --val_metadata /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/validation_metadata_filtered.json --model videomae_v2 --batch_size 8 --gpus 1 --num_train 50000   --num_val 5000 --max_epochs 20