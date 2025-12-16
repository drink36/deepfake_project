#!/bin/bash
#SBATCH --account=PAS3162 
#SBATCH --job-name=test_osc_cv_final_drink36
#SBATCH --time=02:00:00
#SBATCH --cluster=ascend
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1

conda activate cv_env
python models/infer.py --data_root /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus --checkpoint ./ckpt/xception_20251215_202308/last.ckpt --model xception --metadata_txt testB_files.txt --subset testB --batch_size 1024 --gpus 1