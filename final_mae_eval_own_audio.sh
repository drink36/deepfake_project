#!/bin/bash
#SBATCH --account=PAS3162 
#SBATCH --job-name=test_osc_cv_final_drink36
#SBATCH --time=1:00:00
#SBATCH --cluster=pitzer
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
conda activate cv_env
python models/infer.py --data_root /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus --checkpoint ./ckpt/videomae_v2_20251212_150507/last.ckpt --model videomae_v2 --metadata_file /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/audio_mod_validation_metadata_filtered.json --subset audio --batch_size 32 --gpus 1 --take_num 5000