# For TA / Instructor (How to Run)
## All setting were based on 1 A100 with 8 cpu
### train
```bash
# Xception (around  12 hr)
python models/train.py --data_root /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus --train_metadata /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/train_metadata_filtered.json --val_metadata /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/validation_metadata_filtered.json --model xception --batch_size 1024 --gpus 1 --num_train 50000   --num_val 5000 --max_epochs 10
# R(2+1)D (around 10 hr)
python models/train.py --data_root /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus --train_metadata /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/train_metadata_filtered.json --val_metadata /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/validation_metadata_filtered.json --model r2plus1d --batch_size 64 --gpus 1 --num_train 100000   --num_val 5000 --max_epochs 20
# VideoMAEV2 (around 15 hr)
python models/train.py --data_root /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus --train_metadata /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/train_metadata_filtered.json --val_metadata /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/validation_metadata_filtered.json --model videomae_v2 --batch_size 8 --gpus 1 --num_train 100000   --num_val 5000 --max_epochs 15
```
### inference
```bash
# Xception (around 1 min)
python models/infer.py --data_root /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus --checkpoint ./ckpt/xception_20251215_202308/last.ckpt --model xception --metadata_file /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/test_metadata_filtered.json --subset val --batch_size 1024 --gpus 1 --take_num 1000
# R(2+1)D (around 1 min)
python models/infer.py --data_root /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus --checkpoint ./ckpt/r2plus1d_20251214_050221/last.ckpt --model r2plus1d --metadata_file /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/test_metadata_filtered.json --subset val --batch_size 32 --gpus 1 --take_num 1000
# VideoMAEV2 (around 10 min)
python models/infer.py --data_root /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus --checkpoint ./ckpt/videomae_v2_20251212_150507/last.ckpt --model videomae_v2 --metadata_file /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/test_metadata_filtered.json --subset test --batch_size 32 --gpus 1 --take_num 1000
```
After inference, result would show in output file.
### evaluate
```bash
 python models/evaluate.py <file_name> test_metadata_filtered_top1000.json
```