salloc --account=PAS3162 --cluster=ascend --job-name=bash --time=00:20:00 --ntasks=1 --ntasks-per-node=1 --nodes=1 --gpus-per-task=1 --cpus-per-task=8 bash
salloc --account=PAS3162 --cluster=ascend --job-name=bash --time=00:05:00 --ntasks=1 --ntasks-per-node=1 --nodes=1 --gpus-per-task=1 --cpus-per-task=8 bash


python models/train.py   --data_root C:\Users\ooo91\Desktop\School\ComputerVision\Final\test_dataset   --model xception   --batch_size 128   --precision bf16-mixed   --gpus 1   --num_train 1000   --num_val 200   --max_epochs 2 --train_metadata C:\Users\ooo91\Desktop\School\ComputerVision\Final\test_dataset\train_subset.json --val_metadata C:\Users\ooo91\Desktop\School\ComputerVision\Final\test_dataset\val_subset.json

python models/train.py --data_root C:\Users\ooo91\Desktop\School\ComputerVision\Final\test_dataset --train_metadata C:\Users\ooo91\Desktop\School\ComputerVision\Final\test_dataset\train_subset.json --val_metadata C:\Users\ooo91\Desktop\School\ComputerVision\Final\test_dataset\val_subset.json --model videomae_v2 --batch_size 8 --gpus 1 --num_train 50000   --num_val 5000

python models/train.py --data_root /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus --train_metadata /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/train_metadata_filtered.json --val_metadata /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/validation_metadata_filtered.json --model videomae_v2 --batch_size 8 --gpus 1 --num_train 100000   --num_val 5000

python models/train.py --data_root /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus --train_metadata /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/train_metadata_filtered.json --val_metadata /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/validation_metadata_filtered.json --model videomae_v2 --batch_size 8 --gpus 1 --num_train 100000   --num_val 5000 --clip_len 32


python models/infer.py --data_root /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus --checkpoint ./ckpt/videomae_v2_20251212_150507/last.ckpt --model videomae_v2 --metadata_file /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/validation_metadata_filtered.json --subset test --batch_size 8 --gpus 1 --take_num 1000


python models/infer.py --data_root /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus --checkpoint ./ckpt/videomae_v2/videomae_v2-epoch=7-val_loss=0.130.ckpt --model videomae_v2 --metadata_txt /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/testB_files.txt --subset testB --batch_size 8 --gpus 1
python models/evaluate.py output/videomae_v2_val_10000_20251214-011831.txt validation_metadata_filtered_top10000.json

python models/train.py --data_root /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus --train_metadata /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/train_metadata_filtered.json --val_metadata /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/validation_metadata_filtered.json --model r2plus1d --batch_size 8 --gpus 1 --num_train 100000   --num_val 5000



python models/infer.py --data_root /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus --checkpoint ./ckpt/r2plus1d_20251214_050221/last.ckpt --model r2plus1d --metadata_file /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/test_metadata_filtered.json --subset test --batch_size 32 --gpus 1 --take_num 10000

python models/infer.py --data_root /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus --checkpoint ./ckpt/xception/last-v1.ckpt --model xception --metadata_file /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/validation_metadata_filtered.json --subset test --batch_size 32 --gpus 1 --take_num 1000

python models/evaluate.py videomae_v2_val_10000_20251214-011831_combined.txt validation_metadata_filtered_top10000_combined.json