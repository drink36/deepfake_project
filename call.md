salloc --account=PAS3162 --cluster=ascend --job-name=bash --time=00:20:00 --ntasks=1 --ntasks-per-node=1 --nodes=1 --gpus-per-task=1 --cpus-per-task=8 bash
salloc --account=PAS3162 --cluster=ascend --job-name=bash --time=00:05:00 --ntasks=1 --ntasks-per-node=1 --nodes=1 --gpus-per-task=1 --cpus-per-task=8 bash
python /users/PAS2119/drink36/AV-Deepfake1M/examples/xception/new_train.py \
  --data_root /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus \
  --model xception \
  --batch_size 1024 \
  --precision bf16-mixed \
  --gpus 1 \
  --num_train 5000 \
  --num_val 2000 \
  --max_epochs 50
squeue -M ascend -u drink36
sbatch final.sh

python models/infer.py --data_root /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus --checkpoint ckpt/xception/last.ckpt  --model xception --subset train --batch_size 1024 --take_num 300

time ffmpeg -i /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/train/vox_celeb_2/id01358/_1nATum8x78/00030/real.mp4 -threads 32 -vf "scale=96:96:flags=bicubic" /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/train/vox_celeb_2/id01358/_1nATum8x78/00030/out/%06d.png


python /users/PAS2119/drink36/AV-Deepfake1M/examples/xception/new_train.py   --data_root /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus   --model xception   --batch_size 1024   --precision bf16-mixed   --gpus 1   --num_train 50000   --num_val 2000   --max_epochs 50

python models/train.py   --data_root /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus   --model xception   --batch_size 1024   --precision bf16-mixed   --gpus 1   --num_train 50000   --num_val 2000   --max_epochs 50 --train_metadata /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/train_metadata_filtered.json --val_metadata /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/validation_metadata_filtered.json

python models/train.py   --data_root /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus   --model xception   --batch_size 1024   --precision bf16-mixed   --gpus 1   --num_train 1000   --num_val 200   --max_epochs 2 --train_metadata /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/train_metadata_filtered.json --val_metadata /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/validation_metadata_filtered.json

python models/infer.py \
  --data_root /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus \
  --checkpoint ckpt/xception/last.ckpt \
  --model xception \
  --metadata_file first_test.json \
  --subset eval_300 \
  --batch_size 1024 \
  --gpus 1

python models/evaluate.py output/xception_eval_300.txt first_test.json




python models/train.py   --data_root C:\Users\ooo91\Desktop\School\ComputerVision\Final\test_dataset   --model xception   --batch_size 128   --precision bf16-mixed   --gpus 1   --num_train 1000   --num_val 200   --max_epochs 2 --train_metadata C:\Users\ooo91\Desktop\School\ComputerVision\Final\test_dataset\train_subset.json --val_metadata C:\Users\ooo91\Desktop\School\ComputerVision\Final\test_dataset\val_subset.json

python models/trainv2.py --data_root C:\Users\ooo91\Desktop\School\ComputerVision\Final\test_dataset --train_metadata C:\Users\ooo91\Desktop\School\ComputerVision\Final\test_dataset\train_subset.json --val_metadata C:\Users\ooo91\Desktop\School\ComputerVision\Final\test_dataset\val_subset.json --model videomae_v2 --batch_size 8 --gpus 1 --num_train 50000   --num_val 5000

python models/trainv2.py --data_root /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus --train_metadata /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/train_metadata_filtered.json --val_metadata /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/validation_metadata_filtered.json --model videomae_v2 --batch_size 8 --gpus 1 --num_train 100000   --num_val 5000

python models/trainv2.py --data_root /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus --train_metadata /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/train_metadata_filtered.json --val_metadata /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/validation_metadata_filtered.json --model videomae_v2 --batch_size 8 --gpus 1 --num_train 100000   --num_val 5000 --clip_len 32


python models/infer.py --data_root /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus --checkpoint ./ckpt/videomae_v2_20251212_150507/last.ckpt --model videomae_v2 --metadata_file /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/validation_metadata_filtered.json --subset test --batch_size 8 --gpus 1 --take_num 1000


python models/infer.py --data_root /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus --checkpoint ./ckpt/videomae_v2/videomae_v2-epoch=7-val_loss=0.130.ckpt --model videomae_v2 --metadata_txt /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/testB_files.txt --subset testB --batch_size 8 --gpus 1
python models/evaluate.py output/videomae_v2_val_10000_20251214-011831.txt validation_metadata_filtered_top10000.json

python models/trainv2.py --data_root /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus --train_metadata /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/train_metadata_filtered.json --val_metadata /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/validation_metadata_filtered.json --model r2plus1d --batch_size 8 --gpus 1 --num_train 100000   --num_val 5000

python models/evaluate.py videomae_v2_val_10000_20251214-011831_combined.txt validation_metadata_filtered_top10000_combined.json

python models/infer.py --data_root /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus --checkpoint ./ckpt/r2plus1d_20251214_050221/last.ckpt --model r2plus1d --metadata_file /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/test_metadata_filtered.json --subset test --batch_size 32 --gpus 1 --take_num 10000

python models/infer.py --data_root /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus --checkpoint ./ckpt/xception/last-v1.ckpt --model xception --metadata_file /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/validation_metadata_filtered.json --subset test --batch_size 32 --gpus 1 --take_num 1000

python combine.py output/videomae_v2_test_10000_20251215-035913.txt output/videomae_v2_audio_5000_20251214-132251.txt 2000
python combine.py output/r2plus1d_test_10000_20251215-040018.txt output/r2plus1d_test_5000_20251214-150049.txt 2000
python combine.py output/xception_test_10000_20251215-143319.txt output/xception_audio_5000_20251215-144601.txt 2000
python combine.py test_metadata_filtered_top10000.json audio_mod_validation_metadata_filtered_top5000.json 2000