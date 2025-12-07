salloc --account=PAS3162 --cluster=ascend --job-name=bash --time=00:20:00 --ntasks=1 --ntasks-per-node=1 --nodes=1 --gpus-per-task=1 --cpus-per-task=8 bash
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

python /users/PAS2119/drink36/AV-Deepfake1M/examples/xception/new_infer.py --data_root /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus --checkpoint /users/PAS2119/drink36/AV-Deepfake1M/ckpt/xception/last.ckpt  --model xception --subset train --batch_size 1024 --take_num 300

time ffmpeg -i /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/train/vox_celeb_2/id01358/_1nATum8x78/00030/real.mp4 -threads 32 -vf "scale=96:96:flags=bicubic" /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/train/vox_celeb_2/id01358/_1nATum8x78/00030/out/%06d.png


python /users/PAS2119/drink36/AV-Deepfake1M/examples/xception/new_train.py   --data_root /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus   --model xception   --batch_size 1024   --precision bf16-mixed   --gpus 1   --num_train 50000   --num_val 2000   --max_epochs 50

python models/train.py   --data_root /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus   --model xception   --batch_size 1024   --precision bf16-mixed   --gpus 1   --num_train 50000   --num_val 2000   --max_epochs 50 --train_metadata /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/train_metadata_filtered.json --val_metadata /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/validation_metadata_filtered.json

python models/train.py   --data_root /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus   --model xception   --batch_size 1024   --precision bf16-mixed   --num_train 50000   --num_val 2000   --max_epochs 50 --train_metadata /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/train_metadata_filtered.json --val_metadata /fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/validation_metadata_filtered.json