python models/infer.py --data_root test_dataset --checkpoint ./ckpt/videomae_v2_20251212_150507/last.ckpt --model videomae_v2 --metadata_file test_dataset/test_subset.json --subset test --batch_size 32 --gpus 1
python models/infer.py --data_root test_dataset --checkpoint ./ckpt/r2plus1d_20251214_050221/last.ckpt --model r2plus1d --metadata_file test_dataset/test_subset.json --subset test --batch_size 32 --gpus 1
python models/infer.py --data_root test_dataset --checkpoint ./ckpt/xception_20251215_202308/last.ckpt --model xception --metadata_file test_dataset/test_subset.json --subset test --batch_size 1024 --gpus 1
python models/evaluate.py output/videomae_v2_test_100_20251216-163644.txt GT/test_subset.json
python models/evaluate.py output/r2plus1d_test_100_20251216-163842.txt GT/test_subset.json
python models/evaluate.py output/xception_test_100_20251216-164053.txt GT/test_subset.json
