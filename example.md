# Quick Start Guide for Instructors / TAs

This document provides the exact commands used to reproduce our results.
**Note:** All settings and performance estimates are based on a standard node configuration: **1 A100 GPU with 8 CPUs**.

## 0. Environment Setup

To replicate the environment used for this project, follow these steps:

1.  **Conda Environment**: Create and activate the `cv_env` Conda environment with Python 3.10.
    ```bash
    conda create -n cv_env python=3.10
    conda activate cv_env
    ```
2.  **Dependencies**: Install the required Python packages.
    ```bash
    pip install -r requirements.txt
    ```
---

## 0.5. Download Pre-trained Weights

Pre-trained model weights for Xception, R(2+1)D, and VideoMAE V2 are available for download.

1.  **Download from Google Drive**:
    Please download the weights from the following [Google Drive link](https://drive.google.com/drive/folders/1Foz4sQVNeFc2IR_1m0zp_gZ2zBwdpvO5)

2.  **Usage**:
    The downloaded `.ckpt` files can be used with the `models/infer.py` script by specifying the `--checkpoint` argument. For example:
    ```bash
    python models/infer.py \
        --checkpoint /path/to/your/downloaded_model.ckpt \
        --model videomae_v2 \
        # ... other arguments
    ```
    Ensure the `--model` argument matches the architecture of the downloaded checkpoint.

---

## 1. Inference

Run inference to generate prediction scores. Test dataset include 100 videos insides.

**Note:** Replace `<weight path>` with the actual path to your trained model checkpoint (e.g., `ckpt/xception_.../best_model.ckpt`).

### Xception
*   **Estimated Time:** ~10 sec
```bash
python models/infer.py --data_root test_dataset --checkpoint <weight path> --model xception --metadata_file test_dataset/test_subset.json --subset test --batch_size 1024 --gpus 1
```

### R(2+1)D
*   **Estimated Time:** ~10 sec
```bash
python models/infer.py --data_root test_dataset --checkpoint <weight path> --model r2plus1d --metadata_file test_dataset/test_subset.json --subset test --batch_size 32 --gpus 1
```

### VideoMAE V2
*   **Estimated Time:** ~1 min
```bash
python models/infer.py --data_root test_dataset --checkpoint <weight path> --model videomae_v2 --metadata_file test_dataset/test_subset.json --subset test --batch_size 32 --gpus 1
```

*Results will be saved in the `output/` directory.*

---

## 3. Evaluation

Calculate metrics (AUC) using the output file from the inference step.

**Please use the txt file without soft in the end.**
```bash
python models/evaluate.py <path_to_inference_output_file> GT/test_subset.json
```
