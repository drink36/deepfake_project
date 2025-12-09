import argparse
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from data.dataset import DeepfakeDataset, DeepfakeClipDataset
from xception import Xception
from utils import LrLogger, EarlyStoppingLR
from videomae_v2 import DeepfakeVideoMAEV2
parser = argparse.ArgumentParser(description="Classification model training")
parser.add_argument("--data_root", type=str)
parser.add_argument("--train_metadata", type=str, required=True)
parser.add_argument("--val_metadata", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=8) # VideoMAE 很吃 VRAM，建議先改小 (8 或 16)
# === 2. 在參數裡加入 videomae_v2 ===
parser.add_argument("--model", type=str, choices=["xception", "meso4", "videomae_v2"]) 
parser.add_argument("--gpus", type=int, default=1)
parser.add_argument("--max_epochs", type=int, default=50) # Fine-tuning 通常不用跑太多 epoch
parser.add_argument("--num_train", type=int, default=None)
parser.add_argument("--num_val", type=int, default=2000)
parser.add_argument("--precision", default="16-mixed") # 建議用 16-mixed 省顯存
args = parser.parse_args()
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if __name__ == "__main__":

    # You can fix the random seed if you want reproducible subsets each epoch:
    # torch.manual_seed(42)
    # random.seed(42)

    learning_rate = 1e-4
    gpus = args.gpus
    total_batch_size = args.batch_size * gpus
    # learning_rate = learning_rate * total_batch_size / 4

    # Setup model
    if args.model == "xception":
        model = Xception(learning_rate, distributed=gpus > 1)
    elif args.model == "videomae_v2":
        print("🚀 Initializing VideoMAE V2...")
        model = DeepfakeVideoMAEV2(
            learning_rate=1e-4, 
            freeze_backbone=False, # 先試試看全量微調，跑不動再設 True
            distributed=args.gpus > 1   # <--- 關鍵修正：傳入這個參數！
        )
        image_size = 224 # VideoMAE V2 必須是 224
        use_clip = True  # 標記這是 3D 模型
    else:
        raise ValueError(f"Unknown model: {args.model}")

    if use_clip:
        # --- 3D 模型用 DeepfakeClipDataset (吐 16 幀 Clip) ---
        print(f"📦 Using 3D Clip Dataset (Clip Length=16, Size={image_size})")
        
        # 讀取 JSON list (假設你有寫好 helper function 或直接在這裡讀)
        import json
        with open(args.train_metadata, 'r') as f:
            train_meta = json.load(f)
        with open(args.val_metadata, 'r') as f:
            val_meta = json.load(f)
            
        # 這裡需要轉換成 VideoMetadata 物件 (假設你的 Dataset 支援直接傳 list)
        from data.dataset import VideoMetadata
        train_meta_obj = [VideoMetadata(**item) for item in train_meta]
        val_meta_obj = [VideoMetadata(**item) for item in val_meta]
        if args.num_train: train_meta_obj = train_meta_obj[:args.num_train]
        if args.num_val: val_meta_obj = val_meta_obj[:args.num_val]

        train_dataset = DeepfakeClipDataset(
            data_root=args.data_root,
            metadata=train_meta_obj,
            clip_len=16,      # VideoMAE 標準長度
            image_size=image_size,
            take_num=args.num_train,
            mode='train'      # 啟用 Smart Sampling
        )
        
        val_dataset = DeepfakeClipDataset(
            data_root=args.data_root,
            metadata=val_meta_obj,
            clip_len=16,
            image_size=image_size,
            take_num=args.num_val,
            mode='test'       # Validation 用中心裁切
        )
        
    else:
        # --- 2D 模型用原本的 Dataset ---
        print(f"🖼️ Using 2D Frame Dataset (Size={image_size})")
        train_dataset = DeepfakeDataset(
            data_root=args.data_root,
            json_file=args.train_metadata,
            image_size=image_size,
            take_num=args.num_train
        )
        val_dataset = DeepfakeDataset(
            data_root=args.data_root,
            json_file=args.val_metadata,
            image_size=image_size,
            take_num=args.num_val
        )

    # === 5. Trainer 設定 (保持原樣) ===
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"./ckpt/{args.model}",
        save_last=True,
        filename=args.model + "-{epoch}-{val_loss:.3f}",
        monitor="val_loss",
        mode="min"
    )

    trainer = Trainer(
        precision=args.precision,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, LrLogger(), EarlyStoppingLR(lr_threshold=1e-7)],
        accelerator="gpu",
        devices=args.gpus,
        strategy="ddp" if args.gpus > 1 else "auto",
        log_every_n_steps=50
    )

    # 開始訓練
    trainer.fit(
        model,
        train_dataloaders=DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True, persistent_workers=True),
        val_dataloaders=DataLoader(val_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, persistent_workers=True)
    )