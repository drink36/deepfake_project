import argparse
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from data.dataset import DeepfakeDataset, DeepfakeClipDataset, VideoMetadata
from xception import Xception
from utils import LrLogger, EarlyStoppingLR
from models.videomae_v2 import DeepfakeVideoMAEV2

parser = argparse.ArgumentParser(description="Classification model training")
parser.add_argument("--data_root", type=str)
parser.add_argument("--train_metadata", type=str, required=True)
parser.add_argument("--val_metadata", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--model", type=str, choices=["xception", "meso4", "videomae_v2"])
parser.add_argument("--gpus", type=int, default=1)
parser.add_argument("--max_epochs", type=int, default=50)
parser.add_argument("--num_train", type=int, default=None)
parser.add_argument("--num_val", type=int, default=2000)
parser.add_argument("--precision", default="16-mixed")
args = parser.parse_args()

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


if __name__ == "__main__":
    learning_rate = 1e-4
    gpus = args.gpus
    use_clip = False
    image_size = 96  
    if args.model == "xception":
        print("🚀 Initializing Xception...")
        model = Xception(learning_rate, distributed=gpus > 1)
        image_size = 299
        use_clip = False

    elif args.model == "videomae_v2":
        model = DeepfakeVideoMAEV2(
            learning_rate=learning_rate,
            freeze_backbone=False,
            distributed=gpus > 1,
        )
        image_size = 224         
        use_clip = True          

    else:
        raise ValueError(f"Unknown model: {args.model}")

    if use_clip:
        print(f"📦 Using 3D Clip Dataset (Clip Length=16, Size={image_size})")

        import json
        with open(args.train_metadata, "r") as f:
            train_meta_json = json.load(f)
        with open(args.val_metadata, "r") as f:
            val_meta_json = json.load(f)

        train_meta_obj = [VideoMetadata(**item) for item in train_meta_json]
        val_meta_obj = [VideoMetadata(**item) for item in val_meta_json]

        train_dataset = DeepfakeClipDataset(
            data_root=args.data_root,
            metadata=train_meta_obj,
            clip_len=16,
            image_size=image_size,
            take_num=args.num_train,
            mode="train",
        )

        val_dataset = DeepfakeClipDataset(
            data_root=args.data_root,
            metadata=val_meta_obj,
            clip_len=16,
            image_size=image_size,
            take_num=args.num_val,
            mode="test",
        )

    else:
        # ---- 2D: 原本 per-frame Dataset（Xception 用） ----
        print(f"🖼️ Using 2D Frame Dataset (Size={image_size})")
        train_dataset = DeepfakeDataset(
            data_root=args.data_root,
            json_file=args.train_metadata,
            image_size=image_size,
            take_num=args.num_train,
        )
        val_dataset = DeepfakeDataset(
            data_root=args.data_root,
            json_file=args.val_metadata,
            image_size=image_size,
            take_num=args.num_val,
        )

    # ===== 4. Trainer =====
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"./ckpt/test/{args.model}",
        save_last=True,
        filename=args.model + "-{epoch}-{val_loss:.3f}",
        monitor="val_loss",
        mode="min",
    )

    trainer = Trainer(
        precision=args.precision,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, LrLogger(), EarlyStoppingLR(lr_threshold=1e-7)],
        accelerator="gpu",
        devices=args.gpus,
        strategy="ddp" if args.gpus > 1 else "auto",
        log_every_n_steps=50,
    )

    # ===== 5. 開始訓練 =====
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=False,
        persistent_workers=True,
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
