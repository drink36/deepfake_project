import argparse
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from data.dataset import DeepfakeDataset
from xception import Xception
from utils import LrLogger, EarlyStoppingLR

parser = argparse.ArgumentParser(description="Classification model training")
parser.add_argument("--data_root", type=str)
parser.add_argument("--train_metadata", type=str, required=True, help="Path to the training metadata JSON file")
parser.add_argument("--val_metadata", type=str, required=True, help="Path to the validation metadata JSON file")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--model", type=str, choices=["xception", "meso4", "meso_inception4"])
parser.add_argument("--gpus", type=int, default=1)
parser.add_argument("--precision", default="bf16-mixed", choices=["16-mixed", "bf16-mixed", "32"])
parser.add_argument("--num_train", type=int, default=None)
parser.add_argument("--num_val", type=int, default=2000)
parser.add_argument("--max_epochs", type=int, default=500)
parser.add_argument("--resume", type=str, default=None)
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
    else:
        raise ValueError(f"Unknown model: {args.model}")

    train_dataset = DeepfakeDataset(
        data_root=args.data_root,
        json_file=args.train_metadata,
        image_size=96,
        take_num=args.num_train
        
    )
    # For validation, you can still do the normal dataset
    val_dataset = DeepfakeDataset(
        data_root=args.data_root,
        json_file=args.val_metadata,
        image_size=96,
        take_num=args.num_val
    )

    # Parse precision properly
    try:
        precision = int(args.precision)
    except ValueError:
        precision = args.precision

    monitor = "val_loss"

    trainer = Trainer(
        log_every_n_steps=50,
        precision=precision,
        max_epochs=args.max_epochs,
        callbacks=[
            ModelCheckpoint(
                dirpath=f"./ckpt/{args.model}",
                save_last=True,
                filename=args.model + "-{epoch}-{val_loss:.3f}",
                monitor=monitor,
                mode="min"
            ),
            LrLogger(),
            EarlyStoppingLR(lr_threshold=1e-7)
        ],
        enable_checkpointing=True,
        benchmark=True,
        accelerator="gpu",
        devices=args.gpus,
        strategy="ddp" if args.gpus > 1 else "auto",
        # ckpt_path=args.resume,
        # If you're on an older version of Lightning, you may need `strategy='ddp'` just the same, but this is typical.
    )
    num_workers = 8 * args.gpus
    trainer.fit(
        model,
        train_dataloaders=DataLoader(train_dataset, batch_size=args.batch_size, num_workers= 8,
                                        pin_memory=True, persistent_workers=True),
        val_dataloaders=DataLoader(val_dataset, batch_size=args.batch_size, num_workers= 8,
                                      pin_memory=True, persistent_workers=True),
        ckpt_path=args.resume,
    )
