import argparse
import torch
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from avdeepfake1m.loader import AVDeepfake1mPlusPlusImages as BaseDataset, Metadata
import os
from torch.utils.data import get_worker_info
from decord import VideoReader, cpu
import math
from xception import Xception
from utils import LrLogger, EarlyStoppingLR
import json

parser = argparse.ArgumentParser(description="Classification model training")
parser.add_argument("--data_root", type=str)
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
def read_video_decord(path: str, resize_shape: tuple = None):
    """
    高速讀取 + 解碼時縮放 (Decord version)
    """
    try:
        if resize_shape:
            h, w = resize_shape
            # 直接在 C++ 層級做 Resize，這是加速關鍵
            vr = VideoReader(path, ctx=cpu(0), width=w, height=h)
        else:
            vr = VideoReader(path, ctx=cpu(0))
        
        if len(vr) == 0:
            return torch.empty(0)

        # 批次讀取所有 frames
        video_data = vr.get_batch(range(len(vr))).asnumpy()
        
        # (T, H, W, C) -> (T, C, H, W)
        video = torch.from_numpy(video_data).permute(0, 3, 1, 2)
        return video.float() / 255.0
        
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return torch.empty(0)
class OwnAVDeepfake1mPlusPlusImages(BaseDataset):
    def __init__(self, json_file=None, *args, **kwargs):
        metadata_objs = None
        
        if json_file:
            print(f"Loading custom metadata from: {json_file}")
            # 1. 讀取自定義 JSON
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            metadata_objs = [Metadata(**item, fps=25) for item in data]
            
        super().__init__(metadata=metadata_objs, *args, **kwargs)
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            # 單線程：跑全部
            metadata_to_iter = self.metadata
        else:
            # 多線程：切分數據
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            per_worker = int(math.ceil(len(self.metadata) / num_workers))
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.metadata))
            metadata_to_iter = self.metadata[start:end]

        # ==============================
        # 2. 高速讀取迴圈 (取代 SDK 慢速邏輯)
        # ==============================
        
        # 預先定義目標尺寸 (96, 96)，讓解碼器直接縮放
        target_shape = (self.image_size, self.image_size)

        for meta in metadata_to_iter:
            # 拼湊路徑 (參考 SDK 邏輯)
            video_path = os.path.join(self.data_root, meta.split, meta.file)
            
            # --- 關鍵優化：改用 Decord 讀取並縮放 ---
            video = read_video_decord(video_path, resize_shape=target_shape)
            
            # 處理讀取失敗或空影片
            if video.numel() == 0:
                continue
            if self.use_video_label:
                label = float(len(meta.fake_periods) > 0)
                for frame in video:
                    yield frame, label
                    
            elif self.use_seg_label:
                # 為了效能，盡量少用 seg_label，這段運算在 Python 端稍重
                frame_label = torch.zeros(len(video))
                for begin, end in meta.fake_periods:
                    begin = int(begin * 25) # 假設 fps=25
                    end = int(end * 25)
                    frame_label[begin: end] = 1
                
                seg_label = torch.split(frame_label, self.use_seg_label)
                seg_label = torch.nn.utils.rnn.pad_sequence(seg_label, batch_first=True)
                seg_label = (seg_label.sum(dim=1) > 0).float().repeat_interleave(self.use_seg_label)
                
                for i, frame in enumerate(video):
                    yield frame, seg_label[i]
                    
            else:
                # 預設邏輯
                frame_label = torch.zeros(len(video))
                for begin, end in meta.fake_periods:
                    begin = int(begin * 25)
                    end = int(end * 25)
                    frame_label[begin: end] = 1
                
                for i, frame in enumerate(video):
                    yield frame, frame_label[i]

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

    train_dataset = OwnAVDeepfake1mPlusPlusImages(
        subset="train",
        data_root=args.data_root,
        take_num=args.num_train,
        use_video_label= False,
        json_file='/fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/train_metadata_filtered.json'

    )
    # For validation, you can still do the normal dataset
    val_dataset = OwnAVDeepfake1mPlusPlusImages(
        subset="val",
        data_root=args.data_root,
        take_num=args.num_val,
        use_video_label= False,
        json_file='/fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/validation_metadata_filtered.json'
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
