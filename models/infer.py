import argparse
import torch
import json
import os
import sys
from tqdm.auto import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset import DeepfakeVideoDataset, VideoMetadata 
from xception import Xception
from videomae_v2 import DeepfakeVideoMAEV2 
from R2_1D import R2Plus1D
from datetime import datetime
parser = argparse.ArgumentParser(description="Deepfake inference")
parser.add_argument("--data_root", type=str)
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--model", type=str, choices=["xception", "videomae_v2", "r2plus1d"])
parser.add_argument("--batch_size", type=int, default=32) # GPU Batch Size
parser.add_argument("--subset", type=str, default="test") 
parser.add_argument("--gpus", type=int, default=1)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--metadata_file", type=str)
parser.add_argument("--metadata_txt", type=str, default=None)
parser.add_argument("--take_num", type=int, default=None)
parser.add_argument("--prob",type=bool,default=False)

def custom_collate(batch):
    videos = [item[0] for item in batch]
    filenames = [item[1] for item in batch]
    return videos, filenames

if __name__ == '__main__':
    args = parser.parse_args()
    device = "cuda" if args.gpus > 0 else "cpu"

    # === 1. 模型初始化 ===
    if args.model == "xception":
        print("🚀 Loading Xception...")
        model = Xception.load_from_checkpoint(args.checkpoint, lr=None, distributed=False).eval()
        image_size = 96
        is_3d_model = False
    elif args.model == "videomae_v2":
        print("🚀 Loading VideoMAE V2...")
        model = DeepfakeVideoMAEV2.load_from_checkpoint(
            args.checkpoint, lr=None, distributed=False).eval()
        image_size = 224 
        is_3d_model = True
    elif args.model == "r2plus1d":
        print("🚀 Loading R(2+1)D...")
        model = R2Plus1D.load_from_checkpoint(
            args.checkpoint, lr=None, distributed=False).eval()
        image_size = 112 
        is_3d_model = True
    else:
        raise ValueError(f"Unknown model: {args.model}")
    model.to(device)
    if args.model in ["videomae_v2", "r2plus1d"]:
        model.eval()
    else:
        model.train()
    # === 2. 準備數據 ===
    if args.metadata_file is not None:
        print(f"📄 Loading metadata from: {args.metadata_file}")
        with open(args.metadata_file, 'r') as f:
            data = json.load(f)
    elif args.metadata_txt is not None:
        print(f"📄 Loading metadata from: {args.metadata_txt}")
        data = []
        with open(args.metadata_txt, 'r') as f:
            for line in f:
                file_name = line.strip()
                data.append({"file": file_name, "split": args.subset})
    else:
        raise ValueError("Must provide either metadata_file or metadata_txt.")
    
    custom_metadata = [VideoMetadata(**item) for item in data]
    if args.take_num: custom_metadata = custom_metadata[:args.take_num]
    
    test_dataset = DeepfakeVideoDataset(
        data_root=args.data_root,
        metadata=custom_metadata,
        image_size=image_size
    )

    # Loader Batch Size: 一次讀 4 支影片進來 (CPU 平行處理)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8, # Linux 上開大一點，Windows 設 0
        pin_memory=True,
        persistent_workers=True,
        collate_fn=custom_collate
    )
    # add take num
    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = f"output/{args.model}_{args.subset}_{len(test_dataset)}_{date}.txt"
    save_path_soft = f"output/{args.model}_{args.subset}_{len(test_dataset)}_{date}_soft.txt"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    processed_files = set()
    if args.resume is not None and os.path.exists(args.resume):
        with open(args.resume, "r") as f:
            for line in f:
                processed_files.add(line.split(";")[0])

    print(f"🚀 Start Inference on {len(test_dataset)} files...")
    clip_len = 16
    inf_batch_size = args.batch_size # GPU Batch Size
    print(f"file_path: {save_path}")
    print(f"file_path_soft: {save_path_soft}")
    with open(save_path, "w") as f, open(save_path_soft,"w") as f_soft:
        with torch.inference_mode():
            for batch_videos, batch_filenames in tqdm(test_loader):
                
                # 因為 loader batch_size=1，這裡其實只有一支影片
                video = batch_videos[0] 
                file_name = batch_filenames[0]
                
                if file_name in processed_files: continue
                if video.numel() == 0: continue

                pred = 0.0
                pred_soft = 0.0

                if is_3d_model:
                    # === VideoMAE 優化邏輯 ===
                    video = video.float() / 255.0
                    video = video.to(device) # (T, C, H, W)
                    T = video.shape[0]

                    if T < clip_len:
                        padding = clip_len - T
                        last_frame = video[-1].unsqueeze(0)
                        video = torch.cat([video, last_frame.repeat(padding, 1, 1, 1)], dim=0)
                        T = clip_len

                    # 2) 先切「完整」的 clips（0,16,32,...）
                    n_full = T // clip_len
                    full_part = video[:n_full * clip_len]  # (n_full*16, C, H, W)

                    # (n_full, 16, C, H, W)
                    clips = full_part.reshape(n_full, clip_len, video.shape[1], video.shape[2], video.shape[3])

                    # 3) 如果有尾巴，加一個「最後16幀」clip（可能跟前一段重疊）
                    if T % clip_len != 0:
                        tail = video[-clip_len:]  # (16, C, H, W)
                        clips = torch.cat([clips, tail.unsqueeze(0)], dim=0)

                    # 4) -> (N, C, 16, H, W)
                    clips = clips.permute(0, 2, 1, 3, 4)

                    
                    n_clips = clips.shape[0]
                    # 4. 批次推論
                    all_logits = []
                    for k in range(0, n_clips, inf_batch_size):
                        batch_clips = clips[k : k + inf_batch_size]
                        logits = model(batch_clips)
                        all_logits.append(logits)
                    
                    if all_logits:
                        all_logits = torch.cat(all_logits, dim=0).flatten()
                        pred = all_logits.max().item()
                        pred_soft= torch.logsumexp(all_logits,dim=0)

                else:
                    # === Xception 優化邏輯 ===
                    video = video.float()

                    
                    all_logits = []
                    for k in range(0, len(video), inf_batch_size):
                        batch_frames = video[k: k + inf_batch_size].contiguous()  # CPU
                        batch_frames = batch_frames.to(device, non_blocking=True) # GPU (per batch)
                        logits = model(batch_frames)
                        all_logits.append(logits)
                        
                    if all_logits:
                        all_logits = torch.cat(all_logits, dim=0).flatten()
                        pred = all_logits.max().item()
                        pred_soft= torch.logsumexp(all_logits,dim=0)
                prob = torch.sigmoid(torch.tensor(pred)).item()
                prob_soft = torch.sigmoid(pred_soft).item()
                if args.prob:
                    f.write(f"{file_name};{prob}\n")
                    f_soft.write(f"{file_name};{prob_soft.item()}\n")
                else:
                    f.write(f"{file_name};{pred}\n")
                    f_soft.write(f"{file_name};{pred_soft.item()}\n")
                f.flush()