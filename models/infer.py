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

parser = argparse.ArgumentParser(description="Xception inference")
parser.add_argument("--data_root", type=str)
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--model", type=str)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--subset", type=str, default="test") 
parser.add_argument("--gpus", type=int, default=1)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--metadata_file", type=str, required=True, help="Path to the custom metadata JSON file")

if __name__ == '__main__':
    args = parser.parse_args()
    use_gpu = args.gpus > 0
    device = "cuda" if use_gpu else "cpu"

    if args.model == "xception":
        model = Xception.load_from_checkpoint(args.checkpoint, lr=None, distributed=False).eval()
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model.to(device)
    model.eval()
    
    print(f"📄 Loading metadata from: {args.metadata_file}")
    with open(args.metadata_file, 'r') as f:
        data = json.load(f)
    
    custom_metadata = [VideoMetadata(**item) for item in data]
    
    test_dataset = DeepfakeVideoDataset(
        data_root=args.data_root,
        metadata=custom_metadata,
        image_size=96
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=1, 
        shuffle=False,
        num_workers=8,
        collate_fn=None, 
        pin_memory=True 
    )

    save_path = f"output/{args.model}_{args.subset}.txt"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    processed_files = set()
    if args.resume is not None and os.path.exists(args.resume):
        with open(args.resume, "r") as f:
            for line in f:
                processed_files.add(line.split(";")[0])

    print(f"🚀 Start Inference on {len(test_dataset)} files with DataLoader (workers=4)...")
    
    with open(save_path, "a") as f:
        with torch.inference_mode():

            for batch in tqdm(test_loader):

                videos, file_names = batch
                

                video = videos[0]           
                file_name = file_names[0]   

                if file_name in processed_files:
                    continue

                if video.numel() == 0:
                    print(f"⚠️ Warning: Empty video {file_name}")
                    continue

                preds_video = []

                for j in range(0, len(video), args.batch_size):
                    frame_batch = video[j:j + args.batch_size].to(device)
                    preds_video.append(model(frame_batch))

                preds_video = torch.cat(preds_video, dim=0).flatten()
                pred = preds_video.max().item()

                f.write(f"{file_name};{pred}\n")
                f.flush()