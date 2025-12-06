import os
import json
import math
import torch
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info
from decord import VideoReader, cpu
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class VideoMetadata:
    """
    簡化的 Metadata 類別，只保留訓練需要的欄位。
    """
    file: str
    split: str
    fake_periods: List[List[float]]  # 單位：秒
    fps: int = 25

def read_video_decord(path: str, resize_shape: tuple = None, device='cpu'):
    """
    使用 Decord 高速讀取影片，並支援在解碼時直接 Resize。
    """
    try:
        # Decord 的 cpu(0) 對於多線程 DataLoader 比較友善
        ctx = cpu(0) 
        
        if resize_shape:
            h, w = resize_shape
            # 直接在 C++ 層級做 Resize，這是加速關鍵
            vr = VideoReader(path, ctx=ctx, width=w, height=h)
        else:
            vr = VideoReader(path, ctx=ctx)
        
        if len(vr) == 0:
            return torch.empty(0)

        # 批次讀取所有 frames
        video_data = vr.get_batch(range(len(vr))).asnumpy()
        
        # (T, H, W, C) -> (T, C, H, W) 且正規化到 [0, 1]
        video = torch.from_numpy(video_data).permute(0, 3, 1, 2)
        return video.float() / 255.0
        
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return torch.empty(0)

class DeepfakeDataset(IterableDataset):
    def __init__(
        self, 
        data_root: str, 
        metadata_file: str, 
        image_size: int = 96,
        fps: int = 25
    ):
        """
        Args:
            data_root (str): 數據集根目錄。
            metadata_file (str): JSON 標註檔案路徑。
            image_size (int): 讀取影片時的目標尺寸 (H, W)。
            fps (int): 影片的 FPS，用於將秒數轉換為幀數索引。
        """
        super().__init__()
        self.data_root = data_root
        self.image_size = image_size
        self.fps = fps
        
        # 讀取並解析 JSON
        print(f"Loading metadata from: {metadata_file}")
        with open(metadata_file, 'r') as f:
            data = json.load(f)
        
        # 將字典轉換為物件，方便後續存取
        self.metadata = [
            VideoMetadata(
                file=item['file'],
                split=item.get('split', 'train'), # 預設為 train，如果 JSON 沒寫
                fake_periods=item.get('fake_periods', []),
                fps=fps
            ) 
            for item in data
        ]
        print(f"Loaded {len(self.metadata)} video entries.")

    def __iter__(self):
        """
        IterableDataset 的核心邏輯。
        會自動偵測是否在多 Worker 環境下運行，並切分數據。
        """
        worker_info = get_worker_info()
        
        if worker_info is None:
            # 單線程：跑全部數據
            metadata_to_iter = self.metadata
        else:
            # 多線程：根據 worker_id 切分數據
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            per_worker = int(math.ceil(len(self.metadata) / num_workers))
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.metadata))
            metadata_to_iter = self.metadata[start:end]

        # 預先定義目標尺寸
        target_shape = (self.image_size, self.image_size)

        for meta in metadata_to_iter:
            # 拼湊完整影片路徑: data_root/split/filename
            # 注意：這裡假設 JSON 中的 file 只是檔名，split 是子資料夾
            video_path = os.path.join(self.data_root, meta.split, meta.file)
            
            # 使用 Decord 讀取
            video = read_video_decord(video_path, resize_shape=target_shape)
            
            # 處理讀取失敗或空影片
            if video.numel() == 0:
                continue
            
            # === 生成標籤 (Frame-level Label) ===
            # 預設邏輯：全 0 (真實)，如果在 fake_periods 內則設為 1 (偽造)
            frame_label = torch.zeros(len(video))
            
            for begin, end in meta.fake_periods:
                # 將秒數轉換為 Frame Index
                idx_begin = int(begin * self.fps)
                idx_end = int(end * self.fps)
                # 防止索引越界 (雖然通常不會，但保險起見)
                idx_begin = max(0, idx_begin)
                idx_end = min(len(video), idx_end)
                
                frame_label[idx_begin:idx_end] = 1.0
            
            # 逐幀回傳 (Frame, Label)
            # 這樣 DataLoader 會幫你把多個 Frame 組成一個 Batch
            for i in range(len(video)):
                yield video[i], frame_label[i]

    def __len__(self):
        # 估計的長度，不一定精確，但對 tqdm 進度條有幫助
        # 假設平均每支影片 100 幀 (可根據實際情況調整)
        return len(self.metadata) * 100