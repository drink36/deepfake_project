import os
import json
import math
import torch
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info
from decord import VideoReader, cpu
from dataclasses import dataclass
from typing import List, Optional, Any

@dataclass
class VideoMetadata:
    """
    整合後的 Metadata，強制讀取 video_frames 以計算總長度。
    """
    file: str
    split: str
    fake_periods: List[List[float]]
    duration: float = 0.0
    video_frames: int = 0
    extra_data: dict = None 

    def __init__(self, file: str, split: str = "train", fake_periods: List[List[float]] = None, **kwargs):
        self.file = file
        self.split = split
        self.fake_periods = fake_periods if fake_periods is not None else []
        self.extra_data = kwargs
        
        # 這裡很關鍵：嘗試從 JSON 讀取 video_frames
        # 如果 JSON 裡沒寫，預設為 0 (這樣總長度會算錯，但不會報錯)
        self.video_frames = kwargs.get('video_frames', 0)
        self.duration = kwargs.get('duration', 0.0)

def read_video_decord(path: str, resize_shape: tuple = None):
    try:
        vr = VideoReader(path, ctx=cpu(0))
        if resize_shape:
            h, w = resize_shape
            vr = VideoReader(path, ctx=cpu(0), width=w, height=h)
        
        if len(vr) == 0:
            return torch.empty(0)

        video_data = vr.get_batch(range(len(vr))).asnumpy()
        video = torch.from_numpy(video_data).permute(0, 3, 1, 2)
        return video.float() / 255.0
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return torch.empty(0)

class DeepfakeDataset(IterableDataset):
    def __init__(
        self, 
        data_root: str, 
        json_file: str, 
        image_size: int, 
        fps: int = 25,
        use_video_label: bool = False, 
        use_seg_label: Optional[int] = None, 
        take_num: Optional[int] = None
    ):
        self.data_root = data_root
        self.image_size = image_size
        self.fps = fps
        self.use_video_label = use_video_label
        self.use_seg_label = use_seg_label

        print(f"Loading metadata from: {json_file}")
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # 建立 Metadata
        self.metadata = [VideoMetadata(**item) for item in data]

        if take_num is not None:
            self.metadata = self.metadata[:take_num]
        
        # === 關鍵修改：計算總幀數 ===
        # 這樣你的 tqdm 進度條就會是準確的
        self.total_frames = sum([meta.video_frames for meta in self.metadata])
        
        # 如果發現總幀數是 0，代表 JSON 裡可能沒有 video_frames 欄位，印個警告
        if self.total_frames == 0 and len(self.metadata) > 0:
            print("Warning: Total frames is 0. Check if 'video_frames' field exists in your JSON.")

        print(f"Loaded {len(self.metadata)} video entries with {self.total_frames} total frames.")

    def __iter__(self):
        # 1. 多線程分工
        worker_info = get_worker_info()
        if worker_info is None:
            metadata_to_iter = self.metadata
        else:
            per_worker = int(math.ceil(len(self.metadata) / worker_info.num_workers))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.metadata))
            metadata_to_iter = self.metadata[start:end]

        target_shape = (self.image_size, self.image_size)

        # 2. 讀取數據
        for meta in metadata_to_iter:
            video_path = os.path.join(self.data_root, meta.split, meta.file)
            video = read_video_decord(video_path, resize_shape=target_shape)
            
            if video.numel() == 0:
                continue

            # 3. 標籤邏輯
            if self.use_video_label:
                label = float(len(meta.fake_periods) > 0)
                for frame in video:
                    yield frame, label

            elif self.use_seg_label:
                frame_label = torch.zeros(len(video))
                for begin, end in meta.fake_periods:
                    idx_begin = int(begin * self.fps)
                    idx_end = int(end * self.fps)
                    frame_label[idx_begin: idx_end] = 1
                
                seg_chunks = torch.split(frame_label, self.use_seg_label)
                seg_padded = torch.nn.utils.rnn.pad_sequence(list(seg_chunks), batch_first=True)
                seg_result = (seg_padded.sum(dim=1) > 0).float()
                
                seg_expanded = seg_result.repeat_interleave(self.use_seg_label)
                seg_expanded = seg_expanded[:len(video)]
                
                for i, frame in enumerate(video):
                    yield frame, seg_expanded[i]

            else:
                frame_label = torch.zeros(len(video))
                for begin, end in meta.fake_periods:
                    idx_begin = int(begin * self.fps)
                    idx_end = int(end * self.fps)
                    idx_begin = max(0, idx_begin)
                    idx_end = min(len(video), idx_end)
                    frame_label[idx_begin: idx_end] = 1
                
                for i, frame in enumerate(video):
                    yield frame, frame_label[i]

    def __len__(self):
        # === 關鍵修改：回傳精確的總幀數 ===
        return self.total_frames