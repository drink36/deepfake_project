import os
import json
import math
import torch
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info, Dataset
from decord import VideoReader, cpu
from dataclasses import dataclass
from typing import List, Optional, Any
import random
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
    fps: float = 25.0
    extra_data: dict = None 
    def __init__(self, file: str, split: str = "train",**kwargs):
        self.file = file
        self.split = split
        self.extra_data = kwargs
        visual_fake_segments = kwargs.get("visual_fake_segments", None)
        if visual_fake_segments:
            self.fake_periods = visual_fake_segments
        else:
            self.fake_periods = []
        # 從 JSON 讀取這些欄位，如果沒有就用預設值
        self.video_frames = kwargs.get('video_frames', 0)
        self.duration = kwargs.get('duration', 0.0)
        
        self.fps = kwargs.get('fps', 25.0)

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
                    idx_begin = int(begin * meta.fps)   # 改這裡
                    idx_end = int(end * meta.fps)       # 改這裡
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
                    idx_begin = int(begin * meta.fps)   # 改
                    idx_end = int(end * meta.fps)       # 改
                    idx_begin = max(0, idx_begin)
                    idx_end = min(len(video), idx_end)
                    frame_label[idx_begin: idx_end] = 1
                
                for i, frame in enumerate(video):
                    yield frame, frame_label[i]

    def __len__(self):
        # === 關鍵修改：回傳精確的總幀數 ===
        return self.total_frames
class DeepfakeVideoDataset(Dataset):
    """
    專為 Inference 設計的 Dataset。
    一次回傳「整支影片」的所有 Frames，而不是打散的 Frame。
    """
    def __init__(
        self, 
        data_root: str, 
        json_file: str = None, 
        metadata: List[VideoMetadata] = None, # 支援直接傳入物件 (給 infer.py 用)
        image_size: int = 96,
        take_num: Optional[int] = None
    ):
        self.data_root = data_root
        self.image_size = image_size

        # 支援兩種初始化方式：讀 JSON 檔 或 直接傳 Metadata 列表
        if metadata:
            self.metadata = metadata
        elif json_file:
            print(f"Loading metadata from: {json_file}")
            with open(json_file, 'r') as f:
                data = json.load(f)
            self.metadata = [VideoMetadata(**item) for item in data]
        else:
            raise ValueError("Must provide either json_file or metadata.")

        if take_num is not None:
            self.metadata = self.metadata[:take_num]
            
        print(f"Loaded {len(self.metadata)} video entries for inference.")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        meta = self.metadata[index]
        # 拼湊路徑
        video_path = os.path.join(self.data_root, meta.split, meta.file)
        
        # 使用你原本寫好的 Decord 讀取函式
        # 注意：這裡回傳的是 (T, C, H, W) 的完整影片 Tensor
        video = read_video_decord(video_path, resize_shape=(self.image_size, self.image_size))
        
        return video, meta.file
class DeepfakeClipDataset(Dataset):
    def __init__(self, 
                 data_root: str, 
                 metadata: list,  # 你的 VideoMetadata list
                 clip_len: int = 32, # 3D CNN 通常用 8, 16, 32
                 frame_interval: int = 1, # 跳幀採樣 (1=連續, 2=每隔1張採1張)
                 image_size: int = 224, # VideoMAE/TimeSformer 通常要 224
                 take_num: Optional[int] = None,
                 mode: str = 'train'):
        
        self.data_root = data_root
        self.metadata = metadata
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.image_size = image_size
        self.mode = mode
        if take_num is not None:
            self.metadata = self.metadata[:take_num]
    def __len__(self):
        return len(self.metadata)

    def _get_clip_indices(self, total_frames, fake_periods, fps):
        """
        核心邏輯：決定要抓哪 16 幀
        """
        # 計算實際需要的總跨度 (考慮跳幀)
        # 例如 clip_len=16, interval=2 => 需要覆蓋 32 幀的長度
        window_size = (self.clip_len - 1) * self.frame_interval + 1
        
        if total_frames < window_size:
            # 影片太短，回傳全部並 Padding (或直接頭部重複)
            return list(range(total_frames)) + [total_frames-1] * (self.clip_len - total_frames)

        if self.mode == 'train':
            # 如果有 fake_periods，優先從裡面抓
            if len(fake_periods) > 0:
                # 1. 把秒數轉為 Frame Index 區間
                fake_ranges = []
                for start_sec, end_sec in fake_periods:
                    s_idx = int(start_sec * fps)
                    e_idx = int(end_sec * fps)
                    # 確保區間合理
                    if e_idx - s_idx >= window_size:
                        fake_ranges.append((s_idx, e_idx))
                
                # 2. 如果有夠長的 Fake 片段，從裡面隨機選一段
                if fake_ranges:
                    s, e = random.choice(fake_ranges)
                    # 在這個 Fake 區間內，隨機找一個起點
                    max_start = e - window_size
                    start_idx = random.randint(s, max(s, max_start))
                    
                    # 產生 Indices
                    return [start_idx + i * self.frame_interval for i in range(self.clip_len)]
            
            max_start = total_frames - window_size
            start_idx = random.randint(0, max_start)
            return [start_idx + i * self.frame_interval for i in range(self.clip_len)]

        else:
            # 取正中間的一段 (或是你可以改成取多段做 Ensemble)
            start_idx = max(0, (total_frames - window_size) // 2)
            return [start_idx + i * self.frame_interval for i in range(self.clip_len)]

    def __getitem__(self, index):
        meta = self.metadata[index]
        path = os.path.join(self.data_root, meta.split, meta.file)
        
        try:
            vr = VideoReader(path, ctx=cpu(0), width=self.image_size, height=self.image_size)
            total_frames = len(vr)
            

            indices = self._get_clip_indices(total_frames, meta.fake_periods, meta.fps)
            

            buffer = vr.get_batch(indices).asnumpy()
            

            video = torch.from_numpy(buffer).permute(3, 0, 1, 2).float() / 255.0
            

            label = 1.0 if len(meta.fake_periods) > 0 else 0.0
            
            return video, label, meta.file
            
        except Exception as e:
            print(f"Error reading {path}: {e}")
            # 回傳空值，DataLoader 的 collate_fn 需處理
            return torch.zeros(3, self.clip_len, self.image_size, self.image_size), 0.0, meta.file