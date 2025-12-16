import os
import json
import math
import torch
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info, Dataset
from decord import VideoReader, cpu
from dataclasses import dataclass
from typing import List, Optional
import random
@dataclass
class VideoMetadata:
    file: str
    split: str
    fake_periods: List[List[float]]
    duration: float = 0.0
    video_frames: int = 0
    fps: float = 25.0
    extra_data: dict = None 
    def __init__(self, file: str, split: str = "",**kwargs):
        self.file = file
        self.split = split
        self.extra_data = kwargs
        visual_fake_segments = kwargs.get("visual_fake_segments", None)
        if visual_fake_segments:
            self.fake_periods = visual_fake_segments
        else:
            self.fake_periods = []
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
        return video
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
        self.metadata = [VideoMetadata(**item) for item in data]

        if take_num is not None:
            self.metadata = self.metadata[:take_num]
        
        self.total_frames = sum([meta.video_frames for meta in self.metadata])
        
        if self.total_frames == 0 and len(self.metadata) > 0:
            print("Warning: Total frames is 0. Check if 'video_frames' field exists in your JSON.")

        print(f"Loaded {len(self.metadata)} video entries with {self.total_frames} total frames.")

    def __iter__(self):
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

        
        for meta in metadata_to_iter:
            video_path = os.path.join(self.data_root, meta.split, meta.file)
            video = read_video_decord(video_path, resize_shape=target_shape)

            if video.numel() == 0:
                continue

            if video.dtype == torch.uint8:
                video = video.float().div_(255.0)
            else:
                video = video.float()

            video = (video - 0.5) / 0.5

            if self.use_video_label:
                label = float(len(meta.fake_periods) > 0)
                for frame in video:
                    yield frame, label

            elif self.use_seg_label:
                frame_label = torch.zeros(len(video))
                for begin, end in meta.fake_periods:
                    idx_begin = int(begin * meta.fps)   
                    idx_end = int(end * meta.fps)     
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
                    idx_begin = int(begin * meta.fps)   
                    idx_end = int(end * meta.fps)      
                    idx_begin = max(0, idx_begin)
                    idx_end = min(len(video), idx_end)
                    frame_label[idx_begin: idx_end] = 1
                
                for i, frame in enumerate(video):
                    yield frame, frame_label[i]

    def __len__(self):
        return self.total_frames
class DeepfakeVideoDataset(Dataset):
    def __init__(
        self, 
        data_root: str, 
        json_file: str = None, 
        metadata: List[VideoMetadata] = None,
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
        video_path = os.path.join(self.data_root, meta.split, meta.file)
        
        # (T, C, H, W)
        video = read_video_decord(video_path, resize_shape=(self.image_size, self.image_size))
        
        return video, meta.file
class DeepfakeClipDataset(Dataset):
    def __init__(self, 
                 data_root: str, 
                 metadata: list,
                 clip_len: int = 16, 
                 frame_interval: int = 1, 
                 image_size: int = 224, 
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
        window_size = (self.clip_len - 1) * self.frame_interval + 1

        # ---------------------------
        # CASE 1: Fake video
        # ---------------------------
        if len(fake_periods) > 0:
            start_sec, end_sec = random.choice(fake_periods)

            s_idx = int(start_sec * fps)
            e_idx = int(end_sec * fps)

            # Fake center
            center = (s_idx + e_idx) // 2
            center = max(0, min(center, total_frames - 1))

            half = window_size // 2

            if self.mode == "train":
                # Random jitter during training
                jitter = random.randint(-half // 2, half // 2)
            else:
                # No randomness in val/test
                jitter = 0

            start_idx = center - half + jitter

            # Clamp
            start_idx = max(0, min(start_idx, total_frames - window_size))

            return [start_idx + i * self.frame_interval for i in range(self.clip_len)]

        # ---------------------------
        # CASE 2: Real video → center sampling
        # ---------------------------
        center = total_frames // 2
        half = window_size // 2
        start_idx = max(0, center - half)
        start_idx = min(start_idx, total_frames - window_size)

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
            return torch.zeros(3, self.clip_len, self.image_size, self.image_size), 0.0, meta.file