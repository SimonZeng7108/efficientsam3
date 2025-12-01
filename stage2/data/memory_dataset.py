import os
import random
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class Stage2SequenceDataset(Dataset):
    """Loads video segments (features + teacher targets) for Stage2 training.
    
    Loads only a random window of 'seq_len' frames to save I/O and memory.
    """

    def __init__(
        self,
        features_root: str,
        teacher_root: str,
        seq_len: int = 24,
        max_frames: int = -1,
    ):
        assert os.path.isdir(features_root), f"{features_root} missing"
        assert os.path.isdir(teacher_root), f"{teacher_root} missing"
        self.features_root = features_root
        self.teacher_root = teacher_root
        self.seq_len = seq_len
        self.max_frames = max_frames

        feature_videos = sorted(os.listdir(features_root))
        teacher_videos = set(os.listdir(teacher_root))
        self.videos = [v for v in feature_videos if v in teacher_videos]
        if not self.videos:
            raise RuntimeError(
                "No overlapping videos between features and teacher embeddings."
            )

    def __len__(self) -> int:
        return len(self.videos)

    def _load_frames(self, root: str, video: str) -> List[str]:
        video_dir = os.path.join(root, video)
        frames = sorted(
            [
                f
                for f in os.listdir(video_dir)
                if f.endswith(".pt") and os.path.isfile(os.path.join(video_dir, f))
            ]
        )
        if self.max_frames > 0:
            frames = frames[: self.max_frames]
        return frames

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video = self.videos[idx]
        feature_frames = self._load_frames(self.features_root, video)
        teacher_frames = self._load_frames(self.teacher_root, video)
        
        # Ensure alignment
        frame_count = min(len(feature_frames), len(teacher_frames))
        feature_frames = feature_frames[:frame_count]
        teacher_frames = teacher_frames[:frame_count]
        
        # Random Crop Window
        if frame_count > self.seq_len:
            start_idx = random.randint(0, frame_count - self.seq_len)
            end_idx = start_idx + self.seq_len
            feature_frames = feature_frames[start_idx:end_idx]
            teacher_frames = teacher_frames[start_idx:end_idx]
        
        # Load only the selected frames
        def load_feature(frame_name: str) -> Dict[str, torch.Tensor]:
            return torch.load(
                os.path.join(self.features_root, video, frame_name), map_location="cpu"
            )

        def load_teacher(frame_name: str) -> Dict[str, torch.Tensor]:
            return torch.load(
                os.path.join(self.teacher_root, video, frame_name), map_location="cpu"
            )

        feature_data = [load_feature(fname) for fname in feature_frames]
        teacher_data = [load_teacher(fname) for fname in teacher_frames]

        def stack(key: str) -> torch.Tensor:
            tensors = [frame[key].float() for frame in feature_data]
            return torch.stack(tensors, dim=0)

        sample = {
            "video": video,
            "f0": stack("f0"),
            "f1": stack("f1"),
            "f2": stack("f2"),
            "p0": stack("p0"),
            "p1": stack("p1"),
            "p2": stack("p2"),
            "teacher_feat": torch.stack(
                [frame["teacher_feat"].float() for frame in teacher_data], dim=0
            ),
            "mask_high_res": torch.stack(
                [frame["mask_high_res"].float() for frame in teacher_data], dim=0
            ),
            "object_scores": torch.stack(
                [frame["object_scores"].float() for frame in teacher_data], dim=0
            ),
            "is_init": torch.tensor(
                [bool(frame["is_init"]) for frame in teacher_data], dtype=torch.bool
            ),
        }
        return sample
