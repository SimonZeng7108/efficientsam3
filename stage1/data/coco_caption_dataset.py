import json
import os
import torch
from torch.utils.data import Dataset

class COCOCaptionDataset(Dataset):
    def __init__(self, data_root, split='train', num_samples=-1):
        self.data_root = data_root
        self.split = split
        self.anno_path = os.path.join(data_root, 'annotations', f'captions_{split}2017.json')
        
        if not os.path.exists(self.anno_path):
             # Fallback or error
             raise FileNotFoundError(f"Annotation file not found: {self.anno_path}")

        with open(self.anno_path, 'r') as f:
            self.anno_json = json.load(f)
            
        self.captions = []
        self.keys = []
        
        counter = 0
        for anno in self.anno_json['annotations']:
            self.captions.append(anno['caption'])
            # Use annotation id as key to be unique
            self.keys.append(str(anno['id']))
            
            counter += 1
            if num_samples > 0 and counter >= num_samples:
                break
            
    def __len__(self):
        return len(self.captions)
        
    def __getitem__(self, idx):
        return self.captions[idx]
        
    def get_keys(self):
        return self.keys
