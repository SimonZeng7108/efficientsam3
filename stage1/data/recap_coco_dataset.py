import os
import pandas as pd
from torch.utils.data import Dataset

class RecapCOCODataset(Dataset):
    def __init__(self, data_root, split='train', num_samples=-1):
        self.data_root = data_root
        self.split = split
        # We assume the parquet file is at data_root/new_data.parquet
        # or data_root/recap_coco/new_data.parquet depending on config
        
        self.parquet_path = os.path.join(data_root, 'new_data.parquet')
        if not os.path.exists(self.parquet_path):
             # Try subfolder
             self.parquet_path = os.path.join(data_root, 'recap_coco', 'new_data.parquet')
        
        if not os.path.exists(self.parquet_path):
             raise FileNotFoundError(f"Parquet file not found at {self.parquet_path}")

        print(f"Loading Recap-COCO dataset from {self.parquet_path}...")
        self.df = pd.read_parquet(self.parquet_path)
        
        if num_samples > 0:
            self.df = self.df.iloc[:num_samples]
            
        self.captions = self.df['recaption'].tolist()
        # Fallback to caption if recaption is missing/empty? 
        # Inspection showed recaption exists.
        
        # Use image_id as key
        self.keys = self.df['image_id'].astype(str).tolist()
            
    def __len__(self):
        return len(self.captions)
        
    def __getitem__(self, idx):
        return self.captions[idx]
        
    def get_keys(self):
        return self.keys
