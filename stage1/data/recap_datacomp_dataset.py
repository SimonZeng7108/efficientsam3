import os
import glob
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

class RecapDataCompDataset(Dataset):
    def __init__(self, data_root, split='train', num_samples=-1):
        self.data_root = data_root
        self.split = split
        
        # Look for parquet files in data_root or data_root/recap_subset
        # The download script saves to data/recap_subset
        # But the user might point DATA_PATH to data/recap_subset directly
        
        search_paths = [
            os.path.join(data_root, "*.parquet"),
            os.path.join(data_root, "recap_subset", "*.parquet"),
            os.path.join(data_root, "data", "*.parquet"),
            os.path.join(data_root, "data", "train_data", "*.parquet"), # Added for specific structure
            os.path.join(data_root, "**", "*.parquet"), # Recursive search as fallback
        ]
        
        self.parquet_files = []
        for path in search_paths:
            found = glob.glob(path, recursive=True)
            if found:
                self.parquet_files.extend(found)
        
        # Remove duplicates if any
        self.parquet_files = sorted(list(set(self.parquet_files)))
        
        if not self.parquet_files:
            raise FileNotFoundError(f"No parquet files found in {data_root} or subdirectories.")

        print(f"Found {len(self.parquet_files)} parquet files. Loading Recap-DataComp dataset...")
        
        dfs = []
        for p_file in tqdm(self.parquet_files, desc="Loading Parquet Files"):
            try:
                df = pd.read_parquet(p_file)
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {p_file}: {e}")
        
        if not dfs:
             raise RuntimeError("Failed to load any parquet files.")

        self.df = pd.concat(dfs, ignore_index=True)
        
        if num_samples > 0:
            self.df = self.df.iloc[:num_samples]
            
        # Check for columns
        if 're_caption' in self.df.columns:
            self.captions = self.df['re_caption'].tolist()
        elif 'recaption' in self.df.columns:
            self.captions = self.df['recaption'].tolist()
        elif 'text' in self.df.columns:
            self.captions = self.df['text'].tolist()
        elif 'caption' in self.df.columns:
            self.captions = self.df['caption'].tolist()
        else:
            raise ValueError(f"Could not find caption column. Available columns: {self.df.columns}")
            
        # Use uid or image_id or sha256 as key
        if 'key' in self.df.columns:
            self.keys = self.df['key'].astype(str).tolist()
        elif 'uid' in self.df.columns:
            self.keys = self.df['uid'].astype(str).tolist()
        elif 'image_id' in self.df.columns:
            self.keys = self.df['image_id'].astype(str).tolist()
        elif 'sha256' in self.df.columns:
            self.keys = self.df['sha256'].astype(str).tolist()
        else:
            # Fallback to index if no ID found
            print("Warning: No unique ID column found (uid, image_id, sha256). Using index.")
            self.keys = [str(i) for i in range(len(self.captions))]
            
        print(f"Loaded {len(self.captions)} captions.")

    def __len__(self):
        return len(self.captions)
        
    def __getitem__(self, idx):
        return self.captions[idx]
        
    def get_keys(self):
        return self.keys
