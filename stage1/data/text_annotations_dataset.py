import os
import json
from torch.utils.data import Dataset


class TextAnnotationsDataset(Dataset):
    """
    Dataset for loading text annotations from the combined JSON file.
    Used for text encoder distillation in Stage 1.
    
    Expected JSON structure:
    {
        "info": {...},
        "text_annotations": ["annotation1", "annotation2", ...]
    }
    """
    
    def __init__(self, data_root, split='train', num_samples=-1):
        self.data_root = data_root
        self.split = split
        
        # Look for the combined text annotations JSON
        # Support multiple possible paths
        search_paths = [
            os.path.join(data_root, "text_annotations_combined.json"),
            os.path.join(data_root, "data", "text_annotations_combined.json"),
            data_root,  # If data_root is the full path to the JSON file
        ]
        
        json_file = None
        for path in search_paths:
            if os.path.isfile(path) and path.endswith('.json'):
                json_file = path
                break
            elif os.path.isdir(path):
                candidate = os.path.join(path, "text_annotations_combined.json")
                if os.path.isfile(candidate):
                    json_file = candidate
                    break
        
        if json_file is None:
            raise FileNotFoundError(
                f"Could not find text_annotations_combined.json in {data_root} or subdirectories. "
                f"Searched: {search_paths}"
            )
        
        print(f"Loading text annotations from: {json_file}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'text_annotations' not in data:
            raise ValueError(f"JSON file missing 'text_annotations' key. Found keys: {data.keys()}")
        
        self.text_annotations = data['text_annotations']
        
        # Apply num_samples limit if specified
        if num_samples > 0:
            self.text_annotations = self.text_annotations[:num_samples]
        
        # Generate keys (using index as unique identifier)
        self.keys = [f"text_{i}" for i in range(len(self.text_annotations))]
        
        print(f"Loaded {len(self.text_annotations)} text annotations.")
        
        # Show some examples
        if len(self.text_annotations) > 0:
            print("Sample annotations:")
            for i, ann in enumerate(self.text_annotations[:5]):
                print(f"  {i+1}. {ann}")
            if len(self.text_annotations) > 5:
                print(f"  ... and {len(self.text_annotations) - 5} more")

    def __len__(self):
        return len(self.text_annotations)
    
    def __getitem__(self, idx):
        return self.text_annotations[idx]
    
    def get_keys(self):
        return self.keys
