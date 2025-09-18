"""
Data loader for gaze estimation datasets.

Supports loading and preprocessing of MPIIGaze, GazeCapture and other datasets.
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict, List, Any
import h5py
import json
from pathlib import Path

from .utils import preprocess_eye_region


class GazeDataset(Dataset):
    """
    Generic gaze dataset class that can handle multiple dataset formats.
    """
    
    def __init__(self, data_path: str, dataset_type: str = 'mpiigaze', 
                 transform=None, split: str = 'train'):
        """
        Initialize gaze dataset.
        
        Args:
            data_path: Path to dataset directory
            dataset_type: Type of dataset ('mpiigaze', 'gazecapture', 'custom')
            transform: Optional data transforms
            split: Dataset split ('train', 'val', 'test')
        """
        self.data_path = Path(data_path)
        self.dataset_type = dataset_type
        self.transform = transform
        self.split = split
        
        # Load dataset metadata
        self.samples = self._load_dataset()
        
        print(f"Loaded {len(self.samples)} samples from {dataset_type} dataset ({split} split)")
    
    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load dataset based on type."""
        if self.dataset_type == 'mpiigaze':
            return self._load_mpiigaze()
        elif self.dataset_type == 'gazecapture':
            return self._load_gazecapture()
        elif self.dataset_type == 'custom':
            return self._load_custom()
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
    
    def _load_mpiigaze(self) -> List[Dict[str, Any]]:
        """Load MPIIGaze dataset."""
        samples = []
        
        # MPIIGaze structure: person_id/day_id/image_files + annotations
        annotation_file = self.data_path / f"{self.split}_annotations.txt"
        
        if annotation_file.exists():
            # Load from annotation file
            with open(annotation_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        image_path = self.data_path / parts[0]
                        yaw = float(parts[1])
                        pitch = float(parts[2])
                        # Additional metadata if available
                        person_id = parts[3] if len(parts) > 3 else "unknown"
                        
                        if image_path.exists():
                            samples.append({
                                'image_path': str(image_path),
                                'yaw': yaw,
                                'pitch': pitch,
                                'person_id': person_id
                            })
        else:
            # Fallback: scan directory structure
            print(f"Warning: Annotation file {annotation_file} not found. Scanning directory...")
            samples = self._scan_mpiigaze_directory()
        
        return samples
    
    def _scan_mpiigaze_directory(self) -> List[Dict[str, Any]]:
        """Scan MPIIGaze directory structure."""
        samples = []
        
        # Look for .h5 files (common MPIIGaze format)
        h5_files = list(self.data_path.glob("**/*.h5"))
        
        for h5_file in h5_files:
            try:
                with h5py.File(h5_file, 'r') as f:
                    if 'image' in f and 'gaze' in f:
                        images = f['image'][:]
                        gazes = f['gaze'][:]
                        
                        for i, (image, gaze) in enumerate(zip(images, gazes)):
                            samples.append({
                                'image_data': image,
                                'yaw': float(gaze[0]),
                                'pitch': float(gaze[1]),
                                'person_id': h5_file.stem,
                                'frame_id': i
                            })
            except Exception as e:
                print(f"Error loading {h5_file}: {e}")
        
        return samples
    
    def _load_gazecapture(self) -> List[Dict[str, Any]]:
        """Load GazeCapture dataset."""
        samples = []
        
        # GazeCapture typically has a metadata file
        metadata_file = self.data_path / f"{self.split}_metadata.json"
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            for item in metadata:
                image_path = self.data_path / item['image_path']
                if image_path.exists():
                    samples.append({
                        'image_path': str(image_path),
                        'yaw': float(item['gaze_yaw']),
                        'pitch': float(item['gaze_pitch']),
                        'person_id': item.get('person_id', 'unknown'),
                        'device_id': item.get('device_id', 'unknown')
                    })
        else:
            print(f"Warning: Metadata file {metadata_file} not found")
        
        return samples
    
    def _load_custom(self) -> List[Dict[str, Any]]:
        """Load custom dataset format."""
        samples = []
        
        # Look for CSV annotation file
        csv_file = self.data_path / f"{self.split}.csv"
        
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            
            for _, row in df.iterrows():
                image_path = self.data_path / row['image_path']
                if image_path.exists():
                    samples.append({
                        'image_path': str(image_path),
                        'yaw': float(row['yaw']),
                        'pitch': float(row['pitch']),
                        'person_id': row.get('person_id', 'unknown')
                    })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        sample = self.samples[idx]
        
        # Load image
        if 'image_data' in sample:
            # Image data is already loaded (e.g., from HDF5)
            image = sample['image_data']
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
        else:
            # Load image from file
            image_path = sample['image_path']
            image = cv2.imread(image_path)
            if image is None:
                # Return zero tensor if image loading fails
                image = np.zeros((60, 36, 3), dtype=np.uint8)
        
        # Preprocess image
        image_tensor = preprocess_eye_region(image)
        
        # Get gaze target
        gaze_target = torch.tensor([sample['yaw'], sample['pitch']], dtype=torch.float32)
        
        # Apply transforms if specified
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        return image_tensor, gaze_target
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get additional information about a sample."""
        return self.samples[idx]


class GazeDataLoader:
    """
    Data loader manager for gaze estimation datasets.
    """
    
    def __init__(self, data_config: Dict[str, Any]):
        """
        Initialize data loader.
        
        Args:
            data_config: Configuration dictionary containing dataset parameters
        """
        self.data_config = data_config
        self.datasets = {}
        self.data_loaders = {}
    
    def setup_datasets(self) -> None:
        """Setup datasets for training, validation, and testing."""
        for split in ['train', 'val', 'test']:
            if split in self.data_config:
                config = self.data_config[split]
                
                dataset = GazeDataset(
                    data_path=config['data_path'],
                    dataset_type=config.get('dataset_type', 'mpiigaze'),
                    split=split
                )
                
                self.datasets[split] = dataset
                
                # Create data loader
                loader = DataLoader(
                    dataset,
                    batch_size=config.get('batch_size', 32),
                    shuffle=(split == 'train'),
                    num_workers=config.get('num_workers', 4),
                    pin_memory=config.get('pin_memory', True)
                )
                
                self.data_loaders[split] = loader
    
    def get_data_loader(self, split: str) -> Optional[DataLoader]:
        """Get data loader for specified split."""
        return self.data_loaders.get(split)
    
    def get_dataset_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics about the datasets."""
        stats = {}
        
        for split, dataset in self.datasets.items():
            yaw_values = []
            pitch_values = []
            
            for sample in dataset.samples:
                yaw_values.append(sample['yaw'])
                pitch_values.append(sample['pitch'])
            
            stats[split] = {
                'num_samples': len(dataset),
                'yaw_mean': np.mean(yaw_values),
                'yaw_std': np.std(yaw_values),
                'pitch_mean': np.mean(pitch_values),
                'pitch_std': np.std(pitch_values)
            }
        
        return stats


class MultiDatasetLoader:
    """
    Loader that can handle multiple datasets simultaneously.
    """
    
    def __init__(self, dataset_configs: List[Dict[str, Any]]):
        """
        Initialize multi-dataset loader.
        
        Args:
            dataset_configs: List of dataset configurations
        """
        self.dataset_configs = dataset_configs
        self.datasets = []
        self.combined_loader = None
    
    def setup_datasets(self) -> None:
        """Setup multiple datasets."""
        all_samples = []
        
        for config in self.dataset_configs:
            dataset = GazeDataset(
                data_path=config['data_path'],
                dataset_type=config['dataset_type'],
                split=config.get('split', 'train')
            )
            
            # Add dataset identifier to samples
            for sample in dataset.samples:
                sample['dataset_id'] = config.get('name', 'unknown')
            
            all_samples.extend(dataset.samples)
            self.datasets.append(dataset)
        
        # Create combined dataset
        self.combined_dataset = CombinedGazeDataset(all_samples)
        
        # Create combined data loader
        self.combined_loader = DataLoader(
            self.combined_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    
    def get_combined_loader(self) -> DataLoader:
        """Get combined data loader."""
        return self.combined_loader


class CombinedGazeDataset(Dataset):
    """Dataset that combines samples from multiple sources."""
    
    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load and preprocess image
        if 'image_data' in sample:
            image = sample['image_data']
        else:
            image = cv2.imread(sample['image_path'])
            if image is None:
                image = np.zeros((60, 36, 3), dtype=np.uint8)
        
        image_tensor = preprocess_eye_region(image)
        gaze_target = torch.tensor([sample['yaw'], sample['pitch']], dtype=torch.float32)
        
        return image_tensor, gaze_target


def create_data_loader(data_path: str, dataset_type: str = 'mpiigaze', 
                      batch_size: int = 32, split: str = 'train', 
                      num_workers: int = 4) -> DataLoader:
    """
    Convenience function to create a data loader.
    
    Args:
        data_path: Path to dataset
        dataset_type: Type of dataset
        batch_size: Batch size
        split: Dataset split
        num_workers: Number of worker processes
    
    Returns:
        DataLoader instance
    """
    dataset = GazeDataset(data_path, dataset_type, split=split)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )


def download_sample_data(output_dir: str) -> None:
    """
    Download or create sample data for testing.
    
    Args:
        output_dir: Directory to save sample data
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create sample annotation file
    sample_annotations = [
        "sample_images/person1_001.jpg 0.1 -0.05 person1",
        "sample_images/person1_002.jpg 0.15 -0.02 person1",
        "sample_images/person2_001.jpg -0.08 0.12 person2",
        "sample_images/person2_002.jpg -0.05 0.08 person2",
    ]
    
    with open(output_path / "train_annotations.txt", 'w') as f:
        f.write('\n'.join(sample_annotations))
    
    # Create sample images directory
    sample_images_dir = output_path / "sample_images"
    sample_images_dir.mkdir(exist_ok=True)
    
    # Create dummy images for testing
    for annotation in sample_annotations:
        image_name = annotation.split()[0].split('/')[-1]
        dummy_image = np.random.randint(0, 255, (60, 36, 3), dtype=np.uint8)
        cv2.imwrite(str(sample_images_dir / image_name), dummy_image)
    
    print(f"Sample data created in {output_dir}")


if __name__ == "__main__":
    # Example usage
    data_config = {
        'train': {
            'data_path': 'data/mpiigaze',
            'dataset_type': 'mpiigaze',
            'batch_size': 32,
            'num_workers': 4
        }
    }
    
    # Create sample data for testing
    download_sample_data('data/sample_gaze_data')
    
    # Test data loader
    loader_manager = GazeDataLoader(data_config)
    loader_manager.setup_datasets()
    
    train_loader = loader_manager.get_data_loader('train')
    if train_loader:
        print(f"Train loader created with {len(train_loader)} batches")
        
        # Test loading a batch
        for batch_idx, (images, targets) in enumerate(train_loader):
            print(f"Batch {batch_idx}: images shape {images.shape}, targets shape {targets.shape}")
            if batch_idx >= 2:  # Only test a few batches
                break


