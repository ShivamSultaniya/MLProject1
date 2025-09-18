"""
Dataset loaders for eye gaze estimation datasets (MPIIGaze, GazeCapture).
"""

import os
import h5py
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict, Any, List
import json
import pandas as pd
from pathlib import Path
import logging

from .utils import preprocess_eye_image

logger = logging.getLogger(__name__)


class MPIIGazeDataset(Dataset):
    """
    Dataset loader for MPIIGaze dataset.
    
    MPIIGaze is a dataset for appearance-based gaze estimation containing
    213,659 images from 15 participants.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[callable] = None,
        normalize_gaze: bool = True,
        person_ids: Optional[List[str]] = None
    ):
        """
        Initialize MPIIGaze dataset.
        
        Args:
            data_dir: Path to MPIIGaze dataset directory
            split: Dataset split ('train', 'test', 'all')
            transform: Optional image transforms
            normalize_gaze: Whether to normalize gaze vectors
            person_ids: Specific person IDs to include
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.normalize_gaze = normalize_gaze
        
        # Load data
        self.samples = self._load_data(person_ids)
        
        logger.info(f"Loaded {len(self.samples)} samples from MPIIGaze ({split} split)")
    
    def _load_data(self, person_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Load dataset samples."""
        samples = []
        
        # Default person IDs if not specified
        if person_ids is None:
            person_ids = [f"p{i:02d}" for i in range(15)]
        
        for person_id in person_ids:
            person_file = self.data_dir / f"{person_id}.h5"
            
            if not person_file.exists():
                logger.warning(f"Person file not found: {person_file}")
                continue
            
            try:
                with h5py.File(person_file, 'r') as f:
                    # Load images and gaze data
                    images = f['Data']['data'][()]
                    gazes = f['Data']['label'][()]
                    
                    # Split data if needed
                    if self.split == 'train':
                        # Use first 80% for training
                        n_train = int(0.8 * len(images))
                        images = images[:n_train]
                        gazes = gazes[:n_train]
                    elif self.split == 'test':
                        # Use last 20% for testing
                        n_train = int(0.8 * len(images))
                        images = images[n_train:]
                        gazes = gazes[n_train:]
                    # 'all' uses entire dataset
                    
                    # Create sample entries
                    for i, (image, gaze) in enumerate(zip(images, gazes)):
                        samples.append({
                            'image': image,
                            'gaze': gaze,
                            'person_id': person_id,
                            'sample_id': i
                        })
                        
            except Exception as e:
                logger.error(f"Error loading {person_file}: {e}")
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Get image and gaze
        image = sample['image']  # Already normalized in MPIIGaze
        gaze = sample['gaze'].astype(np.float32)
        
        # Convert image to proper format
        if len(image.shape) == 3 and image.shape[0] == 3:
            # Already in CHW format, convert to HWC for processing
            image = np.transpose(image, (1, 2, 0))
        
        # Ensure image is in proper range [0, 255] and uint8
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # Convert to BGR for OpenCV compatibility
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default preprocessing
            image = preprocess_eye_image(image)
        
        # Normalize gaze vector if requested
        if self.normalize_gaze:
            gaze_norm = np.linalg.norm(gaze)
            if gaze_norm > 0:
                gaze = gaze / gaze_norm
        
        return {
            'image': image,
            'gaze': torch.tensor(gaze, dtype=torch.float32),
            'person_id': sample['person_id'],
            'sample_id': sample['sample_id']
        }


class GazeCaptureDataset(Dataset):
    """
    Dataset loader for GazeCapture dataset.
    
    GazeCapture contains gaze data from mobile devices with over 2.5M frames
    from more than 1,450 people.
    """
    
    def __init__(
        self,
        data_dir: str,
        metadata_file: str = 'metadata.json',
        split: str = 'train',
        transform: Optional[callable] = None,
        eye_type: str = 'both',  # 'left', 'right', 'both'
        device_type: Optional[str] = None,  # Filter by device type
        max_samples: Optional[int] = None
    ):
        """
        Initialize GazeCapture dataset.
        
        Args:
            data_dir: Path to GazeCapture dataset directory
            metadata_file: Name of metadata file
            split: Dataset split ('train', 'val', 'test')
            transform: Optional image transforms
            eye_type: Which eye(s) to use
            device_type: Filter by device type (iPhone, iPad, etc.)
            max_samples: Maximum number of samples to load
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.eye_type = eye_type
        self.device_type = device_type
        self.max_samples = max_samples
        
        # Load metadata
        metadata_path = self.data_dir / metadata_file
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Load samples
        self.samples = self._load_samples()
        
        logger.info(f"Loaded {len(self.samples)} samples from GazeCapture ({split} split)")
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load dataset samples based on metadata."""
        samples = []
        
        # Get subjects for current split
        split_subjects = self.metadata.get(f'{self.split}_subjects', [])
        
        for subject_id in split_subjects:
            subject_dir = self.data_dir / f"{subject_id:05d}"
            
            if not subject_dir.exists():
                logger.warning(f"Subject directory not found: {subject_dir}")
                continue
            
            # Load subject metadata
            info_file = subject_dir / "info.json"
            if not info_file.exists():
                continue
            
            try:
                with open(info_file, 'r') as f:
                    subject_info = json.load(f)
                
                # Filter by device type if specified
                if self.device_type and subject_info.get('DeviceName', '').lower() != self.device_type.lower():
                    continue
                
                # Load frames
                frames_file = subject_dir / "frames.json"
                if frames_file.exists():
                    with open(frames_file, 'r') as f:
                        frames_data = json.load(f)
                    
                    for frame_data in frames_data:
                        # Check if required files exist
                        frame_id = frame_data['frame_id']
                        
                        if self.eye_type in ['left', 'both']:
                            left_eye_file = subject_dir / f"appleFace_{frame_id}_left.jpg"
                            if left_eye_file.exists():
                                samples.append({
                                    'subject_id': subject_id,
                                    'frame_id': frame_id,
                                    'eye_type': 'left',
                                    'image_path': left_eye_file,
                                    'gaze': np.array([frame_data['gaze_x'], frame_data['gaze_y']]),
                                    'device_info': subject_info
                                })
                        
                        if self.eye_type in ['right', 'both']:
                            right_eye_file = subject_dir / f"appleFace_{frame_id}_right.jpg"
                            if right_eye_file.exists():
                                samples.append({
                                    'subject_id': subject_id,
                                    'frame_id': frame_id,
                                    'eye_type': 'right',
                                    'image_path': right_eye_file,
                                    'gaze': np.array([frame_data['gaze_x'], frame_data['gaze_y']]),
                                    'device_info': subject_info
                                })
                        
                        # Limit samples if specified
                        if self.max_samples and len(samples) >= self.max_samples:
                            return samples
                            
            except Exception as e:
                logger.error(f"Error loading subject {subject_id}: {e}")
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load image
        image_path = sample['image_path']
        image = cv2.imread(str(image_path))
        
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default preprocessing
            image = preprocess_eye_image(image)
        
        # Convert 2D gaze to 3D (assuming looking at screen)
        gaze_2d = sample['gaze'].astype(np.float32)
        gaze_3d = np.array([gaze_2d[0], gaze_2d[1], 1.0], dtype=np.float32)
        
        # Normalize to unit vector
        gaze_norm = np.linalg.norm(gaze_3d)
        if gaze_norm > 0:
            gaze_3d = gaze_3d / gaze_norm
        
        return {
            'image': image,
            'gaze': torch.tensor(gaze_3d, dtype=torch.float32),
            'subject_id': sample['subject_id'],
            'frame_id': sample['frame_id'],
            'eye_type': sample['eye_type']
        }


class EyeTrackingDataset(Dataset):
    """
    Generic eye tracking dataset that can combine multiple datasets.
    """
    
    def __init__(
        self,
        datasets: List[Dataset],
        weights: Optional[List[float]] = None
    ):
        """
        Initialize combined dataset.
        
        Args:
            datasets: List of individual datasets
            weights: Sampling weights for each dataset
        """
        self.datasets = datasets
        self.weights = weights or [1.0] * len(datasets)
        
        # Calculate cumulative lengths
        self.lengths = [len(ds) for ds in datasets]
        self.cumulative_lengths = np.cumsum([0] + self.lengths)
        self.total_length = sum(self.lengths)
        
        logger.info(f"Combined dataset with {self.total_length} total samples")
        logger.info(f"Dataset lengths: {self.lengths}")
    
    def __len__(self) -> int:
        return self.total_length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Find which dataset this index belongs to
        dataset_idx = np.searchsorted(self.cumulative_lengths[1:], idx, side='right')
        local_idx = idx - self.cumulative_lengths[dataset_idx]
        
        # Get sample from appropriate dataset
        sample = self.datasets[dataset_idx][local_idx]
        
        # Add dataset identifier
        sample['dataset_id'] = dataset_idx
        
        return sample


def create_gaze_dataloaders(
    config: Dict[str, Any],
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create train, validation, and test dataloaders for gaze estimation.
    
    Args:
        config: Dataset configuration
        batch_size: Batch size
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    datasets = []
    
    # Load MPIIGaze if specified
    if 'mpiigaze' in config:
        mpiigaze_config = config['mpiigaze']
        
        train_dataset = MPIIGazeDataset(
            data_dir=mpiigaze_config['data_dir'],
            split='train',
            person_ids=mpiigaze_config.get('person_ids')
        )
        
        val_dataset = MPIIGazeDataset(
            data_dir=mpiigaze_config['data_dir'],
            split='test',
            person_ids=mpiigaze_config.get('person_ids')
        )
        
        datasets.append(('mpiigaze', train_dataset, val_dataset))
    
    # Load GazeCapture if specified
    if 'gazecapture' in config:
        gazecapture_config = config['gazecapture']
        
        train_dataset = GazeCaptureDataset(
            data_dir=gazecapture_config['data_dir'],
            split='train',
            eye_type=gazecapture_config.get('eye_type', 'both'),
            device_type=gazecapture_config.get('device_type'),
            max_samples=gazecapture_config.get('max_samples')
        )
        
        val_dataset = GazeCaptureDataset(
            data_dir=gazecapture_config['data_dir'],
            split='val',
            eye_type=gazecapture_config.get('eye_type', 'both'),
            device_type=gazecapture_config.get('device_type'),
            max_samples=gazecapture_config.get('max_samples')
        )
        
        test_dataset = GazeCaptureDataset(
            data_dir=gazecapture_config['data_dir'],
            split='test',
            eye_type=gazecapture_config.get('eye_type', 'both'),
            device_type=gazecapture_config.get('device_type'),
            max_samples=gazecapture_config.get('max_samples')
        )
        
        datasets.append(('gazecapture', train_dataset, val_dataset, test_dataset))
    
    # Combine datasets if multiple are specified
    if len(datasets) > 1:
        train_datasets = [ds[1] for ds in datasets]
        val_datasets = [ds[2] for ds in datasets]
        test_datasets = [ds[3] for ds in datasets if len(ds) > 3]
        
        combined_train = EyeTrackingDataset(train_datasets)
        combined_val = EyeTrackingDataset(val_datasets)
        combined_test = EyeTrackingDataset(test_datasets) if test_datasets else None
        
    elif len(datasets) == 1:
        combined_train = datasets[0][1]
        combined_val = datasets[0][2]
        combined_test = datasets[0][3] if len(datasets[0]) > 3 else None
        
    else:
        raise ValueError("No datasets specified in configuration")
    
    # Create data loaders
    train_loader = DataLoader(
        combined_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        combined_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = None
    if combined_test:
        test_loader = DataLoader(
            combined_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    return train_loader, val_loader, test_loader
