"""
DEAP Dataset DataLoader for Perceiver-based VAD Prediction

Dataset Structure:
- 32 subjects, 40 trials each (1280 total samples)
- 40 channels: 32 EEG + 8 peripheral (EOG, EMG, GSR, Respiration, Plethysmograph, Temperature)
- 8064 time samples per trial (63 seconds at 128 Hz)
- Target: Valence, Arousal, Dominance (VAD) - 3 continuous values [1-9 scale]
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional


class DEAPDataset(Dataset):
    """
    DEAP Dataset for emotion recognition from physiological signals.
    
    Args:
        data_dir: Path to data_preprocessed_python directory
        subject_ids: List of subject IDs to include (1-32)
        sequence_length: Length of input sequences (default 8064 for full trial)
        stride: Stride for sliding window (if < sequence_length, creates overlapping windows)
        normalize: Whether to normalize each channel independently
        use_channels: List of channel indices to use (default: all 40 channels)
    """
    
    def __init__(
        self,
        data_dir: str,
        subject_ids: Optional[List[int]] = None,
        sequence_length: int = 8064,
        stride: Optional[int] = None,
        normalize: bool = True,
        use_channels: Optional[List[int]] = None,
    ):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.stride = stride if stride is not None else sequence_length
        self.normalize = normalize
        self.use_channels = use_channels if use_channels is not None else list(range(40))
        
        # Default to all subjects if not specified
        if subject_ids is None:
            subject_ids = list(range(1, 33))  # 1-32
        
        self.samples = []
        self._load_data(subject_ids)
        
        print(f"Loaded DEAP dataset:")
        print(f"  - Subjects: {len(subject_ids)}")
        print(f"  - Total samples: {len(self.samples)}")
        print(f"  - Channels: {len(self.use_channels)}")
        print(f"  - Sequence length: {self.sequence_length}")
        print(f"  - Stride: {self.stride}")
    
    def _load_data(self, subject_ids: List[int]):
        """Load data from all specified subjects."""
        for subject_id in subject_ids:
            filepath = os.path.join(self.data_dir, f's{subject_id:02d}.dat')
            
            if not os.path.exists(filepath):
                print(f"Warning: File not found: {filepath}")
                continue
            
            with open(filepath, 'rb') as f:
                subject_data = pickle.load(f, encoding='latin1')
            
            # subject_data['data']: (40 trials, 40 channels, 8064 samples)
            # subject_data['labels']: (40 trials, 4) - [Valence, Arousal, Dominance, Liking]
            
            data = subject_data['data']  # (40, 40, 8064)
            labels = subject_data['labels'][:, :3]  # (40, 3) - Keep only VAD, drop Liking
            
            # Process each trial
            for trial_idx in range(data.shape[0]):
                trial_data = data[trial_idx, self.use_channels, :]  # (n_channels, 8064)
                trial_label = labels[trial_idx]  # (3,) - VAD
                
                # Create windows if sequence_length < full trial length
                num_samples = trial_data.shape[1]
                for start_idx in range(0, num_samples - self.sequence_length + 1, self.stride):
                    end_idx = start_idx + self.sequence_length
                    window_data = trial_data[:, start_idx:end_idx]  # (n_channels, seq_len)
                    
                    self.samples.append({
                        'data': window_data,
                        'label': trial_label,
                        'subject_id': subject_id,
                        'trial_id': trial_idx,
                    })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            data: (n_channels, sequence_length) tensor
            label: (3,) tensor with [Valence, Arousal, Dominance]
        """
        sample = self.samples[idx]
        data = sample['data'].copy()  # (n_channels, sequence_length)
        label = sample['label'].copy()  # (3,)
        
        # Normalize each channel independently (z-score normalization)
        if self.normalize:
            data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)
        
        # Transpose to (sequence_length, n_channels) for Perceiver
        data = data.T
        
        # Convert to torch tensors
        data = torch.from_numpy(data).float()
        label = torch.from_numpy(label).float()
        
        return data, label


def create_dataloaders(
    data_dir: str,
    train_subjects: List[int],
    val_subjects: List[int],
    test_subjects: List[int],
    sequence_length: int = 2048,  # Reduced for faster training
    stride: Optional[int] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    use_channels: Optional[List[int]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Path to data_preprocessed_python directory
        train_subjects: List of subject IDs for training
        val_subjects: List of subject IDs for validation
        test_subjects: List of subject IDs for testing
        sequence_length: Length of input sequences
        stride: Stride for sliding window
        batch_size: Batch size
        num_workers: Number of data loading workers
        use_channels: List of channel indices to use
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    train_dataset = DEAPDataset(
        data_dir=data_dir,
        subject_ids=train_subjects,
        sequence_length=sequence_length,
        stride=stride,
        normalize=True,
        use_channels=use_channels,
    )
    
    val_dataset = DEAPDataset(
        data_dir=data_dir,
        subject_ids=val_subjects,
        sequence_length=sequence_length,
        stride=stride,
        normalize=True,
        use_channels=use_channels,
    )
    
    test_dataset = DEAPDataset(
        data_dir=data_dir,
        subject_ids=test_subjects,
        sequence_length=sequence_length,
        stride=stride,
        normalize=True,
        use_channels=use_channels,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


def get_default_channel_groups():
    """
    Returns predefined channel groups for different modality combinations.
    
    Returns:
        Dictionary with different channel configurations
    """
    return {
        'all': list(range(40)),  # All 40 channels
        'eeg_only': list(range(32)),  # Only EEG channels
        'peripheral_only': list(range(32, 40)),  # Only peripheral channels
        'eeg_gsr_resp': list(range(32)) + [36, 37],  # EEG + GSR + Respiration
        'physiological': [32, 33, 34, 35, 36, 37, 38, 39],  # All peripheral
    }


if __name__ == '__main__':
    # Example usage
    data_dir = './dataset/deap-dataset/data_preprocessed_python'
    
    # Split subjects: 70% train, 15% val, 15% test
    # Subject-independent split (no subject appears in multiple sets)
    train_subjects = list(range(1, 23))  # 22 subjects
    val_subjects = list(range(23, 28))   # 5 subjects
    test_subjects = list(range(28, 33))  # 5 subjects
    
    print(f"Train subjects: {train_subjects}")
    print(f"Val subjects: {val_subjects}")
    print(f"Test subjects: {test_subjects}")
    print()
    
    # Create dataloaders with reduced sequence length for fast testing
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
        sequence_length=2048,  # ~16 seconds at 128 Hz
        stride=2048,  # Non-overlapping windows
        batch_size=8,
        num_workers=0,  # Set to 0 for debugging
        use_channels=None,  # Use all channels
    )
    
    # Test the dataloader
    print("\nTesting dataloader:")
    for batch_idx, (data, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Data shape: {data.shape}")  # (batch, seq_len, n_channels)
        print(f"  Labels shape: {labels.shape}")  # (batch, 3)
        print(f"  Data range: [{data.min():.3f}, {data.max():.3f}]")
        print(f"  Labels sample: {labels[0]}")  # VAD values
        
        if batch_idx >= 2:  # Only show first 3 batches
            break
