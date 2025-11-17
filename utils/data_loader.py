"""
Utility functions for loading and preprocessing EEG data.
"""

import os
import numpy as np
import scipy.io
import torch
from torch.utils.data import TensorDataset


def load_subject_data(subject_num, data_dir='data'):
    """
    Load EEG data for a single subject.
    
    Args:
        subject_num: Subject number (1-29)
        data_dir: Directory containing .mat files
        
    Returns:
        Tuple of (movement_left, movement_right) numpy arrays
        Each array has shape (68 channels, N samples)
    """
    subject_file = os.path.join(data_dir, f's{subject_num:02d}.mat')
    
    if not os.path.exists(subject_file):
        raise FileNotFoundError(f"Subject file not found: {subject_file}")
    
    data = scipy.io.loadmat(subject_file)
    
    # Extract left and right motor imagery data
    movement_left = data['eeg'][0][0][7]
    movement_right = data['eeg'][0][0][8]
    
    return movement_left, movement_right


def prepare_dataset(movement_left, movement_right, mean_center=True):
    """
    Prepare PyTorch dataset from left and right motor imagery data.
    
    Args:
        movement_left: Left motor imagery (68 channels, N samples)
        movement_right: Right motor imagery (68 channels, N samples)
        mean_center: Whether to mean-center the data (default: True)
        
    Returns:
        TensorDataset containing (features, labels)
        - features: shape (2*N, 68)
        - labels: shape (2*N,) with 0=left, 1=right
    """
    # Mean-center the data if requested
    if mean_center:
        movement_left = movement_left - np.mean(movement_left, axis=1, keepdims=True)
        movement_right = movement_right - np.mean(movement_right, axis=1, keepdims=True)
    
    # Concatenate left and right data
    all_data = np.concatenate((movement_left, movement_right), axis=1)
    
    # Transpose to shape (samples, channels)
    all_data = all_data.T
    
    # Create labels (0 for left, 1 for right)
    n_samples = movement_left.shape[1]
    labels = np.concatenate([
        np.zeros(n_samples, dtype=int),
        np.ones(n_samples, dtype=int)
    ])
    
    # Convert to PyTorch tensors
    X = torch.from_numpy(all_data).float()
    y = torch.from_numpy(labels).long()
    
    return TensorDataset(X, y)


def get_data_loaders(subject_num, batch_size=1024, train_split=0.8, data_dir='data'):
    """
    Load data and create train/test data loaders.
    
    Args:
        subject_num: Subject number (1-29)
        batch_size: Batch size for data loaders
        train_split: Fraction of data to use for training (default: 0.8)
        data_dir: Directory containing .mat files
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Load and prepare dataset
    movement_left, movement_right = load_subject_data(subject_num, data_dir)
    dataset = prepare_dataset(movement_left, movement_right)
    
    # Split into train and test
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, test_loader