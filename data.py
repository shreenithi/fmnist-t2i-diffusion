import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Tuple, Optional

from .configs import T2IConfig


class MMapLatentTextDataset(Dataset):
    """Memory-mapped dataset for latent images and text embeddings
    
    This dataset efficiently loads large latent representations and text embeddings
    using memory mapping to reduce memory usage.
    
    Args:
        latent_path: Path to .npy file containing latent representations
        text_path: Path to .npy file containing text embeddings/labels
    """
    def __init__(self, latent_path: str, text_path: str):
        arr = np.load(latent_path, mmap_mode='r')
        self.latents = arr.astype(np.float32) if arr.dtype == np.float16 else arr
        self.texts = np.load(text_path, mmap_mode='r')  # (N, 10) one-hot
        assert len(self.latents) == len(self.texts)

    def __len__(self) -> int: 
        return len(self.latents)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        lat = torch.from_numpy(self.latents[idx].copy()).float()
        txt = torch.from_numpy(self.texts[idx].copy()).float()
        return lat, txt


def setup_data_loaders(cfg: T2IConfig) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Setup train and validation data loaders with configurable parameters
    
    Args:
        cfg: Configuration object containing data loading parameters
        
    Returns:
        Tuple of (train_loader, val_loader) where val_loader may be None
    """
    full = MMapLatentTextDataset(cfg.latent_path, cfg.text_emb_path)
    N = len(full)
    n_val = int(round(N * cfg.val_fraction))
    
    # Create reproducible train/val split
    indices = torch.randperm(N, generator=torch.Generator().manual_seed(cfg.seed_split)).tolist()
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_ds = Subset(full, train_idx)
    val_ds = Subset(full, val_idx) if n_val > 0 else None

    # Train dataloader
    train_dl = DataLoader(
        train_ds, 
        batch_size=cfg.batch_size, 
        shuffle=True,
        num_workers=cfg.num_workers, 
        pin_memory=cfg.pin_memory, 
        drop_last=cfg.drop_last
    )
    
    # Validation dataloader
    val_dl = None
    if val_ds is not None:
        val_dl = DataLoader(
            val_ds, 
            batch_size=cfg.batch_size, 
            shuffle=False,
            num_workers=cfg.val_num_workers, 
            pin_memory=cfg.pin_memory, 
            drop_last=cfg.drop_last
        )
    
    print(f"Train samples: {len(train_ds)}  | Val samples: {0 if val_ds is None else len(val_ds)}")
    print(f"Batch size: {cfg.batch_size} | Accumulation steps: {cfg.accumulation_steps}")
    print(f"Effective batch size: {cfg.batch_size * cfg.accumulation_steps}")
    
    return train_dl, val_dl


def create_text_embeddings(labels: torch.Tensor, cfg: T2IConfig, device: Optional[torch.device] = None) -> torch.Tensor:
    """Create text embeddings based on configuration
    
    Args:
        labels: Input labels (can be class indices or one-hot encodings)
        cfg: Configuration object specifying text encoding type
        device: Device to place embeddings on
        
    Returns:
        Text embeddings in the appropriate format for the model
        
    Raises:
        ValueError: If text encoding type is not supported or labels are in wrong format
    """
    if device is None:
        device = labels.device if hasattr(labels, 'device') else 'cpu'
    
    if cfg.text_encoding_type == "one_hot":
        if labels.dtype in (torch.int32, torch.int64):
            return F.one_hot(labels, num_classes=cfg.num_classes).float().to(device)
        else:
            return labels.float().to(device)
    
    elif cfg.text_encoding_type == "learned":
        if labels.dtype in (torch.int32, torch.int64):
            return labels.to(device)  # Keep as integer indices
        else:
            raise ValueError(f"Expected integer class indices for learned embeddings, got {labels.dtype}")
    
    elif cfg.text_encoding_type == "pretrained":
        raise NotImplementedError("Pretrained text encoders not yet implemented")
    
    else:
        raise ValueError(f"Unknown text_encoding_type: {cfg.text_encoding_type}")


def get_dataset_info(cfg: T2IConfig) -> dict:
    """Get information about the dataset
    
    Args:
        cfg: Configuration object containing dataset paths
        
    Returns:
        Dictionary containing dataset information or error details
    """
    try:
        full = MMapLatentTextDataset(cfg.latent_path, cfg.text_emb_path)
        N = len(full)
        n_val = int(round(N * cfg.val_fraction))
        n_train = N - n_val
        
        # Get sample shapes
        sample_latent, sample_text = full[0]
        latent_shape = sample_latent.shape
        text_shape = sample_text.shape
        
        return {
            'total_samples': N,
            'train_samples': n_train,
            'val_samples': n_val,
            'latent_shape': latent_shape,
            'text_shape': text_shape,
            'latent_dtype': str(sample_latent.dtype),
            'text_dtype': str(sample_text.dtype)
        }
    except Exception as e:
        return {'error': str(e)}
