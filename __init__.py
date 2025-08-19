"""
FMNIST Text-to-Image Transformer Latent Diffusion - Modular Implementation

This package provides a modular implementation of the FMNIST Text-to-Image Transformer Latent Diffusion model,
organized into logical components for better maintainability and reusability.

Modules:
- configs: Configuration classes (T2IConfig)
- models: Neural network model definitions (FMNISTT2IModel)
- attention: Attention mechanisms (FlashAttention and standard)
- data: Dataset and data loading utilities
- data_prep: Data preparation script (FMNIST -> VAE latents)
- diffusion: Diffusion sampling and generation logic
- trainer: Training loop and validation (FMNISTT2ITrainer)
- utils: Utility functions for logging and checkpointing
- main: Main training script entry point with command line arguments
"""

from .configs import T2IConfig
from .models import FMNISTT2IModel, DecoderBlock, PatchEmbed, PatchUnembed
from .attention import FlashSelfAttention, FlashCrossAttention
from .data import MMapLatentTextDataset, setup_data_loaders, create_text_embeddings, get_dataset_info
from .data_prep import prepare_fmnist_latents, encode_batch, make_label_encodings
from .diffusion import generate_image, decode_latents_to_images, encode_images_to_latents
from .trainer import FMNISTT2ITrainer

# Convenience aliases for cleaner imports
Trainer = FMNISTT2ITrainer
from .utils import (
    setup_logging, save_checkpoint, resume_from_checkpoint, 
    get_model_info, log_model_architecture
)

__version__ = "2.0.0"
__all__ = [
    # Configuration
    "T2IConfig",
    
    # Models
    "FMNISTT2IModel",
    "DecoderBlock", 
    "PatchEmbed",
    "PatchUnembed",
    
    # Attention
    "FlashSelfAttention",
    "FlashCrossAttention",
    
    # Data
    "MMapLatentTextDataset",
    "setup_data_loaders",
    "create_text_embeddings",
    "get_dataset_info",
    
    # Data Preparation
    "prepare_fmnist_latents",
    "encode_batch",
    "make_label_encodings",
    
    # Diffusion
    "generate_image",
    "decode_latents_to_images",
    "encode_images_to_latents",
    
    # Training
    "FMNISTT2ITrainer",
    
    # Utilities
    "setup_logging",
    "save_checkpoint",
    "resume_from_checkpoint",
    "get_model_info",
    "log_model_architecture",
]
