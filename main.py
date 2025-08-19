#!/usr/bin/env python3
"""
Main training script for FMNIST Text-to-Image Transformer Latent Diffusion model.
This script uses the modular components to train the model.
"""

import argparse
import sys
from pathlib import Path

from configs import T2IConfig
from trainer import FMNISTT2ITrainer
from data import get_dataset_info


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train FMNIST T2I Diffusion Model")
    
    # Data paths
    parser.add_argument("--latent-path", type=str, help="Path to latent images")
    parser.add_argument("--text-emb-path", type=str, help="Path to text embeddings")
    
    # Model architecture
    parser.add_argument("--embed-dim", type=int, help="Embedding dimension")
    parser.add_argument("--n-layers", type=int, help="Number of transformer layers")
    parser.add_argument("--n-heads", type=int, help="Number of attention heads")
    parser.add_argument("--mlp-multiplier", type=int, help="MLP multiplier")
    
    # Training
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--total-steps", type=int, help="Total training steps")
    parser.add_argument("--accumulation-steps", type=int, help="Gradient accumulation steps")
    
    # Text encoding
    parser.add_argument("--text-encoding", choices=["one_hot", "learned"], 
                       help="Text encoding type")
    parser.add_argument("--num-classes", type=int, help="Number of classes")
    
    # VAE
    parser.add_argument("--vae-path", type=str, help="VAE model path")
    
    # Hardware
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile() if available")
    
    # Logging
    parser.add_argument("--run-id", type=str, help="Custom run ID")
    parser.add_argument("--resume-from", type=str, help="Resume from checkpoint")
    
    return parser.parse_args()


def create_config_from_args(args):
    """Create configuration from command line arguments"""
    cfg = T2IConfig()
    
    # Override with command line arguments
    if args.latent_path:
        cfg.latent_path = args.latent_path
    if args.text_emb_path:
        cfg.text_emb_path = args.text_emb_path
    if args.embed_dim:
        cfg.embed_dim = args.embed_dim
    if args.n_layers:
        cfg.n_layers = args.n_layers
    if args.n_heads:
        cfg.n_heads = args.n_heads
    if args.mlp_multiplier:
        cfg.mlp_multiplier = args.mlp_multiplier
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.lr = args.lr
    if args.total_steps:
        cfg.total_steps = args.total_steps
    if args.accumulation_steps:
        cfg.accumulation_steps = args.accumulation_steps
    if args.text_encoding:
        cfg.text_encoding_type = args.text_encoding
    if args.num_classes:
        cfg.num_classes = args.num_classes
    if args.vae_path:
        cfg.vae_path = args.vae_path
    if args.device:
        cfg.device = args.device
    if args.no_amp:
        cfg.use_amp = False
    if args.compile:
        cfg.compile_model = True
    if args.run_id:
        cfg.run_id = args.run_id
    if args.resume_from:
        cfg.resume_from = args.resume_from
    
    return cfg


def main():
    """Main training function"""
    # Parse command line arguments
    args = parse_args()
    
    # Create configuration
    cfg = create_config_from_args(args)
    
    # Print configuration
    print("=" * 60)
    print("FMNIST Text-to-Image Training Configuration")
    print("=" * 60)
    print(f"Data paths:")
    print(f"  Latent path: {cfg.latent_path}")
    print(f"  Text embeddings: {cfg.text_emb_path}")
    print(f"Model architecture:")
    print(f"  Embedding dimension: {cfg.embed_dim}")
    print(f"  Layers: {cfg.n_layers}")
    print(f"  Attention heads: {cfg.n_heads}")
    print(f"  MLP multiplier: {cfg.mlp_multiplier}")
    print(f"Training:")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  Learning rate: {cfg.lr}")
    print(f"  Total steps: {cfg.total_steps}")
    print(f"  Accumulation steps: {cfg.accumulation_steps}")
    print(f"Text encoding: {cfg.text_encoding_type}")
    print(f"VAE path: {cfg.vae_path}")
    print(f"Device: {cfg.device or 'auto'}")
    print(f"AMP: {cfg.use_amp}")
    print(f"Model compilation: {cfg.compile_model}")
    print("=" * 60)
    
    # Check if data files exist
    if not Path(cfg.latent_path).exists():
        print(f"Error: Latent path does not exist: {cfg.latent_path}")
        sys.exit(1)
    if not Path(cfg.text_emb_path).exists():
        print(f"Error: Text embeddings path does not exist: {cfg.text_emb_path}")
        sys.exit(1)
    
    # Get dataset information
    print("\nDataset information:")
    dataset_info = get_dataset_info(cfg)
    if 'error' in dataset_info:
        print(f"Warning: Could not get dataset info: {dataset_info['error']}")
    else:
        for key, value in dataset_info.items():
            print(f"  {key}: {value}")
    
    # Create and start trainer
    print(f"\nInitializing trainer...")
    trainer = FMNISTT2ITrainer(cfg)
    
    print(f"\nStarting training...")
    trainer.train()


if __name__ == "__main__":
    main()
