#!/usr/bin/env python3
"""
Data preparation script for FMNIST -> SD-VAE latents conversion.
This script converts Fashion MNIST images to VAE latent representations for training.
"""

import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import argparse


def get_transform():
    """Get image transformation pipeline"""
    return transforms.Compose([
        transforms.Resize(64, antialias=True),           # Smaller resolution like reference
        transforms.CenterCrop(64),                       # Center crop like reference
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # 3-channel normalization
    ])


@torch.no_grad()
def encode_batch(images, vae, latent_scale=0.18215, device="cuda"):
    """Encode a batch of images to VAE latents"""
    images = images.to(device, non_blocking=True)
    latents = vae.encode(images).latent_dist.sample() * latent_scale
    return latents.cpu().numpy()


def make_label_encodings(labels, mode="one_hot", num_classes=10):
    """Create label encodings in specified format"""
    if mode == "integer":
        return labels.numpy().astype(np.int64)
    elif mode == "one_hot":
        oh = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
        oh[np.arange(labels.shape[0]), labels.numpy()] = 1.0
        return oh
    else:
        raise ValueError(f"Unknown label encoding mode: {mode}")


def prepare_fmnist_latents(
    output_dir="dataset_fp_32/fmnist",
    vae_model_id="stabilityai/sd-vae-ft-mse",
    batch_size=256,
    num_workers=4,
    split="train",                 # "train" or "test"
    label_encoding="one_hot",      # "one_hot" or "integer"
    partial_every=10000,           # 0 to disable partial saves
    latent_scale=0.18215,          # VAE latent scaling factor
):
    """
    Prepare FMNIST dataset by converting images to VAE latents.
    
    Args:
        output_dir: Directory to save the processed data
        vae_model_id: HuggingFace model ID for the VAE
        batch_size: Batch size for processing
        num_workers: Number of data loader workers
        split: Dataset split ("train" or "test")
        label_encoding: Label encoding format ("one_hot" or "integer")
        partial_every: Save partial results every N samples (0 to disable)
        latent_scale: Scaling factor for VAE latents
    
    Returns:
        tuple: (latents_path, labels_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Dataset & loader - using reference notebook approach
    transform = get_transform()
    ds = datasets.FashionMNIST(
        root="./data",
        train=(split == "train"),
        download=True,
        transform=None,  # We'll apply transforms manually after RGB conversion
    )
    
    # Simple collate function like reference notebook
    def collate_fn(batch):
        labels = [item[1] for item in batch]
        images = [item[0] for item in batch]
        return labels, images
    
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
        collate_fn=collate_fn
    )
    print(f"FMNIST {split} samples: {len(ds)}")

    # VAE
    try:
        from diffusers import AutoencoderKL
        print("Loading VAE...")
        vae = AutoencoderKL.from_pretrained(vae_model_id).to(device).eval()
    except ImportError:
        raise ImportError("diffusers package not found. Install with: pip install diffusers")

    latents_chunks, labels_chunks = [], []
    total = 0

    for labels, images in tqdm(loader, desc="Encoding FMNIST -> latents", dynamic_ncols=True):
        # Convert grayscale PIL images to RGB and apply transforms (like reference)
        images_tensors = torch.cat([transform(image.convert("RGB"))[None] for image in images])

        # Convert labels list to a tensor before passing to make_label_encodings
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        lat_batch = encode_batch(images_tensors, vae, latent_scale=latent_scale, device=device)
        lab_batch = make_label_encodings(labels_tensor, mode=label_encoding)

        latents_chunks.append(lat_batch)
        labels_chunks.append(lab_batch)
        total += len(labels)

        if partial_every and total % partial_every == 0:
            np.save(os.path.join(output_dir, f"image_latents_partial_{total}.npy"),
                    np.concatenate(latents_chunks, axis=0).astype(np.float32))
            np.save(os.path.join(output_dir, f"label_encodings_partial_{total}.npy"),
                    np.concatenate(labels_chunks, axis=0))
            print(f"Saved partial arrays at {total} samples")

    if total == 0:
        raise RuntimeError("No samples processed.")

    image_latents = np.concatenate(latents_chunks, axis=0).astype(np.float32)
    label_encodings = np.concatenate(labels_chunks, axis=0)

    # Final save
    latents_path = os.path.join(output_dir, "image_latents.npy")
    labels_path = os.path.join(output_dir, "label_encodings.npy")
    np.save(latents_path, image_latents)
    np.save(labels_path, label_encodings)

    print(f"Saved latents: {image_latents.shape} -> {latents_path}")
    print(f"Saved labels:  {label_encodings.shape} -> {labels_path}")
    print(f"latents dtype: {image_latents.dtype}, min/max: {image_latents.min():.3f}/{image_latents.max():.3f}")
    
    # Print dataset statistics
    print(f"\nDataset Statistics:")
    print(f"- Total samples: {total}")
    print(f"- Latents shape: {image_latents.shape}")
    print(f"- Labels shape: {label_encodings.shape}")
    print(f"- Latents range: [{image_latents.min():.4f}, {image_latents.max():.4f}]")
    print(f"- Labels encoding: {label_encoding}")
    
    return latents_path, labels_path


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Prepare FMNIST dataset by converting to VAE latents")
    
    # Required arguments
    parser.add_argument("--output-dir", type=str, default="dataset_fp_32/fmnist/new",
                       help="Output directory for processed data")
    
    # Optional arguments
    parser.add_argument("--vae-model", type=str, default="stabilityai/sd-vae-ft-mse",
                       help="VAE model ID from HuggingFace")
    parser.add_argument("--split", type=str, choices=["train", "test"], default="train",
                       help="Dataset split to process")
    parser.add_argument("--label-encoding", type=str, choices=["one_hot", "integer"], default="one_hot",
                       help="Label encoding format")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size for processing")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of data loader workers")
    parser.add_argument("--partial-every", type=int, default=10000,
                       help="Save partial results every N samples (0 to disable)")
    parser.add_argument("--latent-scale", type=float, default=0.18215,
                       help="Scaling factor for VAE latents")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("FMNIST -> VAE Latents Data Preparation")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"VAE model: {args.vae_model}")
    print(f"Split: {args.split}")
    print(f"Label encoding: {args.label_encoding}")
    print(f"Batch size: {args.batch_size}")
    print(f"Workers: {args.num_workers}")
    print(f"Partial saves: every {args.partial_every} samples")
    print(f"Latent scale: {args.latent_scale}")
    print("=" * 60)
    
    try:
        # Run the data preparation
        latents_path, labels_path = prepare_fmnist_latents(
            output_dir=args.output_dir,
            vae_model_id=args.vae_model,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            split=args.split,
            label_encoding=args.label_encoding,
            partial_every=args.partial_every,
            latent_scale=args.latent_scale,
        )
        
        print(f" & Data preparation completed successfully!")
        print(f"Latents saved to: {latents_path}")
        print(f"Labels saved to: {labels_path}")
        
    except Exception as e:
        print(f"\nError during data preparation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
