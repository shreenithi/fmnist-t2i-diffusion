import torch
import torch.nn.functional as F
import numpy as np

from .configs import T2IConfig


def _model_eps(model, latents, labels, timesteps):
    """Helper function to handle model prediction with error handling
    
    Args:
        model: The diffusion model
        latents: Input latent representations
        labels: Text labels/embeddings
        timesteps: Current timestep values
        
    Returns:
        Model prediction for the current timestep
    """
    try:
        return model(latents, labels, timesteps)
    except RuntimeError:
        # If the first call fails, try with reshaped timesteps
        if timesteps.dim() == 1:
            timesteps = timesteps.unsqueeze(-1)  # (B,) -> (B, 1)
        return model(latents, labels, timesteps)


@torch.no_grad()
def generate_image(model, labels, num_samples=None, num_inference_steps=None, seed=None, device=None, config=None):
    """Generate images using continuous timesteps (noise levels)
    
    Args:
        model: Trained diffusion model
        labels: Text labels or embeddings for conditioning
        num_samples: Number of images to generate (defaults to batch size of labels)
        num_inference_steps: Number of denoising steps (defaults to config value)
        seed: Random seed for reproducibility
        device: Device to run generation on
        config: Configuration object (defaults to model config)
        
    Returns:
        Generated latent representations
    """
    if device is None: 
        device = next(model.parameters()).device
    if config is None: 
        config = model.cfg
    if num_inference_steps is None:
        num_inference_steps = config.num_inference_steps
    if seed is not None: 
        torch.manual_seed(seed)

    if isinstance(labels, np.ndarray): 
        labels = torch.from_numpy(labels)
    labels = labels.to(device)
    
    # Handle different label formats
    if labels.dtype in (torch.int32, torch.int64):
        if config.text_encoding_type == "one_hot":
            labels = F.one_hot(labels, num_classes=config.num_classes).float()
        elif config.text_encoding_type == "learned":
            labels = labels  # Keep as integer indices
        else:
            raise ValueError(f"Unexpected text_encoding_type: {config.text_encoding_type}")
    
    labels = labels.float()

    if num_samples is None: 
        num_samples = labels.shape[0]
    latents = torch.randn(num_samples, config.n_channels, config.image_size, config.image_size, device=device)

    # Create noise levels from config
    noise_levels = torch.linspace(config.noise_start, config.noise_end, num_inference_steps + 1, device=device)

    was_train = model.training
    model.eval()
    
    for i in range(len(noise_levels) - 1):
        curr_noise = noise_levels[i]
        next_noise = noise_levels[i + 1]
        
        # Broadcast noise level to match batch size
        curr_noise_batch = torch.full((num_samples,), curr_noise, device=device, dtype=torch.float32)
        
        # Predict clean image at current noise level
        if config.use_amp:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                pred_clean = _model_eps(model, latents, labels, curr_noise_batch)
        else:
            pred_clean = _model_eps(model, latents, labels, curr_noise_batch)
        
        # Update latents using correct reverse diffusion step
        # For a model predicting clean images: x_{t-1} = (1-t_{t-1}) * x_0_pred + t_{t-1} * noise
        next_signal = 1 - next_noise
        latents = next_signal * pred_clean + next_noise * torch.randn_like(latents)
    
    if was_train: 
        model.train()
    return latents


@torch.no_grad()
def decode_latents_to_images(latents, config=None, device=None):
    """Decode VAE latents to images using configurable VAE settings
    
    Args:
        latents: Latent representations to decode
        config: Configuration object with VAE settings
        device: Device to run decoding on
        
    Returns:
        Decoded images in pixel space
    """
    if device is None: 
        device = latents.device
    if config is None:
        # Try to get config from model if available
        if hasattr(latents, 'cfg'):
            config = latents.cfg
        else:
            # Fallback to default config
            config = T2IConfig()
    
    try:
        from diffusers import AutoencoderKL
        
        # Load VAE with configurable path
        vae = AutoencoderKL.from_pretrained(config.vae_path).to(device).eval()
        
        # Apply configurable scaling
        x = (latents.to(torch.float32) / config.vae_latent_scale)
        
        # Decode
        imgs = vae.decode(x).sample
        
        # Normalize to [0, 1] range
        return (imgs.clamp(-1, 1) + 1) * 0.5
        
    except Exception as e:
        print(f"Warning: VAE decoding failed: {e}")
        print("Returning raw latents instead")
        return latents


@torch.no_grad()
def encode_images_to_latents(images, config=None, device=None):
    """Encode images to VAE latents (reverse of decode_latents_to_images)"""
    if device is None: 
        device = images.device
    if config is None:
        config = Config()
    
    try:
        from diffusers import AutoencoderKL
        
        # Load VAE
        vae = AutoencoderKL.from_pretrained(config.vae_path).to(device).eval()
        
        # Normalize images to [-1, 1] range
        images = (images * 2) - 1
        
        # Encode
        latents = vae.encode(images).latent_dist.sample()
        
        # Apply scaling
        latents = latents * config.vae_latent_scale
        
        return latents
        
    except Exception as e:
        print(f"Warning: VAE encoding failed: {e}")
        return None
