from dataclasses import dataclass
from typing import Optional

@dataclass
class T2IConfig:
    """Configuration for Text-to-Image Transformer Latent Diffusion model
    
    This configuration class contains all parameters needed to configure the model,
    training, and inference settings for the FMNIST T2I diffusion model.
    """
    
    # =============================================================================
    # DATA PATHS
    # =============================================================================
    latent_path: str = "dataset/fmnist/image_latents.npy"
    text_emb_path: str = "dataset/fmnist/label_encodings.npy"  # (N,10) one-hot
    
    # =============================================================================
    # OUTPUT PATHS
    # =============================================================================
    tb_logdir: str = "runs/fmnist_t2i"
    ckpt_dir: str = "checkpoints/fmnist_t2i"
    
    # =============================================================================
    # DATA CONFIGURATION
    # =============================================================================
    batch_size: int = 128
    accumulation_steps: int = 1
    val_fraction: float = 0.1          
    seed_split: int = 1337
    num_workers: int = 4
    val_num_workers: int = 2
    pin_memory: bool = True
    drop_last: bool = True  # Keep shapes consistent for FP16 AutoCast
    
    # =============================================================================
    # MODEL ARCHITECTURE
    # =============================================================================
    # Image dimensions
    image_size: int = 8      # SD-VAE 64->8 (matches your new data)
    patch_size: int = 1      # Keep patch_size=1 as you requested
    n_channels: int = 4      # Latent channels
    
    # Transformer dimensions
    embed_dim: int = 384
    mlp_multiplier: int = 4
    n_layers: int = 12
    n_heads: Optional[int] = None  # Auto-calculated as embed_dim // 64 if None
    cross_heads: int = 4
    
    # Embeddings
    noise_embed_dims: int = 128
    text_emb_size: int = 10  # Number of classes for FMNIST
    
    # Regularization
    dropout_prob: float = 0.1
    use_flash_attention: bool = True
    
    # =============================================================================
    # TRAINING CONFIGURATION
    # =============================================================================
    # Learning rate and optimization
    lr: float = 1e-3
    total_steps: int = 1000
    ema_decay: float = 0.999
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Logging and saving
    log_interval: int = 100
    save_interval: int = 500
    val_every: int = 500          
    val_max_batches: int = 50          
    save_best: bool = True
    
    # =============================================================================
    # DIFFUSION CONFIGURATION
    # =============================================================================
    # Continuous timesteps (beta distribution parameters)
    beta_a: float = 1.0               
    beta_b: float = 2.5
    
    # Inference
    num_inference_steps: int = 25
    noise_start: float = 0.99
    noise_end: float = 0.01
    
    # =============================================================================
    # VAE CONFIGURATION
    # =============================================================================
    vae_path: str = "stabilityai/sd-vae-ft-mse"
    vae_latent_scale: float = 0.18215  # Scaling factor for VAE latents
    
    # =============================================================================
    # TEXT ENCODING CONFIGURATION
    # =============================================================================
    text_encoding_type: str = "one_hot"  # "one_hot", "learned", "pretrained"
    text_dropout_prob: float = 0.1  # For classifier-free guidance
    num_classes: int = 10  # FMNIST classes
    
    # For learned embeddings
    text_embed_dim: Optional[int] = None  # If None, uses embed_dim
    text_projection_layers: int = 1  # Number of projection layers
    
    # For pretrained text encoders
    pretrained_text_model: Optional[str] = None  # e.g., "openai/clip-vit-base-patch32"
    text_model_max_length: int = 77
    
    # =============================================================================
    # HARDWARE AND PRECISION
    # =============================================================================
    device: Optional[str] = None  # "cuda", "cpu", or None for auto-detect
    use_amp: bool = True  # Automatic Mixed Precision
    amp_dtype: str = "float16"  # "float16", "bfloat16"
    compile_model: bool = False  # Use torch.compile() if available
    
    # =============================================================================
    # LOGGING AND RESUME
    # =============================================================================
    run_id: Optional[str] = None
    resume_from: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        # Auto-calculate number of heads if not specified
        if self.n_heads is None:
            self.n_heads = self.embed_dim // 64
        
        # Auto-calculate text embedding dimension if not specified
        if self.text_embed_dim is None:
            self.text_embed_dim = self.embed_dim
        
        # Validate patch size compatibility
        if self.image_size % self.patch_size != 0:
            raise ValueError(f"image_size ({self.image_size}) must be divisible by patch_size ({self.patch_size})")
        
        # Validate embedding dimension compatibility
        if self.embed_dim % self.n_heads != 0:
            raise ValueError(f"embed_dim ({self.embed_dim}) must be divisible by n_heads ({self.n_heads})")
        
        # Validate text encoding type
        valid_text_types = ["one_hot", "learned", "pretrained"]
        if self.text_encoding_type not in valid_text_types:
            raise ValueError(f"text_encoding_type must be one of {valid_text_types}")
        
        # Validate noise levels
        if not (0 < self.noise_end < self.noise_start < 1):
            raise ValueError("noise levels must satisfy: 0 < noise_end < noise_start < 1")
    
    @property
    def num_patches(self) -> int:
        """Calculate number of patches"""
        return (self.image_size // self.patch_size) ** 2
    
    @property
    def patch_dim(self) -> int:
        """Calculate patch dimension"""
        return self.n_channels * self.patch_size * self.patch_size
    
    @property
    def total_params(self) -> int:
        """Estimate total parameters (approximate)"""
        # This is a rough estimate based on the architecture
        embed_params = self.embed_dim * self.embed_dim * 4  # QKV + output projections
        layer_params = self.n_layers * (embed_params * 2 + self.embed_dim * self.embed_dim * self.mlp_multiplier * 2)
        patch_params = self.n_channels * self.embed_dim * self.patch_size * self.patch_size * 2  # embed + unembed
        time_params = self.noise_embed_dims * self.embed_dim * 2 + self.embed_dim * self.embed_dim
        text_params = self.text_emb_size * self.embed_dim + self.embed_dim * self.embed_dim
        
        return embed_params + layer_params + patch_params + time_params + text_params
