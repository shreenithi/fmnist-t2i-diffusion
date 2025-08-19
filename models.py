import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .attention import FlashSelfAttention, FlashCrossAttention
from .configs import T2IConfig


class PatchEmbed(nn.Module):
    """Embed image patches into token sequences
    
    Args:
        in_ch: Number of input channels
        embed_dim: Embedding dimension for tokens
        patch_size: Size of each patch (assumes square patches)
    """
    def __init__(self, in_ch: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                 # (B, E, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)  # (B, N, E)
        return x


class PatchUnembed(nn.Module):
    """Convert token sequences back to image patches
    
    Args:
        embed_dim: Embedding dimension of input tokens
        out_ch: Number of output channels
        patch_size: Size of each patch
        img_size: Final image size
    """
    def __init__(self, embed_dim: int, out_ch: int, patch_size: int, img_size: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_ch = out_ch
        self.patch_size = patch_size
        self.img_size = img_size
        self.patch_dim = out_ch * patch_size * patch_size
        self.linear = nn.Linear(embed_dim, self.patch_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        x = self.linear(x)
        grid = self.img_size // self.patch_size
        x = x.view(B, grid, grid, self.out_ch, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, self.out_ch, self.img_size, self.img_size)
        return x


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding for timesteps
    
    Args:
        emb_dim: Embedding dimension
        min_freq: Minimum frequency for sinusoidal components
        max_freq: Maximum frequency for sinusoidal components
    """
    def __init__(self, emb_dim: int, min_freq: float = 1.0, max_freq: float = 1000.0):
        super().__init__()
        freqs = torch.exp(torch.linspace(np.log(min_freq), np.log(max_freq), emb_dim // 2))
        self.register_buffer('frequencies', 2 * np.pi * freqs)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        sin = torch.sin(t * self.frequencies)
        cos = torch.cos(t * self.frequencies)
        return torch.cat([sin, cos], dim=-1)


class MLPSepConv(nn.Module):
    """MLP with depthwise separable convolution
    
    Args:
        dim: Input dimension
        mlp_ratio: Multiplier for hidden dimension
        dropout: Dropout probability
    """
    def __init__(self, dim: int, mlp_ratio: int, dropout: float = 0.0):
        super().__init__()
        hidden = dim * mlp_ratio
        self.fc1 = nn.Linear(dim, hidden)
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        h = self.fc1(x)
        h = h.transpose(1, 2).view(B, -1, H, W)
        h = self.dw(h)
        h = h.flatten(2).transpose(1, 2)
        h = self.act(h)
        return self.dropout(self.fc2(h))


class DecoderBlock(nn.Module):
    """Transformer decoder block with self-attention and cross-attention
    
    Args:
        dim: Input dimension
        mlp_ratio: MLP expansion ratio
        dropout: Dropout probability
        use_flash: Whether to use FlashAttention
        self_heads: Number of self-attention heads
        cross_heads: Number of cross-attention heads
    """
    def __init__(self, dim: int, mlp_ratio: int, dropout: float, 
                 use_flash: bool = False, self_heads: Optional[int] = None, 
                 cross_heads: Optional[int] = None):
        super().__init__()
        self_heads = self_heads or dim // 64  # Default: 384//64=6
        cross_heads = cross_heads or 4
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        if use_flash:
            self.attn1 = FlashSelfAttention(dim, self_heads, dropout=dropout)
            self.attn2 = FlashCrossAttention(dim, cross_heads, dropout=dropout)
        else:
            self.attn1 = nn.MultiheadAttention(dim, self_heads, batch_first=True, dropout=dropout)
            self.attn2 = nn.MultiheadAttention(dim, cross_heads, batch_first=True, dropout=dropout)
        
        self.mlp = MLPSepConv(dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # Self-attention with residual connection
        residual = x
        x_norm = self.norm1(x)
        if isinstance(self.attn1, FlashSelfAttention):
            attention_output = self.attn1(x_norm)
        else:
            attention_output, _ = self.attn1(x_norm, x_norm, x_norm)
        x = residual + attention_output

        # Cross-attention with residual connection
        residual = x
        x_norm = self.norm2(x)
        if isinstance(self.attn2, FlashCrossAttention):
            attention_output = self.attn2(x_norm, context)
        else:
            attention_output, _ = self.attn2(x_norm, context, context)
        x = residual + attention_output

        # MLP with residual connection
        residual = x
        x = residual + self.mlp(self.norm3(x), H, W)
        return x


class FMNISTT2IModel(nn.Module):
    """FMNIST Text-to-Image Transformer Latent Diffusion Model
    
    This model implements a transformer-based diffusion model that generates
    Fashion MNIST images from text labels using VAE latents.
    
    Args:
        cfg: Configuration object containing all model parameters
    """
    def __init__(self, cfg: T2IConfig):
        super().__init__()
        self.cfg = cfg
        
        # Patch processing
        self.patch_embed = PatchEmbed(cfg.n_channels, cfg.embed_dim, cfg.patch_size)
        self.patch_unembed = PatchUnembed(cfg.embed_dim, cfg.n_channels, cfg.patch_size, cfg.image_size)

        # Positional embedding
        pos = torch.zeros(1, cfg.num_patches, cfg.embed_dim)
        nn.init.trunc_normal_(pos, std=0.02)
        self.pos_embed = nn.Parameter(pos)

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(cfg.noise_embed_dims),
            nn.Linear(cfg.noise_embed_dims, cfg.embed_dim),
            nn.GELU(),
            nn.Linear(cfg.embed_dim, cfg.embed_dim)
        )
        
        # Text embedding
        if cfg.text_encoding_type == "one_hot":
            self.text_proj = nn.Linear(cfg.text_emb_size, cfg.embed_dim)
        elif cfg.text_encoding_type == "learned":
            self.text_embed = nn.Embedding(cfg.num_classes, cfg.text_embed_dim)
            self.text_proj = nn.Linear(cfg.text_embed_dim, cfg.embed_dim)
        elif cfg.text_encoding_type == "pretrained":
            # For pretrained text encoders like CLIP
            raise NotImplementedError("Pretrained text encoders not yet implemented")
        
        self.context_norm = nn.LayerNorm(cfg.embed_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DecoderBlock(
                cfg.embed_dim, 
                cfg.mlp_multiplier, 
                cfg.dropout_prob, 
                use_flash=cfg.use_flash_attention,
                self_heads=cfg.n_heads,
                cross_heads=cfg.cross_heads
            )
            for _ in range(cfg.n_layers)
        ])
        
        # Null token for classifier-free guidance
        self.null_token = nn.Parameter(torch.zeros(1, cfg.embed_dim))

    def forward(self, x: torch.Tensor, text_emb: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        H = W = self.cfg.image_size // self.cfg.patch_size
        
        # Patch embedding + positional embedding
        tokens = self.patch_embed(x) + self.pos_embed  # (B, N, E)

        # Time embedding
        t_scaled = timesteps.float().unsqueeze(-1) * 100.0  # Scale for sinusoidal embedding
        time_tokens = self.time_embed(t_scaled).unsqueeze(1)  # (B, 1, E)
        
        # Text embedding
        if self.cfg.text_encoding_type == "one_hot":
            if text_emb.dim() == 2:  # (B, num_classes) - one-hot labels
                text_tokens = self.text_proj(text_emb).unsqueeze(1)  # (B, 1, E)
            elif text_emb.dim() == 3:  # (B, 1, E) - already projected
                text_tokens = text_emb
            else:
                raise ValueError(f"Unexpected text_emb shape: {text_emb.shape}")
        elif self.cfg.text_encoding_type == "learned":
            if text_emb.dtype in (torch.int32, torch.int64):  # Class indices
                text_tokens = self.text_proj(self.text_embed(text_emb)).unsqueeze(1)  # (B, 1, E)
            else:
                raise ValueError(f"Expected integer class indices for learned embeddings, got {text_emb.dtype}")
        
        # Combine time and text context
        context = torch.cat([time_tokens, text_tokens], dim=1)  # (B, 2, E)
        context = self.context_norm(context)

        # Process through transformer blocks
        hidden = tokens
        for block in self.blocks:
            hidden = block(hidden, context, H, W)

        # Unpatch to get final image
        return self.patch_unembed(hidden)  # (B, n_channels, image_size, image_size)
