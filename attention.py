import torch
import torch.nn as nn
from typing import Optional

# Optional FlashAttention
try:
    from flash_attn.flash_attn_interface import flash_attn_func
    _HAS_FLASH = True
except ImportError:
    flash_attn_func = None
    _HAS_FLASH = False


class FlashSelfAttention(nn.Module):
    """FlashAttention-based self-attention module for improved performance
    
    This module uses FlashAttention when available for faster training and inference.
    Falls back to standard attention if FlashAttention is not available.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        """Forward pass with FlashAttention
        
        Args:
            x: Input tensor of shape (B, N, C)
            causal: Whether to use causal attention mask
            
        Returns:
            Output tensor of shape (B, N, C)
            
        Raises:
            ValueError: If FlashAttention is not available or inputs are not on CUDA
        """
        if not (x.is_cuda and _HAS_FLASH):
            raise ValueError("FlashAttention requires CUDA and flash-attn package; set use_flash_attention=False")
        
        B, N, C = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim)
        
        out = flash_attn_func(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            softmax_scale=None, 
            causal=causal
        ).view(B, N, C)
        
        return self.out_proj(out)


class FlashCrossAttention(nn.Module):
    """FlashAttention-based cross-attention module for improved performance
    
    This module uses FlashAttention for efficient cross-attention between
    input sequences and context sequences.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        context_dim: Context sequence embedding dimension (defaults to embed_dim)
        dropout: Dropout probability
    """
    def __init__(self, embed_dim: int, num_heads: int, context_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if context_dim is None:
            context_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(context_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(context_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, context: torch.Tensor, causal: bool = False) -> torch.Tensor:
        """Forward pass with FlashAttention cross-attention
        
        Args:
            x: Input tensor of shape (B, N, C)
            context: Context tensor of shape (B, M, context_dim)
            causal: Whether to use causal attention mask
            
        Returns:
            Output tensor of shape (B, N, C)
            
        Raises:
            ValueError: If FlashAttention is not available or inputs are not on CUDA
        """
        if not (x.is_cuda and context.is_cuda and _HAS_FLASH):
            raise ValueError("FlashAttention requires CUDA and flash-attn package; set use_flash_attention=False")
        
        B, N, _ = x.shape
        M = context.shape[1]
        
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(context).view(B, M, self.num_heads, self.head_dim)
        v = self.v_proj(context).view(B, M, self.num_heads, self.head_dim)
        
        out = flash_attn_func(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            softmax_scale=None, 
            causal=causal
        ).view(B, N, self.embed_dim)
        
        return self.out_proj(out)
