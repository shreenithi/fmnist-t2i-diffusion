import os
import json
import math
from datetime import datetime
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from safetensors.torch import load_file, save_file
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .configs import T2IConfig


def setup_logging(cfg: T2IConfig):
    """Setup TensorBoard logging and progress bar"""
    run_id = cfg.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    full_logdir = f"{cfg.tb_logdir}/run_{run_id}"
    print(f"TensorBoard logs -> {full_logdir}")
    tb_writer = SummaryWriter(full_logdir)
    global_step = 0
    pbar = tqdm(total=cfg.total_steps, desc="Training", unit="step")
    return tb_writer, run_id, global_step, pbar


def save_checkpoint(model, ema_model, optimizer, scaler, step, loss, cfg: T2IConfig, run_id: str, tag=None):
    """Save model checkpoint with comprehensive metadata"""
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    name = f"{tag}_" if tag else ""
    
    # Save main model
    model_path = os.path.join(cfg.ckpt_dir, f"{name}model_{step}.safetensors")
    save_file(model.state_dict(), model_path)

    # Save EMA model if exists
    ema_path = None
    if ema_model is not None:
        ema_path = os.path.join(cfg.ckpt_dir, f"{name}ema_{step}.safetensors")
        save_file(ema_model.state_dict(), ema_path)

    # Save training state
    training_state = {
        "step": step,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "loss": loss,
        "config": cfg,
        "run_id": run_id,
        "model_path": model_path,
        "ema_path": ema_path,
        "timestamp": datetime.now().isoformat(),
    }
    training_path = os.path.join(cfg.ckpt_dir, f"{name}training_state_{step}.pt")
    torch.save(training_state, training_path)

    # Save latest checkpoint info
    latest_info = {
        "latest_model": model_path, 
        "latest_ema": ema_path,
        "latest_training_state": training_path, 
        "step": step, 
        "loss": loss,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(os.path.join(cfg.ckpt_dir, "latest_checkpoint.json"), "w") as f:
        json.dump(latest_info, f, indent=2)


def resume_from_checkpoint(checkpoint_path, model, ema_model, optimizer, scaler, device):
    """Resume training from a checkpoint"""
    print(f"Resuming from checkpoint: {checkpoint_path}")
    
    # Add safe globals for loading config
    from torch.serialization import add_safe_globals
    from .configs import T2IConfig
    add_safe_globals([T2IConfig])
    
    # Load training state
    st = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_path = st.get("model_path", checkpoint_path.replace("training_state_", "model_").replace(".pt", ".safetensors"))
    
    # Load model weights
    if os.path.exists(model_path):
        model.load_state_dict(load_file(model_path))
        print(f"Model loaded from: {model_path}")
    else:
        print(f"Warning: Model file not found: {model_path}")
    
    # Load EMA model weights
    if ema_model is not None and st.get("ema_path") and os.path.exists(st["ema_path"]):
        ema_model.load_state_dict(load_file(st["ema_path"]))
        print(f"EMA model loaded from: {st['ema_path']}")
    
    # Load optimizer state
    optimizer.load_state_dict(st["optimizer_state_dict"])
    print("Optimizer state restored")
    
    # Load scaler state
    if scaler is not None and st.get("scaler_state_dict") is not None:
        scaler.load_state_dict(st["scaler_state_dict"])
        print("Gradient scaler state restored")
    
    global_step = st["step"]
    print(f"Resumed from step {global_step} (last loss: {st.get('loss','N/A')})")
    return global_step


def get_model_info(model, cfg: T2IConfig):
    """Get model information and parameter counts"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_config": {
            "embed_dim": cfg.embed_dim,
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "n_channels": cfg.n_channels,
            "image_size": cfg.image_size,
            "patch_size": cfg.patch_size,
        }
    }
    
    return info


def log_model_architecture(model, cfg: T2IConfig, tb_writer, step=0):
    """Log model architecture information to TensorBoard"""
    model_info = get_model_info(model, cfg)
    
    # Log parameter counts
    tb_writer.add_scalar("model/total_parameters", model_info["total_parameters"], step)
    tb_writer.add_scalar("model/trainable_parameters", model_info["trainable_parameters"], step)
    
    # Log model config
    for key, value in model_info["model_config"].items():
        tb_writer.add_scalar(f"model/config/{key}", value, step)
    
    print(f"Model architecture logged to TensorBoard at step {step}")
    print(f"Total parameters: {model_info['total_parameters']:,}")
    print(f"Trainable parameters: {model_info['trainable_parameters']:,}")
