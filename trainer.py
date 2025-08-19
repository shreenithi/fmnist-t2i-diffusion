import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from .configs import T2IConfig
from .data import setup_data_loaders, create_text_embeddings
from .models import FMNISTT2IModel
from .diffusion import generate_image
from .utils import setup_logging, save_checkpoint, resume_from_checkpoint


class FMNISTT2ITrainer:
    """Trainer for FMNIST Text-to-Image Transformer Latent Diffusion model
    
    This trainer handles the complete training loop including data loading,
    model training, validation, and checkpointing.
    
    Args:
        cfg: Configuration object containing all training parameters
    """
    
    def __init__(self, cfg: T2IConfig):
        self.config = cfg
        
        # Device setup
        if cfg.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg.device)
        
        # Setup logging
        self.tb_writer, self.run_id, self.global_step, self.pbar = setup_logging(cfg)
        
        # Setup data
        self.train_loader, self.val_loader = setup_data_loaders(cfg)
        
        # Create model
        self.model = FMNISTT2IModel(cfg).to(self.device).float()
        
        # Compile model if requested
        if cfg.compile_model and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model)
                print("Model compiled successfully")
            except Exception as e:
                print(f"Model compilation failed: {e}")
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=cfg.lr, 
            betas=cfg.betas, 
            eps=cfg.eps,
            weight_decay=cfg.weight_decay
        )

        # EMA model
        self.ema_model = None
        if cfg.ema_decay > 0:
            self.ema_model = FMNISTT2IModel(cfg).to(self.device).float()
            for p, ep in zip(self.model.parameters(), self.ema_model.parameters()):
                ep.data.copy_(p.data)

        # Gradient scaler for AMP
        self.scaler = torch.amp.GradScaler(self.device) if cfg.use_amp else None

        # Training state
        self.best_val = float('inf')
        self.current_lr = cfg.lr

        # Resume from checkpoint if specified
        if cfg.resume_from:
            self.global_step = resume_from_checkpoint(
                cfg.resume_from, self.model, self.ema_model, 
                self.optimizer, self.scaler, self.device
            )
            self.pbar.n = self.global_step

    def update_ema(self):
        """Update EMA model parameters using exponential moving average"""
        if self.ema_model is None: 
            return
        with torch.no_grad():
            for p, ep in zip(self.model.parameters(), self.ema_model.parameters()):
                ep.data.mul_(self.config.ema_decay).add_(p.data, alpha=1 - self.config.ema_decay)

    def train_step(self, latents, text_emb):
        """Single training step
        
        Args:
            latents: Input latent representations
            text_emb: Text embeddings/labels
            
        Returns:
            Training loss for this step
        """
        B = latents.size(0)
        
        # Ensure input tensors are float32
        latents = latents.float()
        text_emb = text_emb.float()
        
        # Continuous timesteps: sample noise levels from beta distribution
        noise_level = torch.tensor(
            np.random.beta(self.config.beta_a, self.config.beta_b, B), 
            device=self.device, dtype=torch.float32
        )
        signal_level = 1 - noise_level
        
        # Add noise based on continuous noise level
        noise = torch.randn_like(latents, dtype=torch.float32)
        noisy_latents = (noise_level.view(-1, 1, 1, 1) * noise + 
                       signal_level.view(-1, 1, 1, 1) * latents)
        
        # For continuous timesteps, we predict the clean image directly
        target = latents

        # Label dropout for classifier-free guidance
        if self.config.dropout_prob > 0:
            mask = torch.rand(B, device=self.device) < self.config.dropout_prob
            text_emb_cond = torch.where(mask[:, None], torch.zeros_like(text_emb), text_emb)
        else:
            text_emb_cond = text_emb

        # Forward pass with optional AMP
        if self.config.use_amp and self.scaler is not None:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                pred = self.model(noisy_latents, text_emb_cond, noise_level)
                loss = F.mse_loss(pred, target, reduction="mean") / self.config.accumulation_steps
            
            self.scaler.scale(loss).backward()
        else:
            pred = self.model(noisy_latents, text_emb_cond, noise_level)
            loss = F.mse_loss(pred, target, reduction="mean") / self.config.accumulation_steps
            loss.backward()
        
        return loss

    @torch.no_grad()
    def validate(self):
        """Validation step"""
        if self.val_loader is None: 
            return None
        
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        for b, (latents, text_emb) in enumerate(self.val_loader):
            if b >= self.config.val_max_batches: 
                break
                
            latents = latents.to(self.device, non_blocking=True)
            text_emb = text_emb.to(self.device, non_blocking=True)

            # Ensure float32
            latents = latents.float()
            text_emb = text_emb.float()

            B = latents.size(0)
            
            # Continuous timesteps: sample noise levels from beta distribution
            noise_level = torch.tensor(
                np.random.beta(self.config.beta_a, self.config.beta_b, B), 
                device=self.device, dtype=torch.float32
            )
            signal_level = 1 - noise_level
            
            # Add noise based on continuous noise level
            noise = torch.randn_like(latents, dtype=torch.float32)
            noisy_latents = (noise_level.view(-1, 1, 1, 1) * noise + 
                           signal_level.view(-1, 1, 1, 1) * latents)
            
            # For continuous timesteps, we predict the clean image directly (same as training)
            target = latents

            # NOTE: no label-dropout in validation
            if self.config.use_amp:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    pred = self.model(noisy_latents, text_emb, noise_level)
                    loss = F.mse_loss(pred, target, reduction="mean").item()
            else:
                pred = self.model(noisy_latents, text_emb, noise_level)
                loss = F.mse_loss(pred, target, reduction="mean").item()
                
            total_loss += loss
            n_batches += 1

        self.model.train()
        return (total_loss / max(1, n_batches)) if n_batches > 0 else None

    def log_metrics(self, loss, total_norm):
        """Log training metrics to TensorBoard"""
        lr = self.optimizer.param_groups[0]["lr"]
        self.tb_writer.add_scalar("train/loss", loss.item(), self.global_step)
        self.tb_writer.add_scalar("train/lr", lr, self.global_step)
        self.tb_writer.add_scalar("train/grad_norm", total_norm, self.global_step)
        
        if torch.cuda.is_available():
            self.tb_writer.add_scalar("train/gpu_memory", torch.cuda.memory_allocated() / 1024**3, self.global_step)
            self.tb_writer.add_scalar("train/gpu_memory_reserved", torch.cuda.memory_reserved() / 1024**3, self.global_step)

    @torch.no_grad()
    def evaluate_and_log(self):
        """Generate and log sample images"""
        classes = torch.arange(self.config.num_classes, device=self.device)
        
        if self.config.text_encoding_type == "one_hot":
            labels = F.one_hot(classes, num_classes=self.config.num_classes).float()
        elif self.config.text_encoding_type == "learned":
            labels = classes  # Use class indices directly
        
        # Generate samples
        samples = generate_image(
            self.ema_model if self.ema_model is not None else self.model,
            labels=labels, 
            num_samples=labels.size(0),
            config=self.config
        )
        
        try:
            from .diffusion import decode_latents_to_images
            images = decode_latents_to_images(samples, config=self.config, device=self.device)
            self.tb_writer.add_images("eval/by_class", images, self.global_step, dataformats="NCHW")
        except Exception as e:
            print(f"VAE decoding failed: {e}")
            # Log latent statistics instead of trying to log as images
            self.tb_writer.add_scalar("eval/latent_mean", samples.mean().item(), self.global_step)
            self.tb_writer.add_scalar("eval/latent_std", samples.std().item(), self.global_step)
            self.tb_writer.add_scalar("eval/latent_min", samples.min().item(), self.global_step)
            self.tb_writer.add_scalar("eval/latent_max", samples.max().item(), self.global_step)

    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config.total_steps} steps")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Estimated total params: {self.config.total_params:,}")
        print(f"Learning rate: {self.config.lr}")
        print(f"Batch size: {self.config.batch_size} (effective: {self.config.batch_size * self.config.accumulation_steps})")
        print(f"Text encoding: {self.config.text_encoding_type}")
        print(f"VAE path: {self.config.vae_path}")

        self.model.train()
        
        while self.global_step < self.config.total_steps:
            for batch_idx, (latents, text_emb) in enumerate(self.train_loader):
                latents = latents.to(self.device, non_blocking=True)
                text_emb = text_emb.to(self.device, non_blocking=True)

                loss = self.train_step(latents, text_emb)

                if (batch_idx + 1) % self.config.accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                    # Unscale gradients if using AMP
                    if self.config.use_amp and self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)

                    # Calculate gradient norm
                    total_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            total_norm += p.grad.data.norm(2).item() ** 2
                    total_norm = total_norm ** 0.5

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm)
                    
                    # Optimizer step
                    if self.config.use_amp and self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                        
                    self.optimizer.zero_grad()
                    self.update_ema()

                    # Update progress
                    self.global_step += 1
                    self.pbar.update(1)
                    self.pbar.set_postfix(loss=f"{(loss.item()*self.config.accumulation_steps):.4f}")

                    # Logging
                    if self.global_step % self.config.log_interval == 0:
                        self.log_metrics(loss, total_norm)

                    # Validation and checkpointing
                    if self.val_loader is not None and self.global_step % self.config.val_every == 0:
                        vloss = self.validate()
                        if vloss is not None:
                            self.tb_writer.add_scalar("val/loss", vloss, self.global_step)
                            if self.config.save_best and vloss < self.best_val:
                                self.best_val = vloss
                                save_checkpoint(self.model, self.ema_model, self.optimizer, 
                                             self.scaler, self.global_step, loss.item(), 
                                             self.config, self.run_id, tag="best")

                    if self.global_step % self.config.save_interval == 0:
                        self.evaluate_and_log()
                        save_checkpoint(self.model, self.ema_model, self.optimizer, 
                                     self.scaler, self.global_step, loss.item(), 
                                     self.config, self.run_id)

                    if self.global_step >= self.config.total_steps:
                        break
                        
            if self.global_step >= self.config.total_steps:
                break

        # Final validation and saving
        if self.val_loader is not None:
            v = self.validate()
            if v is not None:
                self.tb_writer.add_scalar("val/loss", v, self.global_step)
                
        self.evaluate_and_log()
        save_checkpoint(self.model, self.ema_model, self.optimizer, 
                       self.scaler, self.global_step, loss.item(), 
                       self.config, self.run_id)
        
        self.pbar.close()
        self.tb_writer.close()
        print("Training completed!")
