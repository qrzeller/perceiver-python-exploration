"""
Train Perceiver model for VAD prediction from DEAP dataset

This script trains a PerceiverIO model to predict Valence, Arousal, and Dominance
from multimodal physiological signals (EEG, GSR, EMG, etc.)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: tensorboard not available. Install with: pip install tensorboard")
from tqdm import tqdm
import numpy as np
from datetime import datetime

from perceiver_pytorch import PerceiverIO
from deap_dataloader import create_dataloaders, get_default_channel_groups


class VADPredictor(nn.Module):
    """
    Wrapper for PerceiverIO that predicts VAD values from physiological signals.
    """
    
    def __init__(
        self,
        input_dim: int = 40,  # Number of channels
        dim: int = 64,  # Dimension of sequence encoding
        queries_dim: int = 32,  # Dimension of decoder queries
        depth: int = 4,  # Depth of the network (reduced for fast training)
        num_latents: int = 64,  # Number of latent bottleneck (reduced)
        latent_dim: int = 128,  # Latent dimension (reduced)
        cross_heads: int = 1,  # Cross-attention heads
        latent_heads: int = 4,  # Latent self-attention heads (reduced)
        cross_dim_head: int = 64,
        latent_dim_head: int = 64,
        output_dim: int = 3,  # VAD (3 dimensions)
    ):
        super().__init__()
        
        # Input projection: map from input_dim (channels) to dim
        self.input_proj = nn.Linear(input_dim, dim)
        
        # Perceiver model
        self.perceiver = PerceiverIO(
            dim=dim,
            queries_dim=queries_dim,
            logits_dim=64,  # Intermediate dimension
            depth=depth,
            num_latents=num_latents,
            latent_dim=latent_dim,
            cross_heads=cross_heads,
            latent_heads=latent_heads,
            cross_dim_head=cross_dim_head,
            latent_dim_head=latent_dim_head,
            weight_tie_layers=False,
            seq_dropout_prob=0.1,
        )
        
        # Output head: map from logits to VAD predictions
        self.output_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_dim),
        )
        
        # Learnable query tokens (we use a single query for aggregation)
        self.query = nn.Parameter(torch.randn(1, queries_dim))
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim) - physiological signals
        
        Returns:
            (batch, 3) - VAD predictions
        """
        batch_size = x.shape[0]
        
        # Project input to model dimension
        x = self.input_proj(x)  # (batch, seq_len, dim)
        
        # Expand query for batch
        queries = self.query.expand(batch_size, -1, -1)  # (batch, 1, queries_dim)
        
        # Process through Perceiver
        logits = self.perceiver(x, queries=queries)  # (batch, 1, 64)
        
        # Squeeze and predict VAD
        logits = logits.squeeze(1)  # (batch, 64)
        vad = self.output_head(logits)  # (batch, 3)
        
        return vad


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for data, labels in pbar:
        data = data.to(device)  # (batch, seq_len, n_channels)
        labels = labels.to(device)  # (batch, 3)
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(data)  # (batch, 3)
        
        # Compute loss
        loss = criterion(predictions, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in tqdm(val_loader, desc="Validating"):
            data = data.to(device)
            labels = labels.to(device)
            
            # Forward pass
            predictions = model(data)
            
            # Compute loss
            loss = criterion(predictions, labels)
            
            total_loss += loss.item()
            num_batches += 1
            
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Compute per-dimension metrics
    mae_valence = np.mean(np.abs(all_predictions[:, 0] - all_labels[:, 0]))
    mae_arousal = np.mean(np.abs(all_predictions[:, 1] - all_labels[:, 1]))
    mae_dominance = np.mean(np.abs(all_predictions[:, 2] - all_labels[:, 2]))
    
    metrics = {
        'loss': total_loss / num_batches,
        'mae_valence': mae_valence,
        'mae_arousal': mae_arousal,
        'mae_dominance': mae_dominance,
        'mae_overall': (mae_valence + mae_arousal + mae_dominance) / 3,
    }
    
    return metrics


def main():
    # Configuration
    config = {
        # Data
        'data_dir': './dataset/deap-dataset/data_preprocessed_python',
        'sequence_length': 2048,  # ~16 seconds at 128 Hz (reduced for fast training)
        'stride': 2048,  # Non-overlapping windows
        'use_channels': None,  # Use all 40 channels
        
        # Model
        'input_dim': 40,  # All channels
        'dim': 64,  # Reduced from 128 for fast training
        'queries_dim': 32,
        'depth': 4,  # Reduced from 6
        'num_latents': 64,  # Reduced from 256
        'latent_dim': 128,  # Reduced from 512
        'cross_heads': 1,
        'latent_heads': 4,  # Reduced from 8
        'cross_dim_head': 64,
        'latent_dim_head': 64,
        
        # Training
        'batch_size': 8,  # Reduced for fast training
        'num_epochs': 50,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'num_workers': 0,  # Set to 0 for debugging, increase for speed
        
        # Splits (subject-independent)
        'train_subjects': list(range(1, 23)),  # 22 subjects
        'val_subjects': list(range(23, 28)),   # 5 subjects
        'test_subjects': list(range(28, 33)),  # 5 subjects
    }
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'./outputs/deap_vad_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    import json
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # TensorBoard
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))
    else:
        writer = None
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=config['data_dir'],
        train_subjects=config['train_subjects'],
        val_subjects=config['val_subjects'],
        test_subjects=config['test_subjects'],
        sequence_length=config['sequence_length'],
        stride=config['stride'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        use_channels=config['use_channels'],
    )
    
    # Create model
    print("\nCreating model...")
    model = VADPredictor(
        input_dim=config['input_dim'],
        dim=config['dim'],
        queries_dim=config['queries_dim'],
        depth=config['depth'],
        num_latents=config['num_latents'],
        latent_dim=config['latent_dim'],
        cross_heads=config['cross_heads'],
        latent_heads=config['latent_heads'],
        cross_dim_head=config['cross_dim_head'],
        latent_dim_head=config['latent_dim_head'],
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Logging
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val MAE - Valence: {val_metrics['mae_valence']:.4f}, "
              f"Arousal: {val_metrics['mae_arousal']:.4f}, "
              f"Dominance: {val_metrics['mae_dominance']:.4f}, "
              f"Overall: {val_metrics['mae_overall']:.4f}")
        
        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            writer.add_scalar('MAE/valence', val_metrics['mae_valence'], epoch)
            writer.add_scalar('MAE/arousal', val_metrics['mae_arousal'], epoch)
            writer.add_scalar('MAE/dominance', val_metrics['mae_dominance'], epoch)
            writer.add_scalar('MAE/overall', val_metrics['mae_overall'], epoch)
        
        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_metrics': val_metrics,
                'config': config,
            }, os.path.join(output_dir, 'best_model.pth'))
            print(f"Saved best model (val_loss: {best_val_loss:.4f})")
    
    # Test on best model
    print("\nEvaluating on test set...")
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test MAE - Valence: {test_metrics['mae_valence']:.4f}, "
          f"Arousal: {test_metrics['mae_arousal']:.4f}, "
          f"Dominance: {test_metrics['mae_dominance']:.4f}, "
          f"Overall: {test_metrics['mae_overall']:.4f}")
    
    # Save test results
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    if writer is not None:
        writer.close()
    print(f"\nTraining complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
