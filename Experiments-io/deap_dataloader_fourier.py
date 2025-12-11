"""
Advanced DEAP DataLoader with Fourier Feature Encoding

This module extends the basic dataloader with Fourier feature encoding,
which can help the Perceiver model better capture frequency information
from the physiological signals.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
from deap_dataloader import DEAPDataset


class FourierFeatureEncoding(nn.Module):
    """
    Fourier feature encoding for positional/temporal information.
    
    Maps input positions to higher dimensional space using random Fourier features.
    This helps neural networks learn high-frequency functions.
    
    Reference: "Fourier Features Let Networks Learn High Frequency Functions"
    https://arxiv.org/abs/2006.10739
    """
    
    def __init__(
        self,
        num_features: int = 64,
        scale: float = 1.0,
    ):
        super().__init__()
        self.num_features = num_features
        self.scale = scale
        
        # Random Fourier feature matrix (not trainable)
        self.register_buffer(
            'B',
            torch.randn(1, num_features // 2) * scale
        )
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: (batch, seq_len, 1) - normalized positions [0, 1]
        
        Returns:
            features: (batch, seq_len, num_features) - Fourier features
        """
        # positions: (batch, seq_len, 1)
        # B: (1, num_features // 2)
        
        x_proj = 2 * np.pi * positions * self.B  # (batch, seq_len, num_features // 2)
        
        # Concatenate sin and cos
        features = torch.cat([
            torch.sin(x_proj),
            torch.cos(x_proj)
        ], dim=-1)  # (batch, seq_len, num_features)
        
        return features


class DEAPDatasetWithFourier(DEAPDataset):
    """
    Extended DEAP dataset that includes Fourier feature encoding.
    
    This adds temporal position encoding using Fourier features,
    which can help the model learn temporal patterns better.
    """
    
    def __init__(
        self,
        *args,
        use_fourier: bool = True,
        num_fourier_features: int = 64,
        fourier_scale: float = 1.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.use_fourier = use_fourier
        
        if use_fourier:
            self.fourier_encoder = FourierFeatureEncoding(
                num_features=num_fourier_features,
                scale=fourier_scale,
            )
            print(f"  - Using Fourier features: {num_fourier_features}")
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            data: (sequence_length, n_channels) tensor
            label: (3,) tensor with [Valence, Arousal, Dominance]
            fourier_features: (sequence_length, num_fourier_features) if use_fourier else None
        """
        data, label = super().__getitem__(idx)
        # data: (sequence_length, n_channels)
        
        if not self.use_fourier:
            return data, label, None
        
        # Create temporal positions
        seq_len = data.shape[0]
        positions = torch.linspace(0, 1, seq_len).unsqueeze(-1)  # (seq_len, 1)
        positions = positions.unsqueeze(0)  # (1, seq_len, 1)
        
        # Encode positions with Fourier features
        with torch.no_grad():
            fourier_features = self.fourier_encoder(positions).squeeze(0)  # (seq_len, num_fourier_features)
        
        return data, label, fourier_features


class VADPredictorWithFourier(nn.Module):
    """
    Enhanced VAD predictor that uses Fourier features.
    
    Concatenates Fourier temporal features with channel data before
    feeding into the Perceiver model.
    """
    
    def __init__(
        self,
        input_dim: int = 40,
        num_fourier_features: int = 64,
        dim: int = 64,
        queries_dim: int = 32,
        depth: int = 4,
        num_latents: int = 64,
        latent_dim: int = 128,
        cross_heads: int = 1,
        latent_heads: int = 4,
        cross_dim_head: int = 64,
        latent_dim_head: int = 64,
        output_dim: int = 3,
        use_fourier: bool = True,
    ):
        super().__init__()
        
        self.use_fourier = use_fourier
        
        # Input dimension is channels + Fourier features
        total_input_dim = input_dim + (num_fourier_features if use_fourier else 0)
        
        # Input projection: map from total_input_dim to dim
        self.input_proj = nn.Linear(total_input_dim, dim)
        
        # Import here to avoid circular dependency
        from perceiver_pytorch import PerceiverIO
        
        # Perceiver model
        self.perceiver = PerceiverIO(
            dim=dim,
            queries_dim=queries_dim,
            logits_dim=64,
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
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_dim),
        )
        
        # Learnable query token
        self.query = nn.Parameter(torch.randn(1, queries_dim))
    
    def forward(self, x: torch.Tensor, fourier_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim) - physiological signals
            fourier_features: (batch, seq_len, num_fourier_features) - optional Fourier features
        
        Returns:
            (batch, 3) - VAD predictions
        """
        batch_size = x.shape[0]
        
        # Concatenate Fourier features if provided
        if self.use_fourier and fourier_features is not None:
            x = torch.cat([x, fourier_features], dim=-1)  # (batch, seq_len, input_dim + num_fourier_features)
        
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


# Example usage
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    
    print("Testing Fourier Feature DataLoader\n")
    
    # Create dataset with Fourier features
    dataset = DEAPDatasetWithFourier(
        data_dir='./dataset/deap-dataset/data_preprocessed_python',
        subject_ids=[1, 2],
        sequence_length=2048,
        stride=2048,
        normalize=True,
        use_channels=None,
        use_fourier=True,
        num_fourier_features=64,
        fourier_scale=1.0,
    )
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # Test data loading
    print("Testing data loading:")
    for batch_idx, batch in enumerate(dataloader):
        if len(batch) == 3:
            data, labels, fourier = batch
            print(f"Batch {batch_idx}:")
            print(f"  Data shape: {data.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Fourier features shape: {fourier.shape}")
        else:
            data, labels = batch
            print(f"Batch {batch_idx}:")
            print(f"  Data shape: {data.shape}")
            print(f"  Labels shape: {labels.shape}")
        
        if batch_idx >= 2:
            break
    
    # Test model
    print("\nTesting model with Fourier features:")
    device = torch.device('cpu')
    model = VADPredictorWithFourier(
        input_dim=40,
        num_fourier_features=64,
        use_fourier=True,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params:,}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                data, labels, fourier = batch
                data = data.to(device)
                fourier = fourier.to(device)
                predictions = model(data, fourier)
            else:
                data, labels = batch
                data = data.to(device)
                predictions = model(data)
            
            print(f"\nForward pass test:")
            print(f"  Input shape: {data.shape}")
            print(f"  Output shape: {predictions.shape}")
            print(f"  Sample prediction: {predictions[0]}")
            print(f"  Sample label: {labels[0]}")
            break
    
    print("\n✓ Fourier feature implementation working!")
