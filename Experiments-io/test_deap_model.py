"""
Quick test script to verify the Perceiver model works with DEAP data
"""

import torch
from train_deap_vad import VADPredictor
from deap_dataloader import create_dataloaders

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Create small dataloader for testing
print("Creating test dataloader...")
train_loader, val_loader, test_loader = create_dataloaders(
    data_dir='./dataset/deap-dataset/data_preprocessed_python',
    train_subjects=[1, 2],  # Only 2 subjects for quick test
    val_subjects=[23],
    test_subjects=[28],
    sequence_length=2048,
    stride=2048,
    batch_size=4,
    num_workers=0,
    use_channels=None,
)

# Create model with small parameters for fast testing
print("\nCreating model...")
model = VADPredictor(
    input_dim=40,
    dim=64,
    queries_dim=32,
    depth=4,
    num_latents=64,
    latent_dim=128,
    cross_heads=1,
    latent_heads=4,
    cross_dim_head=64,
    latent_dim_head=64,
).to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_params:,}")

# Test forward pass
print("\nTesting forward pass...")
model.eval()
with torch.no_grad():
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Input shape: {data.shape}")  # (batch, seq_len, channels)
        print(f"  Labels shape: {labels.shape}")  # (batch, 3)
        
        # Forward pass
        predictions = model(data)
        
        print(f"  Predictions shape: {predictions.shape}")  # (batch, 3)
        print(f"  Sample prediction: {predictions[0].cpu().numpy()}")
        print(f"  Sample label (VAD): {labels[0].cpu().numpy()}")
        
        # Compute loss
        loss = torch.nn.functional.mse_loss(predictions, labels)
        print(f"  MSE Loss: {loss.item():.4f}")
        
        if batch_idx >= 2:  # Test first 3 batches
            break

print("\n✓ Model test successful!")
print("\nTo start full training, run:")
print("  python Experiments-io/train_deap_vad.py")
