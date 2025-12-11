import torch
from perceiver_pytorch import Perceiver

# Test with different configurations
print("Testing Perceiver configurations for 48x48 grayscale images...")

configs = [
    {"num_freq_bands": 4, "max_freq": 4.0},
    {"num_freq_bands": 6, "max_freq": 4.0},
    {"num_freq_bands": 3, "max_freq": 4.0},
]

for i, config in enumerate(configs):
    print(f"\n--- Config {i+1} ---")
    print(f"num_freq_bands: {config['num_freq_bands']}, max_freq: {config['max_freq']}")
    
    try:
        model = Perceiver(
            input_channels=1,
            input_axis=2,
            num_freq_bands=config['num_freq_bands'],
            max_freq=config['max_freq'],
            depth=2,
            num_latents=32,
            latent_dim=128,
            cross_heads=1,
            latent_heads=4,
            cross_dim_head=32,
            latent_dim_head=32,
            num_classes=7,
            attn_dropout=0.,
            ff_dropout=0.,
            weight_tie_layers=False,
            fourier_encode_data=True,
            self_per_cross_attn=1
        )
        
        # Test with 48x48 grayscale image in format (batch, height, width, channels)
        test_img = torch.randn(2, 48, 48, 1)
        print(f"Input shape: {test_img.shape}")
        
        output = model(test_img)
        print(f"✓ Success! Output shape: {output.shape}")
        
    except Exception as e:
        print(f"✗ Failed with error: {e}")

print("\n" + "="*60)
print("Testing without Fourier encoding...")
try:
    model_no_fourier = Perceiver(
        input_channels=1,
        input_axis=2,
        num_freq_bands=6,
        max_freq=4.0,
        depth=2,
        num_latents=32,
        latent_dim=128,
        cross_heads=1,
        latent_heads=4,
        cross_dim_head=32,
        latent_dim_head=32,
        num_classes=7,
        attn_dropout=0.,
        ff_dropout=0.,
        weight_tie_layers=False,
        fourier_encode_data=False,  # Disable Fourier encoding
        self_per_cross_attn=1
    )
    
    test_img = torch.randn(2, 48, 48, 1)
    output = model_no_fourier(test_img)
    print(f"✓ Success without Fourier encoding! Output shape: {output.shape}")
    
except Exception as e:
    print(f"✗ Failed: {e}")
