# DEAP VAD Prediction - Project Summary

## What Was Created

A complete pipeline for predicting Valence, Arousal, and Dominance (VAD) from the DEAP dataset using a Perceiver model.

## Files Created

### 1. **deap_dataloader.py** - Core DataLoader
- Loads DEAP dataset from pickle files
- Handles 40 channels of multimodal physiological signals
- Configurable sequence length and windowing
- Subject-independent train/val/test splits
- Per-channel normalization
- Channel group selection (all, EEG only, peripheral only, etc.)

### 2. **train_deap_vad.py** - Training Script
- VADPredictor model (Perceiver-based architecture)
- Complete training loop with validation
- Model checkpointing (saves best model)
- Per-dimension MAE metrics
- TensorBoard logging (optional)
- Optimized parameters for fast testing (~1.6M parameters)

### 3. **test_deap_model.py** - Quick Test
- Rapid verification of the pipeline
- Tests dataloader and model forward pass
- Minimal data loading (2-3 subjects)
- Prints shapes and sample predictions

### 4. **deap_dataloader_fourier.py** - Advanced Features
- Fourier feature encoding for temporal information
- Extended dataset class with positional encoding
- Enhanced model that uses Fourier features
- Helps model learn frequency patterns better

### 5. **README_DEAP.md** - Complete Documentation
- Dataset overview and structure
- Usage instructions
- Customization examples
- Tips for experimentation
- Expected results and benchmarks

## Dataset Structure

**DEAP Dataset:**
- 32 subjects × 40 trials = 1,280 total samples
- 40 channels per trial:
  - 32 EEG channels
  - 8 peripheral signals (EOG, EMG, GSR, Respiration, Plethysmograph, Temperature)
- 8,064 time samples per trial (63 seconds at 128 Hz)
- Labels: Valence, Arousal, Dominance (1-9 scale)

**Data Split (Subject-Independent):**
- Training: Subjects 1-22 (2,640 windows with seq_len=2048)
- Validation: Subjects 23-27 (600 windows)
- Test: Subjects 28-32 (600 windows)

## Model Architecture

```
Input: (batch, 2048, 40)  # 40 channels, 2048 time samples
    ↓
Input Projection: Linear(40 → 64)
    ↓
PerceiverIO:
  - 64 latent tokens
  - 4 layers of cross + self-attention
  - 128-dimensional latent space
    ↓
Query Aggregation: Single learnable query
    ↓
Output Head: MLP(64 → 32 → 3)
    ↓
Output: (batch, 3)  # [Valence, Arousal, Dominance]
```

**Model Size:** ~1.6M parameters (optimized for fast experimentation)

## Key Features

✅ **Multimodal Input**: Handles all 40 channels seamlessly
✅ **Flexible Windowing**: Configurable sequence length and stride
✅ **Subject-Independent**: No data leakage between splits
✅ **Fast Testing**: Reduced model size for quick iteration
✅ **Channel Selection**: Use all channels or specific modalities
✅ **Fourier Features**: Optional temporal encoding (advanced)
✅ **Comprehensive Logging**: Training metrics and checkpoints
✅ **Easy Customization**: Well-documented configuration

## Quick Start

```bash
# Test dataloader
python Experiments-io/deap_dataloader.py

# Test model
python Experiments-io/test_deap_model.py

# Start training
python Experiments-io/train_deap_vad.py
```

## Configuration Highlights

**For Fast Testing (Default):**
- Sequence length: 2048 (~16 seconds)
- Batch size: 8
- Model: 1.6M parameters
- Training time: ~5-10 min/epoch (CPU)

**For Better Performance:**
- Increase sequence length to 4096 or 8064
- Increase model size (dim=128, depth=6, num_latents=256)
- Use more subjects in training
- Add overlapping windows (stride < sequence_length)

## Implementation Details

### Dataloader Features
1. **Lazy Loading**: Only loads subjects when needed
2. **Windowing**: Creates sliding windows from trials
3. **Normalization**: Z-score per channel
4. **Memory Efficient**: Doesn't load all data at once

### Model Features
1. **Perceiver Architecture**: Reduces sequence length via latent bottleneck
2. **Cross-Attention**: Maps input to latents
3. **Self-Attention**: Processes latents
4. **Query Mechanism**: Single learnable query aggregates to VAD

### Training Features
1. **MSE Loss**: Regression on continuous VAD values
2. **AdamW Optimizer**: With weight decay
3. **LR Scheduling**: ReduceLROnPlateau
4. **Metrics**: Per-dimension MAE + overall MAE
5. **Checkpointing**: Saves best model based on validation loss

## Fourier Features (Advanced)

The advanced dataloader adds temporal positional encoding using random Fourier features:

**Benefits:**
- Helps model learn high-frequency patterns
- Improves temporal awareness
- Can enhance performance on physiological signals

**Usage:**
```python
from deap_dataloader_fourier import DEAPDatasetWithFourier, VADPredictorWithFourier

# Create dataset with Fourier features
dataset = DEAPDatasetWithFourier(
    data_dir='./dataset/deap-dataset/data_preprocessed_python',
    use_fourier=True,
    num_fourier_features=64,
)

# Use enhanced model
model = VADPredictorWithFourier(
    input_dim=40,
    num_fourier_features=64,
    use_fourier=True,
)
```

## Customization Examples

### Use Only EEG Channels
```python
from deap_dataloader import get_default_channel_groups
use_channels = get_default_channel_groups()['eeg_only']  # 32 EEG channels
```

### Longer Sequences
```python
sequence_length = 4096  # ~32 seconds at 128 Hz
```

### Overlapping Windows
```python
stride = 1024  # 50% overlap when sequence_length=2048
```

### Larger Model
```python
model = VADPredictor(
    dim=128,
    depth=6,
    num_latents=256,
    latent_dim=512,
    latent_heads=8,
)
```

## Expected Results

**Baseline (Random):** MAE ~2.5 (middle of 1-9 scale)

**Quick Training (10 epochs, small model):** MAE ~2.0-2.5

**Good Training (50 epochs, medium model):** MAE ~1.5-2.0

**State-of-the-Art (optimized):** MAE ~0.8-1.2

## Next Steps for Improvement

1. **Hyperparameter tuning**: Learning rate, batch size, model size
2. **Data augmentation**: Time shifting, channel dropout, noise injection
3. **Larger sequences**: Use full 63-second trials (8064 samples)
4. **Ensemble methods**: Combine multiple models
5. **Frequency domain**: Add spectral features (FFT, wavelets)
6. **Cross-validation**: K-fold subject-wise cross-validation
7. **Multi-task learning**: Predict all 4 labels (VAD + Liking)
8. **Attention visualization**: Analyze what the model focuses on

## Technical Notes

### Why Perceiver?
- **Handles multimodal data**: Different channels processed uniformly
- **Reduces computational cost**: Latent bottleneck vs full self-attention
- **Flexible input**: Variable sequence lengths and channel counts
- **Proven architecture**: Strong performance on diverse tasks

### Why Subject-Independent Split?
- **Realistic evaluation**: Model must generalize to new people
- **Harder task**: Can't memorize subject-specific patterns
- **Clinical relevance**: Real applications use unseen subjects

### Why Reduced Parameters?
- **Fast iteration**: Quick experiments for development
- **Resource efficient**: Works on CPU/laptop
- **Proof of concept**: Verify pipeline before scaling up
- **Easy to scale**: Simply increase dimensions when ready

## Troubleshooting

**Out of Memory:**
- Reduce batch_size
- Reduce sequence_length
- Reduce model size (num_latents, latent_dim)

**Slow Training:**
- Set num_workers > 0 in dataloader
- Use GPU if available
- Reduce sequence_length
- Use smaller model

**Poor Performance:**
- Increase model size
- Train for more epochs
- Use longer sequences
- Try Fourier features
- Check data normalization

**NaN Loss:**
- Lower learning rate
- Add gradient clipping
- Check data for NaNs
- Reduce model size

## References

- **DEAP Dataset**: Koelstra et al., IEEE Transactions on Affective Computing, 2012
- **Perceiver**: Jaegle et al., ICML 2021
- **PerceiverIO**: Jaegle et al., NeurIPS 2021
- **Fourier Features**: Tancik et al., NeurIPS 2020

---

**Status:** ✅ Fully implemented and tested  
**Last Updated:** December 11, 2025
