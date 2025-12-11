# DEAP VAD Prediction with Perceiver

This project implements VAD (Valence, Arousal, Dominance) prediction from multimodal physiological signals using a Perceiver model on the DEAP dataset.

## Dataset Overview

The DEAP (Database for Emotion Analysis using Physiological signals) dataset contains:
- **32 subjects**, each watching 40 video stimuli
- **40 channels** of physiological signals:
  - 32 EEG channels
  - 2 EOG channels (horizontal and vertical)
  - 2 EMG channels (zygomaticus and trapezius)
  - GSR (galvanic skin response)
  - Respiration belt
  - Plethysmograph (blood volume pulse)
  - Temperature
- **8064 time samples** per trial (63 seconds at 128 Hz sampling rate)
- **Labels**: Valence, Arousal, Dominance, and Liking (1-9 scale)

### Data Structure
```
dataset/deap-dataset/data_preprocessed_python/
├── s01.dat  # Subject 1
├── s02.dat  # Subject 2
...
└── s32.dat  # Subject 32
```

Each `.dat` file contains:
- `data`: (40 trials, 40 channels, 8064 samples)
- `labels`: (40 trials, 4) - [Valence, Arousal, Dominance, Liking]

## Project Structure

```
Experiments-io/
├── deap_dataloader.py      # PyTorch DataLoader for DEAP dataset
├── train_deap_vad.py       # Main training script
├── test_deap_model.py      # Quick test script
└── deap.py                 # Original example (reference)
```

## Features

### DataLoader (`deap_dataloader.py`)
- **Flexible windowing**: Configurable sequence length and stride
- **Subject-independent splits**: Ensures no subject appears in multiple sets
- **Channel selection**: Use all channels or specific modalities
- **Normalization**: Per-channel z-score normalization
- **Efficient loading**: Lazy loading with configurable workers

Channel groups available:
```python
'all': list(range(40))              # All 40 channels
'eeg_only': list(range(32))         # Only EEG
'peripheral_only': list(range(32, 40))  # Only peripheral
'eeg_gsr_resp': list(range(32)) + [36, 37]  # EEG + GSR + Respiration
```

### Model (`train_deap_vad.py`)

**VADPredictor** - Perceiver-based architecture:
- Input projection layer (maps channels to model dimension)
- PerceiverIO core with cross-attention and latent self-attention
- Learnable query tokens for aggregation
- Output head for VAD regression

**Reduced parameters for fast testing:**
- `dim`: 64 (vs 128+ for full model)
- `depth`: 4 layers (vs 6-12)
- `num_latents`: 64 (vs 256+)
- `latent_dim`: 128 (vs 512+)
- ~1.6M parameters (optimized for quick experimentation)

### Training Features
- Subject-independent train/val/test split (70/15/15)
- MSE loss for regression
- AdamW optimizer with learning rate scheduling
- Per-dimension MAE metrics (Valence, Arousal, Dominance)
- Model checkpointing (saves best model)
- TensorBoard logging (optional, install with `pip install tensorboard`)

## Quick Start

### 1. Test the DataLoader
```bash
python Experiments-io/deap_dataloader.py
```

Expected output:
```
Loaded DEAP dataset:
  - Subjects: 22
  - Total samples: 2640
  - Channels: 40
  - Sequence length: 2048
  - Stride: 2048
```

### 2. Test the Model
```bash
python Experiments-io/test_deap_model.py
```

This runs a quick forward pass test with ~1.6M parameter model.

### 3. Train the Model
```bash
python Experiments-io/train_deap_vad.py
```

Training configuration (editable in `train_deap_vad.py`):
```python
config = {
    'sequence_length': 2048,  # ~16 seconds at 128 Hz
    'batch_size': 8,
    'num_epochs': 50,
    'learning_rate': 1e-3,
    'train_subjects': list(range(1, 23)),  # Subjects 1-22
    'val_subjects': list(range(23, 28)),   # Subjects 23-27
    'test_subjects': list(range(28, 33)),  # Subjects 28-32
}
```

### 4. Monitor Training

If TensorBoard is installed:
```bash
tensorboard --logdir outputs/
```

Otherwise, training metrics are printed to console.

## Customization

### Adjust Sequence Length
```python
# In train_deap_vad.py or when creating dataloaders
sequence_length = 1024   # ~8 seconds (faster training)
sequence_length = 4096   # ~32 seconds (more context)
sequence_length = 8064   # Full 63-second trial
```

### Use Specific Channels
```python
from deap_dataloader import get_default_channel_groups

# Only EEG
use_channels = get_default_channel_groups()['eeg_only']

# Only peripheral signals
use_channels = get_default_channel_groups()['peripheral_only']

# Custom selection
use_channels = [0, 1, 2, 36, 37]  # First 3 EEG + GSR + Respiration
```

### Scale Up the Model
```python
# In train_deap_vad.py, modify VADPredictor parameters:
model = VADPredictor(
    input_dim=40,
    dim=128,           # Increase from 64
    depth=6,           # Increase from 4
    num_latents=256,   # Increase from 64
    latent_dim=512,    # Increase from 128
    latent_heads=8,    # Increase from 4
)
```

### Overlapping Windows
```python
# Create overlapping windows for more training samples
stride = 1024  # 50% overlap when sequence_length=2048
```

## Model Architecture

```
Input: (batch, seq_len, n_channels)
  ↓
Input Projection: Linear(n_channels → dim)
  ↓
PerceiverIO:
  - Cross Attention: seq → latents
  - Latent Self-Attention (depth layers)
  - Query Cross-Attention: query → logits
  ↓
Output Head: Linear(64 → 32) → ReLU → Dropout → Linear(32 → 3)
  ↓
Output: (batch, 3)  # [Valence, Arousal, Dominance]
```

## Evaluation Metrics

- **MSE Loss**: Mean squared error across all VAD dimensions
- **MAE per dimension**: Mean absolute error for Valence, Arousal, Dominance
- **Overall MAE**: Average MAE across all three dimensions

## Output

Training saves to `./outputs/deap_vad_TIMESTAMP/`:
```
outputs/deap_vad_20231211_143022/
├── config.json           # Training configuration
├── best_model.pth        # Best model checkpoint
├── test_results.json     # Test set metrics
└── logs/                 # TensorBoard logs (if available)
```

## Tips for Fast Experimentation

1. **Start small**: Use 2-5 subjects initially
   ```python
   train_subjects = [1, 2, 3]
   val_subjects = [23]
   test_subjects = [28]
   ```

2. **Reduce sequence length**: Use 1024 or 2048 samples
   ```python
   sequence_length = 1024  # ~8 seconds
   ```

3. **Small model**: Keep default small parameters (~1.6M)

4. **Few epochs**: Start with 10-20 epochs to verify training

5. **Check for GPU**: Model automatically uses CUDA if available
   ```python
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   ```

## Expected Results

With the default small model and fast training settings:
- **Training time**: ~5-10 minutes per epoch (CPU), ~1-2 minutes (GPU)
- **MAE baseline**: ~2.0-3.0 (random predictions would give ~2.5)
- **MAE target**: <1.5 with proper training and larger model

State-of-the-art results on DEAP typically achieve:
- Valence MAE: ~0.8-1.2
- Arousal MAE: ~0.7-1.0
- Dominance MAE: ~0.9-1.3

## Requirements

Core dependencies (already in environment):
```
torch
numpy
pickle
tqdm
```

Optional:
```
tensorboard  # For training visualization
```

## Next Steps

1. **Hyperparameter tuning**: Learning rate, batch size, model size
2. **Data augmentation**: Time shifting, channel dropout
3. **Multi-task learning**: Predict all 4 labels (VAD + Liking)
4. **Cross-subject evaluation**: Evaluate on specific subject groups
5. **Ensemble models**: Combine multiple models or channel groups
6. **Frequency features**: Add Fourier features for better spectral representation

## References

- DEAP Dataset: [IEEE Xplore](https://ieeexplore.ieee.org/document/5871728)
- Perceiver: [arXiv](https://arxiv.org/abs/2103.03206)
- PerceiverIO: [arXiv](https://arxiv.org/abs/2107.14795)
