"""
Data Exploration and Visualization for DEAP Dataset

This script provides utilities to explore and visualize the DEAP dataset,
helping to understand the data distribution and characteristics.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_subject(data_dir, subject_id):
    """Load a single subject's data."""
    filepath = Path(data_dir) / f's{subject_id:02d}.dat'
    with open(filepath, 'rb') as f:
        subject_data = pickle.load(f, encoding='latin1')
    return subject_data


def plot_channel_signals(subject_data, trial_idx=0, channels=[0, 32, 36, 37], duration=5.0):
    """
    Plot sample signals from different channels.
    
    Args:
        subject_data: Loaded subject data dictionary
        trial_idx: Which trial to visualize
        channels: List of channel indices to plot
        duration: Duration to plot in seconds
    """
    data = subject_data['data'][trial_idx]  # (40 channels, 8064 samples)
    labels = subject_data['labels'][trial_idx]  # (4,) VAD + Liking
    
    sampling_rate = 128  # Hz
    num_samples = int(duration * sampling_rate)
    time = np.arange(num_samples) / sampling_rate
    
    channel_names = {
        0: 'EEG (Fp1)',
        32: 'hEOG',
        33: 'vEOG',
        34: 'zEMG',
        35: 'tEMG',
        36: 'GSR',
        37: 'Respiration',
        38: 'Plethysmograph',
        39: 'Temperature',
    }
    
    fig, axes = plt.subplots(len(channels), 1, figsize=(12, 2.5 * len(channels)))
    if len(channels) == 1:
        axes = [axes]
    
    fig.suptitle(f'Trial {trial_idx} - Valence: {labels[0]:.2f}, Arousal: {labels[1]:.2f}, Dominance: {labels[2]:.2f}',
                 fontsize=14, fontweight='bold')
    
    for idx, (ax, ch) in enumerate(zip(axes, channels)):
        signal = data[ch, :num_samples]
        ax.plot(time, signal, linewidth=0.8)
        ax.set_ylabel(channel_names.get(ch, f'Channel {ch}'), fontsize=11)
        ax.grid(True, alpha=0.3)
        
        if idx == len(channels) - 1:
            ax.set_xlabel('Time (seconds)', fontsize=11)
    
    plt.tight_layout()
    return fig


def plot_label_distribution(data_dir, subject_ids=None):
    """
    Plot distribution of VAD labels across subjects.
    
    Args:
        data_dir: Path to data directory
        subject_ids: List of subject IDs to include (default: all)
    """
    if subject_ids is None:
        subject_ids = list(range(1, 33))
    
    all_labels = []
    
    for subject_id in subject_ids:
        try:
            subject_data = load_subject(data_dir, subject_id)
            all_labels.append(subject_data['labels'][:, :3])  # VAD only
        except FileNotFoundError:
            continue
    
    all_labels = np.concatenate(all_labels, axis=0)  # (n_samples, 3)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    labels_names = ['Valence', 'Arousal', 'Dominance']
    
    for idx, (ax, name) in enumerate(zip(axes, labels_names)):
        ax.hist(all_labels[:, idx], bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel(f'{name} Score (1-9)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{name} Distribution', fontsize=12, fontweight='bold')
        ax.axvline(all_labels[:, idx].mean(), color='red', linestyle='--', 
                   label=f'Mean: {all_labels[:, idx].mean():.2f}')
        ax.axvline(np.median(all_labels[:, idx]), color='green', linestyle='--',
                   label=f'Median: {np.median(all_labels[:, idx]):.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_channel_statistics(data_dir, subject_ids=None, num_samples=100):
    """
    Plot statistics (mean, std) for each channel across multiple trials.
    
    Args:
        data_dir: Path to data directory
        subject_ids: List of subject IDs to sample from
        num_samples: Number of trials to analyze
    """
    if subject_ids is None:
        subject_ids = [1, 2, 3]  # Sample a few subjects
    
    all_means = []
    all_stds = []
    
    count = 0
    for subject_id in subject_ids:
        try:
            subject_data = load_subject(data_dir, subject_id)
            data = subject_data['data']  # (40 trials, 40 channels, 8064)
            
            for trial_idx in range(data.shape[0]):
                trial_data = data[trial_idx]  # (40 channels, 8064)
                all_means.append(trial_data.mean(axis=1))
                all_stds.append(trial_data.std(axis=1))
                
                count += 1
                if count >= num_samples:
                    break
            
            if count >= num_samples:
                break
        except FileNotFoundError:
            continue
    
    all_means = np.array(all_means)  # (num_samples, 40 channels)
    all_stds = np.array(all_stds)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Mean values
    axes[0].boxplot([all_means[:, i] for i in range(40)], 
                     labels=[str(i) for i in range(40)])
    axes[0].set_xlabel('Channel Index', fontsize=11)
    axes[0].set_ylabel('Mean Value', fontsize=11)
    axes[0].set_title('Channel Mean Distribution Across Trials', fontsize=12, fontweight='bold')
    axes[0].axvline(31.5, color='red', linestyle='--', alpha=0.5, label='EEG | Peripheral')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Std values
    axes[1].boxplot([all_stds[:, i] for i in range(40)],
                     labels=[str(i) for i in range(40)])
    axes[1].set_xlabel('Channel Index', fontsize=11)
    axes[1].set_ylabel('Standard Deviation', fontsize=11)
    axes[1].set_title('Channel Std Distribution Across Trials', fontsize=12, fontweight='bold')
    axes[1].axvline(31.5, color='red', linestyle='--', alpha=0.5, label='EEG | Peripheral')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_correlation_matrix(data_dir, subject_id=1, trial_idx=0):
    """
    Plot correlation matrix between channels for a single trial.
    
    Args:
        data_dir: Path to data directory
        subject_id: Which subject to analyze
        trial_idx: Which trial to analyze
    """
    subject_data = load_subject(data_dir, subject_id)
    data = subject_data['data'][trial_idx]  # (40 channels, 8064 samples)
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(data)  # (40, 40)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    
    ax.set_title(f'Channel Correlation Matrix - Subject {subject_id}, Trial {trial_idx}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Channel Index', fontsize=11)
    ax.set_ylabel('Channel Index', fontsize=11)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', fontsize=11)
    
    # Add grid lines to separate EEG and peripheral
    ax.axhline(31.5, color='black', linewidth=2, linestyle='--', alpha=0.5)
    ax.axvline(31.5, color='black', linewidth=2, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return fig


def main():
    """Main exploration script."""
    data_dir = './dataset/deap-dataset/data_preprocessed_python'
    
    print("DEAP Dataset Exploration")
    print("=" * 50)
    
    # Load a sample subject
    print("\nLoading subject 1...")
    subject_data = load_subject(data_dir, 1)
    
    print(f"Data shape: {subject_data['data'].shape}")
    print(f"Labels shape: {subject_data['labels'].shape}")
    print(f"Number of trials: {subject_data['data'].shape[0]}")
    print(f"Number of channels: {subject_data['data'].shape[1]}")
    print(f"Number of samples per trial: {subject_data['data'].shape[2]}")
    print(f"Duration per trial: {subject_data['data'].shape[2] / 128:.1f} seconds")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    print("1. Plotting sample signals...")
    fig1 = plot_channel_signals(subject_data, trial_idx=0, 
                                 channels=[0, 36, 37, 39],  # EEG, GSR, Resp, Temp
                                 duration=10.0)
    fig1.savefig('deap_sample_signals.png', dpi=150, bbox_inches='tight')
    print("   Saved: deap_sample_signals.png")
    
    print("2. Plotting label distribution...")
    fig2 = plot_label_distribution(data_dir, subject_ids=list(range(1, 11)))
    fig2.savefig('deap_label_distribution.png', dpi=150, bbox_inches='tight')
    print("   Saved: deap_label_distribution.png")
    
    print("3. Plotting channel statistics...")
    fig3 = plot_channel_statistics(data_dir, subject_ids=[1, 2, 3], num_samples=50)
    fig3.savefig('deap_channel_statistics.png', dpi=150, bbox_inches='tight')
    print("   Saved: deap_channel_statistics.png")
    
    print("4. Plotting correlation matrix...")
    fig4 = plot_correlation_matrix(data_dir, subject_id=1, trial_idx=0)
    fig4.savefig('deap_correlation_matrix.png', dpi=150, bbox_inches='tight')
    print("   Saved: deap_correlation_matrix.png")
    
    print("\n✓ Exploration complete!")
    print("\nVisualization files saved to current directory.")


if __name__ == '__main__':
    # Check if matplotlib is available
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        main()
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        print("\nAlternatively, you can still use the dataloader without visualization.")
