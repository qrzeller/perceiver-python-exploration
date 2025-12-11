import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image
from perceiver_pytorch import Perceiver
from tqdm import tqdm
import os
from collections import Counter
import numpy as np

# --- Hyperparameters for the new scheduler ---
WARMUP_EPOCHS = 5 # Warmup phase duration in epochs
MIN_LR_RATIO = 0.01 # Minimum LR = learning_rate * MIN_LR_RATIO

# Define transforms for FER2013 (Unchanged)
transform_train = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

transform_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Training function
# --- MODIFIED: REMOVED PLATFORM SCHEDULER, ADDED TOTAL_STEPS ---
def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=1e-4, device='cuda', class_weights=None, total_steps=None):
    """
    Train the Perceiver model on FER2013 with Weighted Loss and Cosine Annealing LR Schedule.
    """
    model = model.to(device)
    
    # Loss Criterion Setup (Weighted Cross-Entropy)
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device)) 
        print(f"Using Weighted Cross-Entropy Loss with weights: {class_weights.tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using standard Cross-Entropy Loss.")
        
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # --- NEW SCHEDULER PARAMETERS ---
    steps_per_epoch = len(train_loader)
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    max_lr = learning_rate
    min_lr = learning_rate * MIN_LR_RATIO
    current_step = 0
    
    best_val_acc = 0.0
    
    # --- FUNCTION TO UPDATE LEARNING RATE ---
    def update_lr(optimizer, step, total_steps, warmup_steps, max_lr, min_lr):
        if step < warmup_steps:
            # Linear Warmup Phase
            lr = min_lr + (max_lr - min_lr) * (step / warmup_steps)
        else:
            # Cosine Annealing Phase
            decay_steps = total_steps - warmup_steps
            step_after_warmup = step - warmup_steps
            # Calculate the decay factor using cosine function
            cosine_factor = 0.5 * (1 + np.cos(np.pi * step_after_warmup / decay_steps))
            lr = min_lr + (max_lr - min_lr) * cosine_factor

        # Apply the calculated LR to all parameter groups
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

    print(f"Using Warmup ({WARMUP_EPOCHS} epochs) + Cosine Annealing schedule.")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in train_bar:
            # --- LR UPDATE ON EACH STEP ---
            current_lr = update_lr(optimizer, current_step, total_steps, warmup_steps, max_lr, min_lr)
            current_step += 1
            # -------------------------------

            images = images.permute(0, 2, 3, 1).to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # NOTE: Consider adding Gradient Clipping here for stability, as suggested previously.
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            train_bar.set_postfix({
                'loss': f'{train_loss/(train_bar.n+1):.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%',
                'lr': f'{current_lr:.6f}' # Display current LR
            })
        
        # Validation phase (Unchanged)
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, labels in val_bar:
                images = images.permute(0, 2, 3, 1).to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                val_bar.set_postfix({
                    'loss': f'{val_loss/(val_bar.n+1):.4f}',
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        # Calculate metrics (Note: No scheduler.step() needed for val loss)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f'\nEpoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}% (LR: {current_lr:.6f})')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_perceiver_fer2013.pth')
            print(f'Saved best model with validation accuracy: {best_val_acc:.2f}%')
    
    return model

# Prediction function (Unchanged)
def predict_emotion(model, image_path, device='cuda'):
    """
    Predict emotion from a single image
    """
    model.eval()
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    # Load and preprocess image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = transform_test(image).unsqueeze(0)  # Add batch dimension
    image = image.permute(0, 2, 3, 1).to(device)  # Reshape for Perceiver
    
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return emotion_labels[predicted_class], confidence, probabilities[0].cpu().numpy()

# Main execution
if __name__ == "__main__":
    # Set device - prioritize MPS (Apple Silicon GPU), then CUDA, then CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f'Using device: mps (Apple Silicon GPU)')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using device: cuda')
    else:
        device = torch.device('cpu')
        print(f'Using device: cpu')
    
    # Paths to FER2013 train and test folders
    train_dir = './dataset/train'
    test_dir = './dataset/test'
    
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        print("Loading FER2013 dataset from folders...")
        
        # Create datasets using ImageFolder
        train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
        test_dataset = datasets.ImageFolder(test_dir, transform=transform_test)
        
        # Get class names
        class_names = train_dataset.classes
        print(f'Emotion classes: {class_names}')
        
        # --- CLASS WEIGHT CALCULATION (Same as previous step) ---
        train_targets = np.array(train_dataset.targets)
        class_counts = Counter(train_targets)
        num_classes = len(class_names)
        total_samples = len(train_dataset)

        counts = [class_counts[i] for i in range(num_classes)]
        class_weights = total_samples / (num_classes * torch.tensor(counts, dtype=torch.float32))
        
        print(f'\nClass Sample Counts: {counts}')
        print(f'Calculated Class Weights: {class_weights.tolist()}')
        class_weights = class_weights.to(device)
        # --------------------------------------------------------
        
        # Initialize Perceiver model for FER2013
        print("\nInitializing Perceiver model...")
        model = Perceiver(
            input_channels = 1, input_axis = 2, num_freq_bands = 6, max_freq = 4., 
            depth = 8, num_latents = 32, latent_dim = 32, cross_heads = 1, latent_heads = 8, 
            cross_dim_head = 64, latent_dim_head = 64, num_classes = 7, attn_dropout = 0.1,
            ff_dropout = 0.1, weight_tie_layers = False, fourier_encode_data = True, self_per_cross_attn = 1
        )
        
        # Adjust batch size and workers based on device
        if device.type == 'cpu':
            batch_size = 16
            num_workers = 0
        elif device.type == 'mps':
            batch_size = 64
            num_workers = 2
        else:  # cuda
            batch_size = 64
            num_workers = 4

        num_epochs = 50 # Defined here for convenience
        
        # --- NEW: CALCULATE TOTAL STEPS FOR SCHEDULER ---
        steps_per_epoch = int(np.ceil(len(train_dataset) / batch_size))
        TOTAL_STEPS = num_epochs * steps_per_epoch
        print(f"Total training steps: {TOTAL_STEPS}")
        # -----------------------------------------------
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        # Train the model 
        print(f"\nStarting training with batch_size={batch_size}...")
        model = train_model(
            model, 
            train_loader, 
            test_loader, 
            num_epochs=num_epochs, 
            learning_rate=1e-4, 
            device=device,
            class_weights=class_weights, 
            total_steps=TOTAL_STEPS # <-- PASS TOTAL STEPS
        )
        
        print("\nTraining completed!")
        
    else:
        # ... (error handling code remains the same)
        print(f"FER2013 folders not found!")
        print(f"Expected train folder at: {train_dir}")
        print(f"Expected test folder at: {test_dir}")
        
        # ... (simple forward pass test)