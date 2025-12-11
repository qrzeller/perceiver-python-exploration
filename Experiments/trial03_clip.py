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

# Added gradient clipping, ang changed parameters size and depth of the model

# --- Hyperparameters for the new scheduler ---
WARMUP_EPOCHS = 5      # Warmup phase duration in epochs
MIN_LR_RATIO = 0.01    # Minimum LR = learning_rate * MIN_LR_RATIO
GRAD_CLIP_NORM = 1.0   # Maximum L2 norm for gradient clipping

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
def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=1e-4, device='cuda', class_weights=None, total_steps=None):
    """
    Train the Perceiver model with Weighted Loss, Cosine Annealing, and Gradient Clipping.
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
    
    # --- SCHEDULER PARAMETERS ---
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
            cosine_factor = 0.5 * (1 + np.cos(np.pi * step_after_warmup / decay_steps))
            lr = min_lr + (max_lr - min_lr) * cosine_factor

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
            
            # --- GRADIENT CLIPPING ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            # -------------------------
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            train_bar.set_postfix({
                'loss': f'{train_loss/(train_bar.n+1):.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%',
                'lr': f'{current_lr:.6f}'
            })
        
        # Validation phase
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
        
        # Calculate metrics
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
    image = Image.open(image_path).convert('L')
    image = transform_test(image).unsqueeze(0)
    image = image.permute(0, 2, 3, 1).to(device)
    
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return emotion_labels[predicted_class], confidence, probabilities[0].cpu().numpy()

# Main execution
if __name__ == "__main__":
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f'Using device: mps (Apple Silicon GPU)')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using device: cuda')
    else:
        exit("No GPU device found. Please run on a machine with CUDA or MPS support.")
    
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
        
        # --- CLASS WEIGHT CALCULATION ---
        train_targets = np.array(train_dataset.targets)
        class_counts = Counter(train_targets)
        num_classes = len(class_names)
        total_samples = len(train_dataset)

        counts = [class_counts[i] for i in range(num_classes)]
        class_weights = total_samples / (num_classes * torch.tensor(counts, dtype=torch.float32))
        
        print(f'\nClass Sample Counts: {counts}')
        print(f'Calculated Class Weights: {class_weights.tolist()}')
        class_weights = class_weights.to(device)
        # --------------------------------
        
        # Initialize Perceiver model for FER2013
        print("\nInitializing Perceiver model with INCREASED CAPACITY...")
        model = Perceiver(
            input_channels = 1,          
            input_axis = 2,              
            num_freq_bands = 6,          
            max_freq = 8.,               # <-- INCREASED FREQ
            depth = 12,                  # <-- INCREASED DEPTH
            num_latents = 256,            # <-- INCREASED LATENTS
            latent_dim = 32,             # <-- INCREASED LATENT DIM
            cross_heads = 2,             
            latent_heads = 8,            
            cross_dim_head = 64,         
            latent_dim_head = 64,        
            num_classes = 7,             
            attn_dropout = 0.1,
            ff_dropout = 0.1,
            weight_tie_layers = False,
            fourier_encode_data = True,
            self_per_cross_attn = 1      
        )
        
 

        batch_size = 4
        num_workers = 2
        num_epochs = 50 
        
        # --- CALCULATE TOTAL STEPS FOR SCHEDULER ---
        steps_per_epoch = int(np.ceil(len(train_dataset) / batch_size))
        TOTAL_STEPS = num_epochs * steps_per_epoch
        print(f"Total training steps: {TOTAL_STEPS}")
        # -------------------------------------------
        
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
            total_steps=TOTAL_STEPS
        )
        
        print("\nTraining completed!")
        
    else:
        print(f"FER2013 folders not found!")
        print(f"Expected train folder at: {train_dir}")
        print(f"Expected test folder at: {test_dir}")
        
        # Example: Create a simple test with random data
        print("\nRunning simple forward pass test with random data...")
        model = Perceiver(
            input_channels = 1, input_axis = 2, num_freq_bands = 6, max_freq = 8., depth = 12, 
            num_latents = 64, latent_dim = 64, cross_heads = 1, latent_heads = 8, 
            cross_dim_head = 64, latent_dim_head = 64, num_classes = 7, attn_dropout = 0.1,
            ff_dropout = 0.1, weight_tie_layers = False, fourier_encode_data = True, self_per_cross_attn = 1
        )
        test_img = torch.randn(1, 48, 48, 1).to(device)
        model = model.to(device)
        output = model(test_img)
        print(f"Output shape: {output.shape}")
        print(f"Predicted emotion class: {output.argmax(dim=1).item()}")