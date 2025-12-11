import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image
from perceiver_pytorch import Perceiver
from tqdm import tqdm
import os

# Define transforms for FER2013
transform_train = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure single channel
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

transform_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure single channel
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Training function
def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=1e-4, device='cuda'):
    """
    Train the Perceiver model on FER2013
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in train_bar:
            # Reshape images from (batch, 1, 48, 48) to (batch, 48, 48, 1) for Perceiver
            images = images.permute(0, 2, 3, 1).to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            train_bar.set_postfix({
                'loss': f'{train_loss/(train_bar.n+1):.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
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
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'\nEpoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_perceiver_fer2013.pth')
            print(f'Saved best model with validation accuracy: {best_val_acc:.2f}%')
    
    return model

# Prediction function
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
        print(f'Training samples: {len(train_dataset)}')
        print(f'Test samples: {len(test_dataset)}')
        
        # Initialize Perceiver model for FER2013
        print("\nInitializing Perceiver model...")
        model = Perceiver(
            input_channels = 1,          # grayscale images (1 channel)
            input_axis = 2,              # 2D images
            num_freq_bands = 6,          # frequency bands for positional encoding
            max_freq = 4.,               # reduced for 48x48 images
            depth = 8,                   # reduced depth for smaller dataset
            num_latents = 32,            # reduced for smaller images
            latent_dim = 32,            # latent dimension
            cross_heads = 1,             # cross attention heads
            latent_heads = 8,            # self attention heads
            cross_dim_head = 64,         # dimensions per cross attention head
            latent_dim_head = 64,        # dimensions per latent self attention head
            num_classes = 7,             # 7 emotion classes in FER2013
            attn_dropout = 0.1,
            ff_dropout = 0.1,
            weight_tie_layers = False,
            fourier_encode_data = True,  # use fourier encoding for positional information
            self_per_cross_attn = 1      # reduced for efficiency
        )
        
        # Adjust batch size and workers based on device
        if device.type == 'cpu':
            batch_size = 16
            num_workers = 0
        elif device.type == 'mps':
            batch_size = 64  # MPS can handle larger batches
            num_workers = 2  # Fewer workers for MPS
        else:  # cuda
            batch_size = 64
            num_workers = 4
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        # Train the model (using test set as validation for simplicity)
        print(f"\nStarting training with batch_size={batch_size}...")
        model = train_model(model, train_loader, test_loader, num_epochs=50, learning_rate=1e-4, device=device)
        
        print("\nTraining completed!")
        
    else:
        print(f"FER2013 folders not found!")
        print(f"Expected train folder at: {train_dir}")
        print(f"Expected test folder at: {test_dir}")
        print("\nMake sure your FER2013 dataset is organized as:")
        print("dataset/")
        print("  train/")
        print("    angry/")
        print("    disgust/")
        print("    fear/")
        print("    happy/")
        print("    neutral/")
        print("    sad/")
        print("    surprise/")
        print("  test/")
        print("    angry/")
        print("    disgust/")
        print("    ...")
        print("\nFor prediction only (with pre-trained model):")
        print("emotion, confidence, probs = predict_emotion(model, 'path/to/image.jpg', device=device)")
        
        # Example: Create a simple test with random data
        print("\nRunning simple forward pass test with random data...")
        test_img = torch.randn(1, 48, 48, 1).to(device)  # (batch, height, width, channels)
        model = model.to(device)
        output = model(test_img)
        print(f"Output shape: {output.shape}")  # Should be (1, 7)
        print(f"Predicted emotion class: {output.argmax(dim=1).item()}")