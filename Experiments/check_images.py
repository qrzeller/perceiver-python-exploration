import torch
from torchvision import transforms, datasets
from PIL import Image
import os

# Check what the dataset is actually loading
train_dir = './dataset/train'

transform_test = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

if os.path.exists(train_dir):
    dataset = datasets.ImageFolder(train_dir, transform=transform_test)
    
    # Get first image
    img, label = dataset[0]
    print(f"Image tensor shape from dataloader: {img.shape}")
    print(f"Expected: (1, 48, 48) for grayscale or (3, 48, 48) for RGB")
    
    # Load original image
    img_path = dataset.imgs[0][0]
    print(f"\nLoading image from: {img_path}")
    pil_img = Image.open(img_path)
    print(f"PIL Image mode: {pil_img.mode}")
    print(f"PIL Image size: {pil_img.size}")
    
    # Check a few more images
    print("\nChecking first 5 images:")
    for i in range(min(5, len(dataset))):
        img_path = dataset.imgs[i][0]
        pil_img = Image.open(img_path)
        img_tensor, _ = dataset[i]
        print(f"  {i+1}. Mode: {pil_img.mode}, Tensor shape: {img_tensor.shape}")
