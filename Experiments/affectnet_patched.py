"""
A patched copy of `affectnet.py` that uses the channel-fusing patch embedder
`Experiments.patchers.channel_fuse.ChannelFusePatchEmbed`.

- Uses NHWC token output from the patcher and feeds tokens directly to Perceiver.
- Default `out_dim=1` (fuses RGB into a single value per patch). Adjust if you
  want richer per-patch features.

Run:
    python Experiments/affectnet_patched.py

This file intentionally mirrors the original experiment structure so you can
swap it in without editing the Perceiver library.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image
from perceiver_pytorch import Perceiver
from patchers.channel_fuse import ChannelFusePatchEmbed
from tqdm import tqdm


transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --- Training function ---

def train_model(model, patcher, train_loader, val_loader, num_epochs=10, learning_rate=3e-4, device='cuda'):
    model = model.to(device)
    patcher = patcher.to(device)

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # Slight label smoothing helps generalization on noisy facial datasets
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    # Cosine scheduling generally works well with AdamW
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # detect bf16 support
    use_bf16 = False
    if device.type == 'cuda' and hasattr(torch.cuda, 'is_bf16_supported'):
        use_bf16 = torch.cuda.is_bf16_supported()

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in train_bar:
            # images: (B, C, H, W) -> NHWC
            images = images.permute(0, 2, 3, 1).to(device, non_blocking=True)
            labels = labels.to(device)

            optimizer.zero_grad()

            # patchify + fuse channels -> tokens (B, H', W', out_dim)
            tokens = patcher(images)

            if use_bf16:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = model(tokens)
                    loss = criterion(outputs.float(), labels)
            else:
                outputs = model(tokens)
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

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, labels in val_bar:
                images = images.permute(0, 2, 3, 1).to(device, non_blocking=True)
                labels = labels.to(device)

                tokens = patcher(images)

                if use_bf16:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        outputs = model(tokens)
                        loss = criterion(outputs.float(), labels)
                else:
                    outputs = model(tokens)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                val_bar.set_postfix({
                    'loss': f'{val_loss/(val_bar.n+1):.4f}',
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })

        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        print(f'\nEpoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_perceiver_affectnet_patched.pth')
            print(f'Saved best model with validation accuracy: {best_val_acc:.2f}%')

    return model


# --- Prediction helper ---

def predict_emotion(model, patcher, image_path, device='cuda'):
    model.eval()
    patcher = patcher.to(device)

    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    image = Image.open(image_path).convert('RGB')
    image = transform_test(image).unsqueeze(0)  # (1, C, H, W)
    image = image.permute(0, 2, 3, 1).to(device)

    with torch.no_grad():
        tokens = patcher(image)
        output = model(tokens)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    return emotion_labels[predicted_class], confidence, probabilities[0].cpu().numpy()


# --- Main ---
if __name__ == '__main__':
    # device selection
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print('Using device: mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using device: cuda')
    else:
        device = torch.device('cpu')
        print('Using device: cpu')

    # dataset paths
    train_dir = './dataset/affectnet/Train'
    test_dir = './dataset/affectnet/Test'

    if os.path.exists(train_dir) and os.path.exists(test_dir):
        train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
        test_dataset = datasets.ImageFolder(test_dir, transform=transform_test)

        class_names = train_dataset.classes
        print(f'Emotion classes: {class_names}')
        print(f'Training samples: {len(train_dataset)}')
        print(f'Test samples: {len(test_dataset)}')

        # Patch embedder
        # Richer patch tokens usually help: smaller patches, overlap, higher out_dim
        patch_size = 8
        out_dim = 32  # fused channel features -> Perceiver input_channels must match
        overlap = True
        patcher = ChannelFusePatchEmbed(in_ch=3, out_dim=out_dim, patch_size=patch_size, overlap=overlap, use_bn=True)

        # Perceiver init: input_channels must match patcher.out_dim
        # Increase latent capacity and cross heads to better capture local tokens
        model = Perceiver(
            input_channels = out_dim,
            input_axis = 2,
            num_freq_bands = 6,
            max_freq = 6.,
            depth = 8,
            num_latents = 128,
            latent_dim = 512,
            cross_heads = 4,
            latent_heads = 8,
            cross_dim_head = 64,
            latent_dim_head = 64,
            num_classes = len(class_names),
            attn_dropout = 0.1,
            ff_dropout = 0.1,
            weight_tie_layers = False,
            fourier_encode_data = True,
            self_per_cross_attn = 2
        )

        # dataloader params
        if device.type == 'cpu':
            batch_size = 16
            num_workers = 0
            pin_memory = False
        elif device.type == 'mps':
            batch_size = 8
            num_workers = 2
            pin_memory = False
        else:
            batch_size = 16
            num_workers = 4
            pin_memory = True

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers>0), prefetch_factor=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers>0), prefetch_factor=2)

        print(f'Starting training with batch_size={batch_size}, patch_size={patch_size}, out_dim={out_dim}...')
        model = train_model(model, patcher, train_loader, test_loader, num_epochs=50, learning_rate=1e-4, device=device)

        print('Training completed')
    else:
        print('Train/Test folders not found. Run the script after placing dataset in expected paths.')
