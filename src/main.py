import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from vision_transformer import VisionTransformer
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
from train import train, evaluate


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(15),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def main(args):
    train_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device_name = args.device.lower()

    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_name == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    img_size = 32
    num_patches = (img_size // args.patch_size) ** 2

    model = VisionTransformer(
        img_size=img_size,
        patch_size=args.patch_size,
        in_channels=3,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        num_classes=10,
        dropout=args.dropout
    ).to(device)
    
    if args.use_compile:
        if device.type == "mps":
            print("torch.compile is not supported on MPS — skipping.")
        else:
            model = torch.compile(model)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    print("Starting training...")
    train(model, train_loader, criterion, optimizer, scheduler, device, epochs=args.epochs)
    print("Starting evaluation...")
    evaluate(model, test_loader, device)

    print(f"Saving model to {args.save_path}")
    torch.save(model, args.save_path)
    print("Model saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Vision Transformer on CIFAR-10')

    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Input batch size for training')
    parser.add_argument('--lr', type=float, default=0.003, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')

    parser.add_argument('--patch-size', type=int, default=4, help='Size of image patches')
    parser.add_argument('--embed-dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--num-heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--mlp-dim', type=int, default=512, help='Dimension of the MLP layer in the transformer')
    parser.add_argument('--num-layers', type=int, default=6, help='Number of transformer encoder layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

    parser.add_argument('--data-dir', type=str, default='./data', help='Directory for storing dataset')
    parser.add_argument('--save-path', type=str, default='vit_cifar10_state.pth', help='Path to save the trained model')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of subprocesses for data loading')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'mps', 'cpu'], help='Device to run the model on: "cuda" for Nvidia GPU, "mps" for Apple Silicon GPU, or "cpu" for CPU')
    parser.add_argument('--use-compile', action='store_true', help="Use torch.compile for model acceleration")
    args = parser.parse_args()
    main(args)