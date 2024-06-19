import torch
import torch.nn as nn
import torch.optim as optim
import os
from PIL import Image, ImageFile, ImageOps
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import argparse
from utils import *
from modelVAE import *
from dataset import *


def vae_loss(reconstructed_A, images_A, reconstructed_G, images_G, mu_A, logvar_A, mu_G, logvar_G):
    # Reconstruction loss
    recon_loss_A = F.mse_loss(reconstructed_A, images_A, reduction='sum')
    recon_loss_G = F.mse_loss(reconstructed_G, images_G, reduction='sum')

    # KL divergence
    kld_A = -0.5 * torch.sum(1 + logvar_A - mu_A.pow(2) - logvar_A.exp())
    kld_G = -0.5 * torch.sum(1 + logvar_G - mu_G.pow(2) - logvar_G.exp())

    return recon_loss_A + recon_loss_G + kld_A + kld_G


# Example of training loop
def train_vae(model, train_loader, optimizer, epochs=10, device='cuda'):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for images_A, images_G in tqdm(train_loader):
            images_A, images_G = images_A.to(device), images_G.to(device)

            optimizer.zero_grad()
            reconstructed_A, reconstructed_G, mu_A, logvar_A, mu_G, logvar_G, _, _ = model(images_A, images_G)
            loss = vae_loss(reconstructed_A, images_A, reconstructed_G, images_G, mu_A, logvar_A, mu_G, logvar_G)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader.dataset)}")


# Assuming the rest of the script is present with the necessary imports and functions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a VAE model.')
    parser.add_argument('--save_path', '-p', type=str, default='vae_model', help='Path to save the model and results')
    args = parser.parse_args()

    # Constants
    image_channels = 3
    image_size = 224
    aerial_scaling = 3
    hidden_dims = 512
    n_encoded = 1024
    n_phi = 10
    batch_size = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CrossViewVAE(n_phi, n_encoded, hidden_dims, image_size, output_channels=image_channels, pretrained=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    transform_cutouts = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop((image_size, image_size)),
        transforms.ToTensor()
    ])

    transform_panos = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=image_size, scale=(0.8, 1.0)),
        transforms.ToTensor()
    ])

    transform_aerial = transforms.Compose([
        transforms.Resize((int(image_size * aerial_scaling), int(image_size * aerial_scaling))),
        transforms.RandomRotation(degrees=360),
        transforms.CenterCrop((image_size, image_size)),
        transforms.ToTensor()
    ])

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    train_filenames_panos, val_filenames_panos = sample_paired_images('/home/lrusso/cvusa', sample_percentage=0.2, split_ratio=0.8, groundtype='panos')
    train_filenames_cutouts, val_filenames_cutouts = sample_paired_images('/home/lrusso/cvusa', sample_percentage=0.2, split_ratio=0.8, groundtype='cutouts')

    train_dataset_combined = CombinedPairedImagesDataset(train_filenames_panos, train_filenames_cutouts, transform_aerial=transform_aerial, transform_panos=transform_panos, transform_cutouts=transform_cutouts)
    train_dataloader_combined = DataLoader(train_dataset_combined, batch_size=batch_size, shuffle=True, num_workers=8)

    train_vae(model, train_dataloader_combined, optimizer, epochs=10, device=device)

    # Save the trained model
    os.makedirs(args.save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.save_path, 'vae_model.pth'))
    print('Training Complete! Model saved.')
