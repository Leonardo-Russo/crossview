import torch
import torch.nn as nn
import torch.optim as optim
import os
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import argparse
from utils import *
from model import *
from dataset import *

def train(model, train_loader, val_loader, device, criterion_A, criterion_G, optimizer, epochs=1, save_path='untitled', debug=False):
    model.to(device)

    model_path = os.path.join('models', save_path)
    metrics_path = os.path.join('models', save_path, 'metrics')
    results_path = os.path.join('models', save_path, 'results')
    os.makedirs('models', exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(metrics_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(os.path.join(results_path, 'aerial'), exist_ok=True)
    os.makedirs(os.path.join(results_path, 'ground'), exist_ok=True)

    save_dataset_samples(train_loader, os.path.join(model_path, 'training_samples.png'), num_images=16, title='Training Samples')
    save_dataset_samples(val_loader, os.path.join(model_path, 'validation_samples.png'), num_images=16, title='Validation Samples')

    train_huber_losses = []
    val_huber_losses = []
    train_ssim_losses = []
    val_ssim_losses = []
    best_val_loss = np.inf
    patience_counter = 0
    best_model_path = None

    embedded_loss = True

    for epoch in range(epochs):
        
        model.train()
        running_huber_loss = 0.0
        running_ssim_loss = 0.0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}', unit='batch') as pbar:
            for images_A, images_G in train_loader:
                images_A, images_G = images_A.to(device), images_G.to(device)

                # Forward Pass
                reconstructed_A, reconstructed_G, encoded_A, encoded_G, phi, encoded_A_phi, encoded_G_phi = model(images_A, images_G)

                # Compute Pixel-wise Loss
                if embedded_loss:
                    loss_A = nn.functional.l1_loss(encoded_A_phi, encoded_G)
                    loss_G = nn.functional.l1_loss(encoded_G_phi, encoded_A)
                else:
                    loss_A = criterion_A(reconstructed_A, images_A)
                    loss_G = criterion_G(reconstructed_G, images_G)
                huber_loss = loss_A + loss_G
                running_huber_loss += huber_loss.item()

                # Compute SSIM Loss
                ssim_loss_A = ssim_loss(reconstructed_A, images_A)
                ssim_loss_G = ssim_loss(reconstructed_G, images_G)
                ssim_loss_value = 1 - (ssim_loss_A + ssim_loss_G) / 2
                running_ssim_loss += ssim_loss_value.item()

                # Reset Gradients
                optimizer.zero_grad()
                
                # Backward Propagation and Optimization Step
                huber_loss.backward()
                optimizer.step()

                # Update the progress bar with the current loss
                pbar.set_postfix({'Huber Loss': running_huber_loss / (pbar.n + 1), 'SSIM Loss': running_ssim_loss / (pbar.n + 1)})
                pbar.update()

        train_huber_loss = running_huber_loss / len(train_loader)
        train_ssim_loss = running_ssim_loss / len(train_loader)
        train_huber_losses.append(train_huber_loss)
        train_ssim_losses.append(train_ssim_loss)

        # Validation
        val_huber_loss, val_ssim_loss = validate(model, val_loader, criterion_A, criterion_G, epoch, epochs, results_path, device)
        val_huber_losses.append(val_huber_loss)
        val_ssim_losses.append(val_ssim_loss)

        # Check for Best Model
        if val_huber_loss < best_val_loss:
            best_val_loss = val_huber_loss
            patience_counter = 0

            if best_model_path is not None and os.path.exists(best_model_path):
                os.remove(best_model_path)

            best_model_path = os.path.join(model_path, f'best_model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        # Update the plot with the current losses
        update_plot(epoch + 1, train_huber_losses, val_huber_losses, train_ssim_losses, val_ssim_losses, metrics_path)

    # Save the Model
    torch.save(model.state_dict(), os.path.join(model_path, f'last_model_epoch_{epoch+1}.pth'))
    print('Training Complete!\nPatience Counter:', patience_counter)

    # Perform additional validation step with the best model
    if best_model_path:
        print(f'Loading the model from {best_model_path}...')
        model.load_state_dict(torch.load(best_model_path))
        final_val_loss, final_val_ssim_loss = validate(model, val_loader, criterion_A, criterion_G, "best", epochs, results_path, device)
        print(f'Best Validation Loss: {final_val_loss:.4f}')


def validate(model, val_loader, criterion_A, criterion_G, epoch, epochs, results_path, device):
    model.eval()
    val_huber_loss = 0
    val_ssim_loss = 0
    first_batch = True

    embedded_loss = True

    with torch.no_grad():
        for images_A, images_G in val_loader:
            images_A, images_G = images_A.to(device), images_G.to(device)

            # Forward Pass
            reconstructed_A, reconstructed_G, encoded_A, encoded_G, phi, encoded_A_phi, encoded_G_phi = model(images_A, images_G)

            # Compute Pixel-wise Loss
            if embedded_loss:
                loss_A = nn.functional.l1_loss(encoded_A_phi, encoded_G)
                loss_G = nn.functional.l1_loss(encoded_G_phi, encoded_A)
            else:
                loss_A = criterion_A(reconstructed_A, images_A)
                loss_G = criterion_G(reconstructed_G, images_G)
            huber_loss = loss_A + loss_G
            val_huber_loss += huber_loss.item()

            # Compute SSIM Loss
            ssim_loss_A = ssim_loss(reconstructed_A, images_A)
            ssim_loss_G = ssim_loss(reconstructed_G, images_G)
            ssim_loss_value = 1 - (ssim_loss_A + ssim_loss_G) / 2
            val_ssim_loss += ssim_loss_value.item()

            if first_batch:
                first_batch = False
                if epoch != "best":
                    visualize_reconstruction(images_A, reconstructed_A, epoch, save_path=os.path.join(results_path, 'aerial', f'epoch_{epoch + 1}_reconstruction.png'))
                    visualize_reconstruction(images_G, reconstructed_G, epoch, save_path=os.path.join(results_path, 'ground', f'epoch_{epoch + 1}_reconstruction.png'))
                else:
                    visualize_reconstruction(images_A, reconstructed_A, epoch, save_path=os.path.join(results_path, 'aerial', 'best_reconstruction.png'))
                    visualize_reconstruction(images_G, reconstructed_G, epoch, save_path=os.path.join(results_path, 'ground', 'best_reconstruction.png'))

    val_avg_huber_loss = val_huber_loss / len(val_loader)
    val_avg_ssim_loss = val_ssim_loss / len(val_loader)
    return val_avg_huber_loss, val_avg_ssim_loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--save_path', '-p', type=str, default='untitled', help='Path to save the model and results')
    args = parser.parse_args()

    # Constants
    image_channels = 3                      # RGB images dimensions
    attention_channels = 1                  # attention map dimensions
    image_size = 224                        # assuming square images
    aerial_scaling = 3                      # scaling factor for aerial images
    hidden_dims = 512                       # hidden dimensions
    n_encoded = 1024                        # output size for the encoders
    n_phi = 10                              # size of phi
    batch_size = 64
    shuffle = True

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    # Initialize the Architecture
    model = CrossView(n_phi, n_encoded, hidden_dims, image_size, output_channels=image_channels, pretrained=True).to(device)
    print(model)

    # Optimizer
    learning_rate = 1e-5
    params = [{"params": model.parameters()}]
    weight_decay = 1e-5
    optimizer = optim.Adam(params=params, lr=learning_rate, weight_decay=weight_decay)

    # Loss Function
    encoder_A = Encoder(latent_dim=n_encoded).to(device)
    encoder_G = Encoder(latent_dim=n_encoded).to(device)
    for param in encoder_A.parameters():
        param.requires_grad = False
    for param in encoder_G.parameters():
        param.requires_grad = False
    criterion_A = PerceptualLoss(encoder_A)
    criterion_G = PerceptualLoss(encoder_G)
    # criterion_A = nn.HuberLoss()
    # criterion_G = nn.HuberLoss()

    # Transformations
    transform_cutouts = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop((image_size, image_size)),
        transforms.ToTensor()
    ])

    transform_panos = transforms.Compose([
        RandomHorizontalShiftWithWrap(shift_range=1.0),
        transforms.RandomResizedCrop(size=image_size, scale=(0.8, 1.0)),
        transforms.ToTensor()
    ])

    transform_aerial = transforms.Compose([
        transforms.Resize((int(image_size * aerial_scaling), int(image_size * aerial_scaling))),
        RandomRotationWithExpand(degrees=360),
        transforms.CenterCrop((image_size, image_size)),
        transforms.ToTensor()
    ])

    # Enable loading truncated images
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # Sample paired images
    train_filenames_panos, val_filenames_panos = sample_paired_images('/home/lrusso/cvusa', sample_percentage=0.2, split_ratio=0.8, groundtype='panos')
    train_filenames_cutouts, val_filenames_cutouts = sample_paired_images('/home/lrusso/cvusa', sample_percentage=0.2, split_ratio=0.8, groundtype='cutouts')

    # Define the Combined Dataset
    train_dataset_combined = CombinedPairedImagesDataset(train_filenames_panos, train_filenames_cutouts, transform_aerial=transform_aerial, transform_panos=transform_panos, transform_cutouts=transform_cutouts)
    val_dataset_combined = CombinedPairedImagesDataset(val_filenames_panos, val_filenames_cutouts, transform_aerial=transform_aerial, transform_panos=transform_panos, transform_cutouts=transform_cutouts)

    # Define the DataLoaders
    train_dataloader_combined = DataLoader(train_dataset_combined, batch_size=batch_size, shuffle=shuffle, num_workers=8)
    val_dataloader_combined = DataLoader(val_dataset_combined, batch_size=batch_size, shuffle=shuffle, num_workers=8)

    # Train the Model
    train(model, train_dataloader_combined, val_dataloader_combined, device, criterion_A, criterion_G, optimizer, epochs=100, save_path=args.save_path, debug=False)
