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

def add_noise(images, noise_factor=0.5):
    noisy_images = images + noise_factor * torch.randn(*images.shape).to(images.device)
    noisy_images = torch.clip(noisy_images, 0., 1.)
    return noisy_images

def train_denoising_ae(encoder, decoder, train_loader, val_loader, device, criterion, optimizer, epochs=1, save_path='untitled', image_type='ground', noise_factor=0.5, debug=False):

    encoder.to(device)
    decoder.to(device)

    model_path = os.path.join('autoencoders', save_path)
    metrics_path = os.path.join('autoencoders', save_path, 'metrics')
    results_path = os.path.join('autoencoders', save_path, 'results')
    os.makedirs('autoencoders', exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(metrics_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    save_dataset_samples(train_loader, os.path.join(model_path, 'training_samples_panos.png'), num_images=16, title='Training Samples')
    save_dataset_samples(val_loader, os.path.join(model_path, 'validation_samples_panos.png'), num_images=16, title='Validation Samples')

    train_huber_losses = []
    val_huber_losses = []
    train_ssim_losses = []
    val_ssim_losses = []
    best_val_loss = np.inf
    patience_counter = 0
    best_encoder_path = None

    for epoch in range(epochs):
        
        encoder.train()
        decoder.train()
        running_huber_loss = 0.0
        running_ssim_loss = 0.0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}', unit='batch') as pbar:
            for images_A, images_G in train_loader:

                # Get images on device
                images_A, images_G = images_A.to(device), images_G.to(device)
                noisy_images_A = add_noise(images_A, noise_factor)
                noisy_images_G = add_noise(images_G, noise_factor)

                # Forward Pass and Compute Loss
                if image_type == 'aerial':
                    encoded_A = encoder(noisy_images_A)
                    reconstructed_A = decoder(encoded_A)
                    huber_loss = criterion(reconstructed_A, images_A)
                    ssim_loss_value = ssim_loss(reconstructed_A, images_A)
                elif image_type == 'ground':
                    encoded_G = encoder(noisy_images_G)
                    reconstructed_G = decoder(encoded_G)
                    huber_loss = criterion(reconstructed_G, images_G)
                    ssim_loss_value = ssim_loss(reconstructed_G, images_G)
                else:
                    raise ValueError('Invalid image type. Use either "aerial" or "ground".')
                
                ssim_loss_value = 1 - ssim_loss_value

                running_huber_loss += huber_loss.item()
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
        val_huber_loss, val_ssim_loss = validate_denoising_ae(encoder, decoder, val_loader, criterion, epoch, epochs, results_path, image_type, device, noise_factor)
        val_huber_losses.append(val_huber_loss)
        val_ssim_losses.append(val_ssim_loss)

        # Check for Best Model
        if val_huber_loss < best_val_loss:
            best_val_loss = val_huber_loss
            patience_counter = 0

            if best_encoder_path is not None and os.path.exists(best_encoder_path):     # delete previous best model if exists
                os.remove(best_encoder_path)
                os.remove(best_decoder_path)

            best_encoder_path = os.path.join(model_path, f'best_encoder_epoch_{epoch+1}.pth')
            best_decoder_path = os.path.join(model_path, f'best_decoder_epoch_{epoch+1}.pth')
            torch.save(encoder.state_dict(), best_encoder_path)               # save the best encoder
            torch.save(decoder.state_dict(), best_decoder_path)               # save the best decoder
        else:
            patience_counter += 1

        # Update the plot with the current losses
        update_plot(epoch + 1, train_huber_losses, val_huber_losses, train_ssim_losses, val_ssim_losses, metrics_path)

    # Save the Model
    torch.save(encoder.state_dict(), os.path.join(model_path, f'last_encoder_epoch_{epoch+1}.pth'))
    torch.save(decoder.state_dict(), os.path.join(model_path, f'last_decoder_epoch_{epoch+1}.pth'))
    print('Training Complete!\nPatience Counter:', patience_counter)

    # Perform additional validation step with the best model
    if best_encoder_path:
        print(f'Loading the model from {best_encoder_path}...')
        encoder.load_state_dict(torch.load(best_encoder_path))
        decoder.load_state_dict(torch.load(best_decoder_path))
        final_val_loss, final_val_ssim_loss = validate_denoising_ae(encoder, decoder, val_loader, criterion, "best", epochs, results_path, image_type, device, noise_factor)
        print(f'Best Validation Loss: {final_val_loss:.4f}')


def validate_denoising_ae(encoder, decoder, val_loader, criterion, epoch, epochs, results_path, image_type, device, noise_factor):
    
    encoder.eval()
    decoder.eval()
    val_huber_loss = 0
    val_ssim_loss = 0
    first_batch = True

    with torch.no_grad():
        for images_A, images_G in val_loader:

            # Get images on device
            images_A, images_G = images_A.to(device), images_G.to(device)
            noisy_images_A = add_noise(images_A, noise_factor)
            noisy_images_G = add_noise(images_G, noise_factor)

            # Forward Pass and Compute Loss
            if image_type == 'aerial':
                encoded_A = encoder(noisy_images_A)
                reconstructed_A = decoder(encoded_A)
                huber_loss = criterion(reconstructed_A, images_A)
                ssim_loss_value = ssim_loss(reconstructed_A, images_A)
            elif image_type == 'ground':
                encoded_G = encoder(noisy_images_G)
                reconstructed_G = decoder(encoded_G)
                huber_loss = criterion(reconstructed_G, images_G)
                ssim_loss_value = ssim_loss(reconstructed_G, images_G)
            else:
                raise ValueError('Invalid image type. Use either "aerial" or "ground".')
            
            ssim_loss_value = 1 - ssim_loss_value

            val_huber_loss += huber_loss.item()
            val_ssim_loss += ssim_loss_value.item()

            # Visualize Reconstructions for a batch during validation
            if first_batch:
                first_batch = False
                if epoch != "best":
                    if image_type == 'aerial':
                        visualize_reconstruction(images_A, reconstructed_A, epoch, save_path=os.path.join(results_path, f'epoch_{epoch + 1}_reconstruction.png'))
                    elif image_type == 'ground':
                        visualize_reconstruction(images_G, reconstructed_G, epoch, save_path=os.path.join(results_path, f'epoch_{epoch + 1}_reconstruction.png'))
                else:
                    if image_type == 'aerial':
                        visualize_reconstruction(images_A, reconstructed_A, epoch, save_path=os.path.join(results_path, 'best_reconstruction.png'))
                    elif image_type == 'ground':
                        visualize_reconstruction(images_G, reconstructed_G, epoch, save_path=os.path.join(results_path, 'best_reconstruction.png'))

    val_avg_huber_loss = val_huber_loss / len(val_loader)
    val_avg_ssim_loss = val_ssim_loss / len(val_loader)
    return val_avg_huber_loss, val_avg_ssim_loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a denoising autoencoder model.')
    parser.add_argument('--save_path', '-p', type=str, default='untitled', help='Path to save the model and results')
    parser.add_argument('--image_type', '-i', type=str, choices=['aerial', 'ground'], required=True, help='Type of images to use (aerial or ground)')
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
    encoder = Encoder(latent_dim=n_encoded).to(device)
    decoder = Decoder(input_dims=n_encoded, hidden_dims=hidden_dims, output_channels=3, initial_size=7).to(device)
    print(encoder, decoder)

    # Optimizer and Loss Function
    learning_rate = 1e-5
    params = [{"params": encoder.parameters()},
              {"params": decoder.parameters()}]
    weight_decay = 1e-5
    optimizer = optim.Adam(params=params, lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.HuberLoss()

    # Transformations
    transform_cutouts = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop((image_size, image_size)),
        transforms.ToTensor()
    ])

    transform_panos = transforms.Compose([
        RandomHorizontalShiftWithWrap(shift_range=1.0),
        transforms.RandomHorizontalFlip(),
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

    # Train the Denoising Autoencoder
    train_denoising_ae(encoder, decoder, train_dataloader_combined, val_dataloader_combined, device, criterion, optimizer, epochs=100, save_path=args.save_path, image_type=args.image_type, debug=False)
