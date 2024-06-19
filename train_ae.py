import torch
import torch.nn as nn
import torch.optim as optim
import os
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor, to_pil_image
import matplotlib.pyplot as plt
import shutil
from utils import *
from model import *
from dataset import *
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import argparse


def train_ae(encoder, decoder, train_loader_panos, val_loader_panos, train_loader_cutouts, val_loader_cutouts, device, criterion, optimizer, epochs=1, save_path='untitled', image_type='ground', debug=False):

    encoder.to(device)
    decoder.to(device)

    model_path = os.path.join('autoencoders', save_path)
    metrics_path = os.path.join('autoencoders', save_path, 'metrics')
    results_path = os.path.join('autoencoders', save_path, 'results')
    os.makedirs('autoencoders', exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(metrics_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    save_dataset_samples(train_loader_panos, os.path.join(model_path, 'training_samples_panos.png'), num_images=16, title='Training Samples')
    save_dataset_samples(val_loader_panos, os.path.join(model_path, 'validation_samples_panos.png'), num_images=16, title='Validation Samples')
    save_dataset_samples(train_loader_cutouts, os.path.join(model_path, 'training_samples_cutouts.png'), num_images=16, title='Training Samples')
    save_dataset_samples(val_loader_cutouts, os.path.join(model_path, 'validation_samples_cutouts.png'), num_images=16, title='Validation Samples')

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

        train_loader = random.choice([train_loader_panos, train_loader_cutouts])

        for images_A, images_G in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):

            # Get images on device
            images_A, images_G = images_A.to(device), images_G.to(device)

            # Forward Pass and Compute Loss
            if image_type == 'aerial':
                encoded_A = encoder(images_A)
                reconstructed_A = decoder(encoded_A)
                huber_loss = criterion(reconstructed_A, images_A)
                ssim_loss_value = ssim_loss(reconstructed_A, images_A)
            elif image_type == 'ground':
                encoded_G = encoder(images_G)
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

        train_huber_loss = running_huber_loss / len(train_loader)
        train_ssim_loss = running_ssim_loss / len(train_loader)
        train_huber_losses.append(train_huber_loss)
        train_ssim_losses.append(train_ssim_loss)

        # Validation
        val_huber_loss, val_ssim_loss = validate(encoder, decoder, val_loader_panos, val_loader_cutouts, criterion, epoch, epochs, results_path, image_type, device)
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
        final_val_loss = validate(encoder, decoder, val_loader_panos, val_loader_cutouts, criterion, "best", epochs, results_path, image_type, device)
        print(f'Best Validation Loss: {final_val_loss:.4f}')


def validate(encoder, decoder, val_loader_panos, val_loader_cutouts, criterion, epoch, epochs, results_path, image_type, device):
    
    encoder.eval()
    decoder.eval()
    val_huber_loss = 0
    val_ssim_loss = 0
    first_batch = True
    skip_attention = True

    val_loader = random.choice([val_loader_panos, val_loader_cutouts])

    with torch.no_grad():
        for images_A, images_G in val_loader:

            # Get images on device
            images_A, images_G = images_A.to(device), images_G.to(device)

            # Forward Pass and Compute Loss
            if image_type == 'aerial':
                encoded_A = encoder(images_A)
                reconstructed_A = decoder(encoded_A)
                huber_loss = criterion(reconstructed_A, images_A)
                ssim_loss_value = ssim_loss(reconstructed_A, images_A)
            elif image_type == 'ground':
                encoded_G = encoder(images_G)
                reconstructed_G = decoder(encoded_G)
                huber_loss = criterion(reconstructed_G, images_G)
                ssim_loss_value = ssim_loss(reconstructed_G, images_G)
            else:
                raise ValueError('Invalid image type. Use either "aerial" or "ground".')
            
            ssim_loss_value = 1 - ssim_loss_value

            val_huber_loss += huber_loss.item()
            val_ssim_loss += ssim_loss_value.item()

            # Visualize Attention Maps and Reconstructions for a batch during validation
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

    parser = argparse.ArgumentParser(description='Train a model.')
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
        RandomHorizontalShiftWithWrap(shift_range=0.2),
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
    train_filenames_panos, val_filenames_panos = sample_paired_images('/home/lrusso/cvusa', sample_percentage=1, split_ratio=0.8, groundtype='panos')
    train_filenames_cutouts, val_filenames_cutouts = sample_paired_images('/home/lrusso/cvusa', sample_percentage=1, split_ratio=0.8, groundtype='cutouts')

    # Define the Datasets
    train_dataset_panos = SampledPairedImagesDataset(train_filenames_panos, transform_aerial=transform_aerial, transform_ground=transform_panos)
    val_dataset_panos = SampledPairedImagesDataset(val_filenames_panos, transform_aerial=transform_aerial, transform_ground=transform_panos)
    train_dataset_cutouts = SampledPairedImagesDataset(train_filenames_cutouts, transform_aerial=transform_aerial, transform_ground=transform_cutouts)
    val_dataset_cutouts = SampledPairedImagesDataset(val_filenames_cutouts, transform_aerial=transform_aerial, transform_ground=transform_cutouts)

    # Define the DataLoaders
    train_dataloader_panos = DataLoader(train_dataset_panos, batch_size=batch_size, shuffle=shuffle, num_workers=8)
    val_dataloader_panos = DataLoader(val_dataset_panos, batch_size=batch_size, shuffle=shuffle, num_workers=8)
    train_dataloader_cutouts = DataLoader(train_dataset_cutouts, batch_size=batch_size, shuffle=shuffle, num_workers=8)
    val_dataloader_cutouts = DataLoader(val_dataset_cutouts, batch_size=batch_size, shuffle=shuffle, num_workers=8)

    # Train the Autoencoder
    train_ae(encoder, decoder, train_dataloader_panos, val_dataloader_panos, train_dataloader_cutouts, val_dataloader_cutouts, device, criterion, optimizer, epochs=100, save_path=args.save_path, image_type=args.image_type, debug=False)
