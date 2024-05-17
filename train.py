import torch
import torch.nn as nn
import torch.optim as optim
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor, to_pil_image
import matplotlib.pyplot as plt
import shutil
import timm
from utils import *
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import argparse


def train(model, train_loader, val_loader, device, criterion, optimizer, epochs=1, save_path='untitled'):

    model.to(device)

    model_path = os.path.join('models', save_path)
    metrics_path = os.path.join('models', save_path, 'metrics')
    results_path = os.path.join('models', save_path, 'results')
    os.makedirs('models', exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(metrics_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    save_dataset_samples(train_dataloader, os.path.join(model_path, 'training_samples.png'), num_images=16, title='Training Samples')
    save_dataset_samples(val_dataloader, os.path.join(model_path, 'validation_samples.png'), num_images=16, title='Validation Samples')

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        
        model.train()
        running_loss = 0.0

        for images_A, images_G in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):

            # Get images on device
            images_A, images_G = images_A.to(device), images_G.to(device)

            # Forward Pass
            reconstructed_A, reconstructed_G = model(images_A, images_G)

            # Compute loss for both reconstructions
            loss_A = criterion(reconstructed_A, images_A)
            loss_G = criterion(reconstructed_G, images_G)
            total_loss = loss_A + loss_G
            running_loss += total_loss.item()

            # Reset Gradients
            optimizer.zero_grad()
            
            # Backward Propagation and Optimization Step
            total_loss.backward()
            optimizer.step()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        print(f'Epoch {epoch+1}/{epochs}: Training Loss = {train_loss:.4f}')    

        # Validation
        val_loss = validate(model, val_loader, criterion, epoch, epochs, results_path, device)
        val_losses.append(val_loss)

        # Update the plot with the current losses
        update_plot(epoch + 1, train_losses, val_losses, metrics_path)

    # Save the Model
    torch.save(model.state_dict(), os.path.join(model_path, f'crossview_epoch_{epoch+1}.pth'))


def validate(model, val_loader, criterion, epoch, epochs, results_path, device):
    
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for images_A, images_G in val_loader:

            # Get images on device
            images_A, images_G = images_A.to(device), images_G.to(device)

            # Forward Pass
            reconstructed_A, reconstructed_G = model(images_A, images_G)

            # Compute loss for both reconstructions
            loss_A = criterion(reconstructed_A, images_A)
            loss_G = criterion(reconstructed_G, images_G)
            total_loss = loss_A + loss_G
            val_loss += total_loss.item()

    val_avg_loss = val_loss / len(val_loader)

    print(f'Epoch {epoch+1}/{epochs}: Validation Loss = {val_avg_loss:.4f}')

    visualize_reconstruction(images_A, reconstructed_A, epoch, save_path=os.path.join(results_path, 'aerial'))
    visualize_reconstruction(images_G, reconstructed_G, epoch, save_path=os.path.join(results_path, 'ground'))

    return val_avg_loss



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('save_path', type=str, help='Path to save the model and results')
    args = parser.parse_args()

    # Constants
    image_channels = 3          # for RGB images
    image_size = 224            # assuming square images
    aerial_scaling = 3          # scaling factor for aerial images
    hidden_dims = 512           # hidden dimensions
    n_encoded = 1000            # output size for the encoders
    n_phi = 10                  # size of phi
    batch_size = 64
    shuffle = True

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    # Initialize the Architecture
    model = CrossView(n_phi, n_encoded, hidden_dims).to(device)
    print(model)

    # Optimizer and Loss Function
    learning_rate = 1e-5
    params = [{"params": model.parameters()}]
    weight_decay = 1e-5
    optimizer = optim.Adam(params=params, lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.HuberLoss()

    # Transformations
    transform_ground = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop((image_size, image_size)),
        transforms.ToTensor()
    ])


    transform_aerial = transforms.Compose([
        transforms.Resize((int(image_size*aerial_scaling), int(image_size*aerial_scaling))),
        transforms.CenterCrop((image_size, image_size)),
        transforms.ToTensor()
    ])

    transform_aug = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop((image_size, image_size)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.5),
        transforms.ToTensor()
    ])

    # Sample paired images
    train_filenames, val_filenames = sample_paired_images('/home/lrusso/cvusa', sample_percentage=0.2, split_ratio=0.8)

    # Define the Datasets
    train_dataset = SampledPairedImagesDataset(train_filenames, transform_aerial=transform_aerial, transform_ground=transform_ground)
    val_dataset = SampledPairedImagesDataset(val_filenames, transform_aerial=transform_aerial, transform_ground=transform_ground)

    # Define the DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)

    train(model, train_dataloader, val_dataloader, device, criterion, optimizer, epochs=100, save_path=args.save_path)
