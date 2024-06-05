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


def train(model, train_loader, val_loader, device, criterion, optimizer, epochs=1, save_path='untitled', debug=False):
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

    loss_map = False

    for epoch in range(epochs):
        
        model.train()
        running_huber_loss = 0.0
        running_ssim_loss = 0.0

        for images_A, images_G in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            images_A, images_G = images_A.to(device), images_G.to(device)

            # Forward Pass
            reconstructed_A, reconstructed_G, attention_A, attention_G, attended_A, attended_G = model(images_A, images_G)

            # Compute HuberLoss
            loss_A = criterion(reconstructed_A, images_A)
            loss_G = criterion(reconstructed_G, images_G)
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

        train_huber_loss = running_huber_loss / len(train_loader)
        train_ssim_loss = running_ssim_loss / len(train_loader)
        train_huber_losses.append(train_huber_loss)
        train_ssim_losses.append(train_ssim_loss)

        # Validation
        val_huber_loss, val_ssim_loss = validate(model, val_loader, criterion, epoch, epochs, results_path, device)
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
        final_val_loss, final_val_ssim_loss = validate(model, val_loader, criterion, "best", epochs, results_path, device)
        print(f'Best Validation Loss: {final_val_loss:.4f}')


def validate(model, val_loader, criterion, epoch, epochs, results_path, device):
    model.eval()
    val_huber_loss = 0
    val_ssim_loss = 0
    first_batch = True
    skip_attention = True

    with torch.no_grad():
        for images_A, images_G in val_loader:
            images_A, images_G = images_A.to(device), images_G.to(device)

            # Forward Pass
            reconstructed_A, reconstructed_G, attention_A, attention_G, attended_A, attended_G = model(images_A, images_G)

            # Compute HuberLoss
            loss_A = criterion(reconstructed_A, images_A)
            loss_G = criterion(reconstructed_G, images_G)
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
                    # visualize_attention_reconstruction(images_A, reconstructed_A, None, attention_A, None, attended_A, epoch, save_path=os.path.join(results_path, 'aerial', f'epoch_{epoch + 1}_reconstruction_attention.png'))
                    # visualize_attention_reconstruction(images_G, reconstructed_G, None, attention_G, None, attended_G, epoch, save_path=os.path.join(results_path, 'ground', f'epoch_{epoch + 1}_reconstruction_attention.png'))
                    visualize_reconstruction(images_A, reconstructed_A, epoch, save_path=os.path.join(results_path, 'aerial', f'epoch_{epoch + 1}_reconstruction.png'))
                    visualize_reconstruction(images_G, reconstructed_G, epoch, save_path=os.path.join(results_path, 'ground', f'epoch_{epoch + 1}_reconstruction.png'))
                else:
                    # visualize_attention_reconstruction(images_A, reconstructed_A, None, attention_A, None, attended_A, epoch, save_path=os.path.join(results_path, 'aerial', 'best_reconstruction_attention.png'))
                    # visualize_attention_reconstruction(images_G, reconstructed_G, None, attention_G, None, attended_G, epoch, save_path=os.path.join(results_path, 'ground', 'best_reconstruction_attention.png'))
                    visualize_reconstruction(images_A, reconstructed_A, epoch, save_path=os.path.join(results_path, 'aerial', 'best_reconstruction.png'))
                    visualize_reconstruction(images_G, reconstructed_G, epoch, save_path=os.path.join(results_path, 'ground', 'best_reconstruction.png'))

    val_avg_huber_loss = val_huber_loss / len(val_loader)
    val_avg_ssim_loss = val_ssim_loss / len(val_loader)
    return val_avg_huber_loss, val_avg_ssim_loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('save_path', nargs='?', const='untitled', default='untitled', type=str, help='Path to save the model and results')
    parser.add_argument('--save_path', '-s', type=str, help='Path to save the model and results', dest='save_path')
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

    # Optimizer and Loss Function
    learning_rate = 1e-5
    params = [{"params": model.parameters()}]
    weight_decay = 1e-5
    optimizer = optim.Adam(params=params, lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.HuberLoss()
    # criterion = PerceptualLoss()

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

    # Enable loading truncated images
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # Sample paired images
    train_filenames, val_filenames = sample_paired_images('/home/lrusso/cvusa', sample_percentage=0.2, split_ratio=0.8)

    # Define the Datasets
    train_dataset = SampledPairedImagesDataset(train_filenames, transform_aerial=transform_aerial, transform_ground=transform_ground)
    val_dataset = SampledPairedImagesDataset(val_filenames, transform_aerial=transform_aerial, transform_ground=transform_ground)

    # Define the DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)

    train(model, train_dataloader, val_dataloader, device, criterion, optimizer, epochs=100, save_path=args.save_path, debug=False)
