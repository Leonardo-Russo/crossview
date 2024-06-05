import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from PIL import Image
from torchvision import transforms
import torchvision.models as models
from torchvision.models import VGG16_Weights
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import shutil
import math
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import numpy as np
import matplotlib.pyplot as plt
import random


def plot_metrics(epochs, train_loss, val_loss, val_psnr, val_ssim, path='training_plots'):
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(2, 2, 2)
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(2, 2, 3)
    plt.plot(epochs, val_psnr, label='Validation PSNR')
    plt.title('Validation PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')

    plt.subplot(2, 2, 4)
    plt.plot(epochs, val_ssim, label='Validation SSIM')
    plt.title('Validation SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')

    plt.tight_layout()
    
    # Ensure the directory exists
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, f'training_metrics_epoch_{len(epochs)}.png'))
    plt.close()  # Close the figure to free memory

    
def save_model(model, path):
    torch.save(model.state_dict(), path)


def visualize_reconstruction(original, reconstructed, epoch, save_path=None, num_images=16):
    """
    Visualize a comparison of original, reconstructed images, and attention maps in a grid format.

    Parameters:
    - original (torch.Tensor): The original images tensor.
    - reconstructed (torch.Tensor): The reconstructed images tensor.
    - attention_maps (torch.Tensor): The attention maps tensor.
    - epoch (int): Current epoch number for titling the plot.
    - save_path (str, optional): Path to save the resulting plot. If None, the plot is displayed.
    - num_images (int): Number of images to display from the batch.
    """
    # Ensure that we do not exceed the number of images in the batch
    num_images = min(num_images, original.size(0))

    # Randomly select indices for display
    indices = torch.randperm(original.size(0))[:num_images]

    # Select the images from the tensors
    selected_original = original[indices].cpu()
    selected_reconstructed = reconstructed[indices].cpu()

    # Create grids
    original_grid = make_grid(selected_original, nrow=int(num_images**0.5), normalize=False)
    reconstructed_grid = make_grid(selected_reconstructed, nrow=int(num_images**0.5), normalize=False)

    # Convert to numpy arrays
    original_npimg = original_grid.numpy().transpose((1, 2, 0))
    reconstructed_npimg = reconstructed_grid.numpy().transpose((1, 2, 0))

    # Create figure and subplots
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(original_npimg)
    if epoch != "best":
        plt.title(f'Epoch: {epoch + 1} - Original Images')
    else:
        plt.title(f'Best Model - Original Images')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_npimg)
    if epoch != "best":
        plt.title(f'Epoch: {epoch + 1} - Reconstructed Images')
    else:
        plt.title(f'Best Model - Reconstructed Images')
    plt.axis('off')

    # Save or show the image
    if save_path != None:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()


def visualize_reconstruction(original, reconstructed, epoch, save_path=None, num_images=16):
    """
    Visualize a comparison of original and reconstructed images in a grid format.

    Parameters:
    - original (torch.Tensor): The original images tensor.
    - reconstructed (torch.Tensor): The reconstructed images tensor.
    - epoch (int): Current epoch number for titling the plot.
    - save_path (str, optional): Path to save the resulting plot. If None, the plot is displayed.
    - num_images (int): Number of images to display from the batch.
    """
    # Ensure that we do not exceed the number of images in the batch
    num_images = min(num_images, original.size(0), reconstructed.size(0))

    # Randomly select indices for display
    indices = torch.randperm(original.size(0))[:num_images]

    # Select the images from the tensors
    selected_original = original[indices].cpu()
    selected_reconstructed = reconstructed[indices].cpu()

    # Create grids
    original_grid = make_grid(selected_original, nrow=int(num_images**0.5), normalize=True)
    reconstructed_grid = make_grid(selected_reconstructed, nrow=int(num_images**0.5), normalize=True)

    # Convert to numpy arrays
    original_npimg = original_grid.numpy().transpose((1, 2, 0))
    reconstructed_npimg = reconstructed_grid.numpy().transpose((1, 2, 0))

    # Create figure and subplots
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(original_npimg)
    plt.title(f'Epoch: {epoch + 1} - Original Images')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_npimg)
    plt.title(f'Epoch: {epoch + 1} - Reconstructed Images')
    plt.axis('off')

    # Save or show the image
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()


def save_dataset_samples(dataloader, save_path=None, num_images=16, title="Dataset Images"):
    """
    This function displays two grids of paired images from the provided DataLoader.
    
    Parameters:
        dataloader (DataLoader): The DataLoader from which to load the images.
        title (str): The title of the plot.
        num_images (int): The number of images to display in the grid.
    """
    # Get a batch of paired images
    paired_images = next(iter(dataloader))
    if isinstance(paired_images, list) and len(paired_images) == 2:
        images_A, images_B = paired_images
    else:
        raise ValueError("The dataloader should return a list of (images_A, images_B)")

    # Ensure that we do not exceed the number of images in the batch
    num_images = min(num_images, images_A.size(0), images_B.size(0))

    # Randomly select indices for display
    indices = torch.randperm(images_A.size(0))[:num_images]

    # Select the images from the tensors
    selected_images_A = images_A[indices]
    selected_images_B = images_B[indices]

    # Create grids from the selected images
    grid_img_A = make_grid(selected_images_A, nrow=int(math.sqrt(num_images)), normalize=True, scale_each=True)
    grid_img_B = make_grid(selected_images_B, nrow=int(math.sqrt(num_images)), normalize=True, scale_each=True)

    npimg_A = grid_img_A.numpy().transpose((1, 2, 0))
    npimg_B = grid_img_B.numpy().transpose((1, 2, 0))

    # Create a filename from the title
    if save_path is None:
        save_path = title.lower().replace(" ", "_") + '.png'
    
    plt.figure(figsize=(20, 10))

    # Display ground-level images
    plt.subplot(1, 2, 1)
    plt.imshow(npimg_A)
    plt.title('Ground Level Images')
    plt.axis('off')

    # Display aerial images
    plt.subplot(1, 2, 2)
    plt.imshow(npimg_B)
    plt.title('Aerial Images')
    plt.axis('off')

    plt.suptitle(title)

    plt.savefig(save_path)
    plt.close()


def psnr(target, output, max_val=1.0):
    # target and output are torch.Tensors
    target_np = target.detach().cpu().numpy()
    output_np = output.detach().cpu().numpy()
    scores = [psnr_metric(t, o, data_range=max_val) for t, o in zip(target_np, output_np)]
    return np.mean(scores)


def sample_paired_images(dataset_path, sample_percentage=0.2, split_ratio=0.8):
    """
    Function to sample a percentage of the dataset and split it into training and validation sets.
    
    Parameters:
        dataset_path (str): Path to the dataset root directory.
        sample_percentage (float): Percentage of the dataset to sample.
        split_ratio (float): Ratio to split the sampled data into training and validation sets.
        
    Returns:
        train_filenames (list): List of training filenames (tuples of panorama and satellite image paths).
        val_filenames (list): List of validation filenames (tuples of panorama and satellite image paths).
    """
    panorama_dir = os.path.join(dataset_path, 'streetview', 'panos')
    satellite_dir = os.path.join(dataset_path, 'streetview_aerial')

    paired_filenames = []
    for root, _, files in os.walk(panorama_dir):
        for file in files:
            if file.endswith('.jpg'):
                pano_path = os.path.join(root, file)
                lat, lon = get_metadata(pano_path)
                if lat is None or lon is None:
                    continue
                zoom = 18  # Only consider zoom level 18
                sat_path = get_aerial_path(satellite_dir, lat, lon, zoom)
                if os.path.exists(sat_path):
                    paired_filenames.append((pano_path, sat_path))
    
    num_to_select = int(len(paired_filenames) * sample_percentage)
    selected_filenames = random.sample(paired_filenames, num_to_select)
    
    random.shuffle(selected_filenames)
    split_point = int(split_ratio * len(selected_filenames))
    train_filenames = selected_filenames[:split_point]
    val_filenames = selected_filenames[split_point:]

    return train_filenames, val_filenames


def update_plot(epoch, train_huber_losses, val_huber_losses, train_ssim_losses, val_ssim_losses, save_path):
    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epoch + 1), train_huber_losses, label='Training Huber Loss')
    plt.plot(range(1, epoch + 1), val_huber_losses, label='Validation Huber Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Huber Loss')
    plt.legend()
    plt.title('Training and Validation Huber Loss over Epochs')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epoch + 1), train_ssim_losses, label='Training SSIM Loss')
    plt.plot(range(1, epoch + 1), val_ssim_losses, label='Validation SSIM Loss')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM Loss (1 - SSIM)')
    plt.legend()
    plt.title('Training and Validation SSIM Loss over Epochs')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'loss_plot.png'))
    plt.close()



def get_metadata(fname):
    if 'streetview' in fname:
        parts = fname[:-4].rsplit('/', 1)[1].split('_')
        if len(parts) == 2:
            lat, lon = parts
            return lat, lon
        else:
            print(f"Unexpected filename format: {fname}")
            return None, None
    return None


def get_aerial_path(root_dir, lat, lon, zoom):
    lat_bin = int(float(lat))
    lon_bin = int(float(lon))
    return os.path.join(root_dir, f'{zoom}/{lat_bin}/{lon_bin}/{lat}_{lon}.jpg')


def visualize_attention_reconstruction(original, reconstructed, loss_maps, attention_maps, attended_loss_maps, attended, epoch, save_path=None, num_images=16):
    """
    Visualize a comparison of original, reconstructed images, and attention maps in a grid format.

    Parameters:
    - original (torch.Tensor): The original images tensor.
    - reconstructed (torch.Tensor): The reconstructed images tensor.
    - attention_maps (torch.Tensor): The attention maps tensor.
    - epoch (int): Current epoch number for titling the plot.
    - save_path (str, optional): Path to save the resulting plot. If None, the plot is displayed.
    - num_images (int): Number of images to display from the batch.
    """
    # Ensure that we do not exceed the number of images in the batch
    num_images = min(num_images, original.size(0), reconstructed.size(0), loss_maps.size(0), attention_maps.size(0), attended_loss_maps.size(0), attended.size(0))

    # Randomly select indices for display
    indices = torch.randperm(original.size(0))[:num_images]

    # Select the images from the tensors
    selected_original = original[indices].cpu()
    selected_reconstructed = reconstructed[indices].cpu()
    selected_attention_maps = attention_maps[indices].cpu()
    selected_loss_maps = loss_maps[indices].cpu()
    selected_attended_loss_maps = attended_loss_maps[indices].cpu()
    selected_attended = attended[indices].cpu()

    # Ensure the attention maps are correctly shaped for grayscale display
    if selected_loss_maps.dim() == 4 and selected_loss_maps.size(1) == 1:
        selected_loss_maps = selected_loss_maps.squeeze(1)                  # remove the channel dimension
    if selected_attention_maps.dim() == 4 and selected_attention_maps.size(1) == 1:
        selected_attention_maps = selected_attention_maps.squeeze(1)
    if selected_attended_loss_maps.dim() == 4 and selected_attended_loss_maps.size(1) == 1:
        selected_attended_loss_maps = selected_attended_loss_maps.squeeze(1)

    # Create grids
    original_grid = make_grid(selected_original, nrow=int(num_images**0.5), normalize=False)
    reconstructed_grid = make_grid(selected_reconstructed, nrow=int(num_images**0.5), normalize=False)
    loss_grid = make_grid(selected_loss_maps.unsqueeze(1), nrow=int(num_images**0.5), normalize=False)
    attention_grid = make_grid(selected_attention_maps.unsqueeze(1), nrow=int(num_images**0.5), normalize=False)
    attended_loss_grid = make_grid(selected_attended_loss_maps.unsqueeze(1), nrow=int(num_images**0.5), normalize=False)
    attended_grid = make_grid(selected_attended, nrow=int(num_images**0.5), normalize=False)

    # Convert to numpy arrays
    original_npimg = original_grid.numpy().transpose((1, 2, 0))
    reconstructed_npimg = reconstructed_grid.numpy().transpose((1, 2, 0))
    loss_npimg = loss_grid.numpy().transpose((1, 2, 0))[:, :, 0]            # ensure it's single-channel for grayscale
    attention_npimg = attention_grid.numpy().transpose((1, 2, 0))[:, :, 0]
    attended_loss_npimg = attended_loss_grid.numpy().transpose((1, 2, 0))[:, :, 0]
    attended_npimg = attended_grid.numpy().transpose((1, 2, 0))

    # Create figure and subplots
    plt.figure(figsize=(24, 16))
    plt.subplot(2, 3, 1)
    plt.imshow(original_npimg)
    if epoch != "best":
        plt.title(f'Epoch: {epoch + 1} - Original Images')
    else:
        plt.title(f'Best Model - Original Images')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(reconstructed_npimg)
    if epoch != "best":
        plt.title(f'Epoch: {epoch + 1} - Reconstructed Images')
    else:
        plt.title(f'Best Model - Reconstructed Images')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(loss_npimg, cmap='gray')
    if epoch != "best":
        plt.title(f'Epoch: {epoch + 1} - Loss Maps')
    else:
        plt.title(f'Best Model - Loss Maps')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(attention_npimg, cmap='gray')
    if epoch != "best":
        plt.title(f'Epoch: {epoch + 1} - Attention Maps')
    else:
        plt.title(f'Best Model - Attention Maps')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(attended_loss_npimg, cmap='gray')
    if epoch != "best":
        plt.title(f'Epoch: {epoch + 1} - Attended Loss Maps')
    else:
        plt.title(f'Best Model - Attended Loss Maps')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(attended_npimg)
    if epoch != "best":
        plt.title(f'Epoch: {epoch + 1} - Attended Images')
    else:
        plt.title(f'Best Model - Attended Images')
    plt.axis('off')

    # Save or show the image
    if save_path != None:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = models.vgg16(weights=VGG16_Weights.DEFAULT).features[:16]
        for param in self.vgg.parameters():
            param.requires_grad = False     # freeze VGG layers

    def forward(self, reconstructed, original):

        # Compute features and the loss
        reconstructed_features = self.vgg(reconstructed)
        target_features = self.vgg(original)
        loss = nn.functional.l1_loss(reconstructed_features, target_features)
        
        return loss


def attention_regularization(attention_map):
    # Compute the total variation loss for the attention map
    diff_i = torch.mean(torch.abs(attention_map[:, :, :, :-1] - attention_map[:, :, :, 1:]))
    diff_j = torch.mean(torch.abs(attention_map[:, :, :-1, :] - attention_map[:, :, 1:, :]))
    return diff_i + diff_j
    

def freeze(model):

    for param in model.parameters():
        param.requires_grad = False


def transfer_decoder(decoder, pretrained_decoder, n_encoded, hidden_dims, initial_size):
    # Create new weight and bias tensors without gradients
    new_fc0_weight = torch.empty_like(decoder.fc[0].weight)
    new_fc0_bias = torch.empty_like(decoder.fc[0].bias)
    new_fc2_weight = torch.empty_like(decoder.fc[2].weight)
    new_fc2_bias = torch.empty_like(decoder.fc[2].bias)

    # Initialize the new fc layers
    with torch.no_grad():
        # Copy pretrained weights for the first n_encoded elements
        new_fc0_weight[:2*n_encoded, :n_encoded] = pretrained_decoder.fc[0].weight
        new_fc0_bias[:2*n_encoded] = pretrained_decoder.fc[0].bias

        # Initialize the new n_phi neurons randomly (this is already done by torch.empty_like above)
        new_fc2_weight[:, :2*n_encoded] = pretrained_decoder.fc[2].weight
        new_fc2_bias[:] = pretrained_decoder.fc[2].bias

    # Assign the new weights and biases to the decoder
    decoder.fc[0].weight = nn.Parameter(new_fc0_weight)
    decoder.fc[0].bias = nn.Parameter(new_fc0_bias)
    decoder.fc[2].weight = nn.Parameter(new_fc2_weight)
    decoder.fc[2].bias = nn.Parameter(new_fc2_bias)

    # Copy the remaining layers from the pretrained decoder
    decoder.unflatten = pretrained_decoder.unflatten
    decoder.upsample = pretrained_decoder.upsample

    return decoder


def ssim_loss(img1, img2):
    # SSIM expects images in [0, 255] range, convert images to this range
    img1 = (img1 * 255.0).clamp(0, 255).byte()
    img2 = (img2 * 255.0).clamp(0, 255).byte()
    # Convert images to numpy
    img1 = img1.cpu().numpy().transpose((0, 2, 3, 1))
    img2 = img2.cpu().numpy().transpose((0, 2, 3, 1))
    # Compute SSIM for each image in the batch
    ssim_values = [ssim_metric(i1, i2, multichannel=True, channel_axis=-1, win_size=7) for i1, i2 in zip(img1, img2)]
    # Convert list to tensor on the appropriate device
    return torch.tensor(ssim_values, device='cuda' if torch.cuda.is_available() else 'cpu').mean()


class PairedImagesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Dataset to load paired satellite and panorama images.
        Args:
            root_dir (str): Root directory containing the 'streetview' and 'streetview_aerial' subdirectories.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.panorama_dir = os.path.join(root_dir, 'streetview', 'panos')
        self.satellite_dir = os.path.join(root_dir, 'streetview_aerial')
        self.transform = transform

        self.paired_filenames = []
        for root, _, files in os.walk(self.panorama_dir):
            for file in files:
                if file.endswith('.jpg'):
                    pano_path = os.path.join(root, file)
                    lat, lon = get_metadata(pano_path)
                    if lat is None or lon is None:
                        continue
                    zoom = 18  # Only consider zoom level 18
                    sat_path = get_aerial_path(self.satellite_dir, lat, lon, zoom)
                    if os.path.exists(sat_path):
                        self.paired_filenames.append((pano_path, sat_path))
        
        if len(self.paired_filenames) == 0:
            print(f"Check if the directory paths are correct and accessible: {self.panorama_dir} and {self.satellite_dir}")

    def __len__(self):
        return len(self.paired_filenames)

    def __getitem__(self, idx):
        pano_img_path, sat_img_path = self.paired_filenames[idx]

        pano_image = Image.open(pano_img_path).convert('RGB')
        sat_image = Image.open(sat_img_path).convert('RGB')

        if self.transform:
            pano_image = self.transform(pano_image)
            sat_image = self.transform(sat_image)

        return pano_image, sat_image
    

# Define the Datasets using sampled filenames
class SampledPairedImagesDataset(Dataset):
    def __init__(self, filenames, transform_aerial=None, transform_ground=None):
        self.filenames = filenames
        self.transform_aerial = transform_aerial
        self.transform_ground = transform_ground

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        pano_img_path, sat_img_path = self.filenames[idx]

        pano_image = Image.open(pano_img_path).convert('RGB')
        sat_image = Image.open(sat_img_path).convert('RGB')

        if self.transform_aerial:
            pano_image = self.transform_ground(pano_image)

        if self.transform_ground:
            sat_image = self.transform_aerial(sat_image)

        return sat_image, pano_image