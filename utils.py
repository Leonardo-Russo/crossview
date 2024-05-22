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


def update_plot(epoch, train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epoch + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epoch + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss over Epochs')
    plt.grid(True)
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


def visualize_attention_reconstruction(original, reconstructed, loss_maps, attention_maps, epoch, save_path=None, num_images=16):
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
    num_images = min(num_images, original.size(0), reconstructed.size(0), loss_maps.size(0), attention_maps.size(0))

    # Randomly select indices for display
    indices = torch.randperm(original.size(0))[:num_images]

    # Select the images from the tensors
    selected_original = original[indices].cpu()
    selected_reconstructed = reconstructed[indices].cpu()
    selected_attention_maps = attention_maps[indices].cpu()
    selected_loss_maps = loss_maps[indices].cpu()

    # Ensure the attention maps are correctly shaped for grayscale display
    if selected_loss_maps.dim() == 4 and selected_loss_maps.size(1) == 1:
        selected_loss_maps = selected_loss_maps.squeeze(1)                  # remove the channel dimension
    if selected_attention_maps.dim() == 4 and selected_attention_maps.size(1) == 1:
        selected_attention_maps = selected_attention_maps.squeeze(1)

    # Create grids
    original_grid = make_grid(selected_original, nrow=int(num_images**0.5), normalize=True)
    reconstructed_grid = make_grid(selected_reconstructed, nrow=int(num_images**0.5), normalize=True)
    loss_grid = make_grid(selected_loss_maps.unsqueeze(1), nrow=int(num_images**0.5), normalize=True)
    attention_grid = make_grid(selected_attention_maps.unsqueeze(1), nrow=int(num_images**0.5), normalize=False)

    # Convert to numpy arrays
    original_npimg = original_grid.numpy().transpose((1, 2, 0))
    reconstructed_npimg = reconstructed_grid.numpy().transpose((1, 2, 0))
    loss_npimg = loss_grid.numpy().transpose((1, 2, 0))[:, :, 0]            # ensure it's single-channel for grayscale
    attention_npimg = attention_grid.numpy().transpose((1, 2, 0))[:, :, 0]

    # Create figure and subplots
    plt.figure(figsize=(24, 16))
    plt.subplot(2, 2, 1)
    plt.imshow(original_npimg)
    plt.title(f'Epoch: {epoch + 1} - Original Images')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(reconstructed_npimg)
    plt.title(f'Epoch: {epoch + 1} - Reconstructed Images')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(loss_npimg, cmap='gray')
    plt.title(f'Epoch: {epoch + 1} - Loss Maps')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(attention_npimg, cmap='gray')
    plt.title(f'Epoch: {epoch + 1} - Attention Maps')
    plt.axis('off')

    # Save or show the image
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()


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


class ViTEncoder(nn.Module):
    def __init__(self, out_features=100, model_name='dinov2_vits14_reg_lc'):
        super(ViTEncoder, self).__init__()
        
        # # Load the ViT from timm
        # self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)

        # Load the DINOv2 from Facebook Research
        self.vit = torch.hub.load('facebookresearch/dinov2', model_name)

        for param in self.vit.parameters():
            param.requires_grad = False         # freeze all parameters of the Vision Transformer

        # # Retrive the number of features in the last layer of the ViT
        # linear_features = self.vit.num_features    # get the number of features in the model
        
        # Retrieve the number of features in the last layer of DINOv2
        linear_features = self.vit.linear_head.in_features

        # # Remove the classifier head from the ViT
        # self.vit.head = nn.Identity()
                
        # Remove the classifier head from DINOv2
        self.vit.linear_head = nn.Identity()

        self.reducer = nn.Sequential(
            nn.Linear(linear_features, (linear_features + out_features) // 2),      # floored average between input and output features
            nn.ELU(True),
            nn.Linear((linear_features + out_features) // 2, out_features)
        )

    def forward(self, x):
        x = self.vit(x)
        x = self.reducer(x)     # reduce the feature to (out_features x 1)
        return x


class Encoder(nn.Module):

	def __init__(self, latent_dim):
		super(Encoder, self).__init__()

		self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ELU(True),
            nn.Conv2d(512, 1024, 3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ELU(True)                # 3x224x224 -> 64x112x112 -> 128x56x56 -> 256x28x28 -> 512x14x14 -> 1024x7x7
        )

		self.flatten = nn.Flatten(start_dim=1)

		self.fc = nn.Sequential(
			nn.Linear(1024*7*7, 5000),
			nn.ELU(True),
			nn.Linear(5000, latent_dim)
		)

	def forward(self, x):
		x = self.cnn(x)
		# print("Encoder CNN Output Size: ", x.shape)
		x = self.flatten(x)
		x = self.fc(x)
		return x
   

class Decoder(nn.Module):
    def __init__(self, input_dims=100, hidden_dims=1024, output_channels=4, initial_size=7, image_size=224):
        super(Decoder, self).__init__()

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_channels = output_channels
        self.initial_size = initial_size
        self.image_size = image_size

        self.fc = nn.Sequential(
            nn.Linear(input_dims, 5000),
            nn.ELU(True),
            nn.Linear(5000, hidden_dims * initial_size * initial_size)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(hidden_dims, initial_size, initial_size))

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims, hidden_dims // 2, kernel_size=3, stride=2, padding=1, output_padding=1),  # 1024x7x7 -> 512x14x14
            nn.BatchNorm2d(hidden_dims // 2),
            nn.ELU(True),
            nn.ConvTranspose2d(hidden_dims // 2, hidden_dims // 4, kernel_size=3, stride=2, padding=1, output_padding=1),  # 512x14x14 -> 256x28x28
            nn.BatchNorm2d(hidden_dims // 4),
            nn.ELU(True),
            nn.ConvTranspose2d(hidden_dims // 4, hidden_dims // 8, kernel_size=3, stride=2, padding=1, output_padding=1),  # 256x28x28 -> 128x56x56
            nn.BatchNorm2d(hidden_dims // 8),
            nn.ELU(True),
            nn.ConvTranspose2d(hidden_dims // 8, hidden_dims // 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 128x56x56 -> 64x112x112
            nn.BatchNorm2d(hidden_dims // 16),
            nn.ELU(True),
            nn.ConvTranspose2d(hidden_dims // 16, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1)  # 64x112x112 -> 4x224x224
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.unflatten(x)
        x = self.upsample(x)
        image = x[:, :3, :, :]          # first 3 channels for RGB image
        image = torch.sigmoid(image)
        attention_map = x[:, 3, :, :] + 100  # last channel for attention map
        # print(attention_map.mean())

        # Reshape attention map for softmax
        batch_size = attention_map.size(0)
        attention_map_flat = attention_map.view(batch_size, -1)  # [batch_size, image_size^2]

        # print("Batch Size: ", batch_size)
        # print("Attention Map Size: ", attention_map_flat.shape)
        # print("Attention Map: ", attention_map_flat[0, :10])

        # Apply softmax
        attention_map_flat = torch.softmax(attention_map_flat, dim=1) * self.image_size * self.image_size
        # print("Attention Map: ", attention_map_flat[0, :10])

        # Reshape back to image_size x image_size
        attention_map = attention_map_flat.view(batch_size, 1, self.image_size, self.image_size)

        # print("Decoder Output Size: ", image.shape, attention_map.shape)

        return image, attention_map

    

class MLP(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dims, (input_dims + output_dims) // 2),
            nn.ELU(True),
            nn.Linear((input_dims + output_dims) // 2, output_dims)
        )

    def forward(self, x):
        return self.fc(x)


class CrossView(nn.Module):
    def __init__(self, n_phi, n_encoded, hidden_dims, image_size, output_channels=3, debug=False):
        super(CrossView, self).__init__()

        self.encoder_A = Encoder(latent_dim=n_encoded)
        self.encoder_G = Encoder(latent_dim=n_encoded)
        self.mlp = MLP(input_dims=2*n_encoded, output_dims=n_phi)
        self.decoder_A2G = Decoder(input_dims=n_phi+n_encoded, hidden_dims=hidden_dims, output_channels=output_channels, initial_size=7)
        self.decoder_G2A = Decoder(input_dims=n_phi+n_encoded, hidden_dims=hidden_dims, output_channels=output_channels, initial_size=7)
        self.image_size = image_size
        self.debug = debug
    
    def forward(self, images_A, images_G):

        # Encode images A and G
        encoded_A = self.encoder_A(images_A)
        encoded_G = self.encoder_G(images_G)

        # Concatenate and process through MLP
        phi = self.mlp(torch.cat((encoded_A, encoded_G), dim=-1))

        # Decode the MLP output into reconstructed images
        reconstructed_A, attention_A = self.decoder_G2A(torch.cat((phi, encoded_G), dim=1))
        reconstructed_G, attention_G = self.decoder_A2G(torch.cat((phi, encoded_A), dim=1))

        # Print shapes for debugging
        if self.debug:
            print(f"Encoded A shape: {encoded_A.shape}, Encoded G shape: {encoded_G.shape}, "
                  f"Phi shape: {phi.shape}, Concat Phi with Encoded G shape: {torch.cat((phi, encoded_G), dim=1).shape}")
            
        return reconstructed_A, reconstructed_G, attention_A, attention_G
