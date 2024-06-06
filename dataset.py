import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from PIL import Image, ImageFile, ImageChops
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


def sample_paired_images(dataset_path, sample_percentage=0.2, split_ratio=0.8, groundtype='panos'):
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
    
    if groundtype == 'panos':
        ground_dir = os.path.join(dataset_path, 'streetview', 'panos')
    elif groundtype == 'cutouts':
        ground_dir = os.path.join(dataset_path, 'streetview', 'cutouts')
    else:   
        raise ValueError("Invalid groundtype. Choose either 'panos' or 'cutouts'.")
    satellite_dir = os.path.join(dataset_path, 'streetview_aerial')

    paired_filenames = []
    for root, _, files in os.walk(ground_dir):
        for file in files:
            if file.endswith('.jpg'):
                ground_path = os.path.join(root, file)
                lat, lon = get_metadata(ground_path)
                if lat is None or lon is None:
                    continue
                zoom = 18  # Only consider zoom level 18
                sat_path = get_aerial_path(satellite_dir, lat, lon, zoom)
                if os.path.exists(sat_path):
                    paired_filenames.append((ground_path, sat_path))
    
    num_to_select = int(len(paired_filenames) * sample_percentage)
    selected_filenames = random.sample(paired_filenames, num_to_select)
    
    random.shuffle(selected_filenames)
    split_point = int(split_ratio * len(selected_filenames))
    train_filenames = selected_filenames[:split_point]
    val_filenames = selected_filenames[split_point:]

    return train_filenames, val_filenames


def get_metadata(fname):
    if 'streetview' in fname:
        parts = fname[:-4].rsplit('/', 1)[1].split('_')
        if len(parts) == 2:
            lat, lon = parts
            return lat, lon
        elif len(parts) == 3:
            lat, lon, orientation = parts
            return lat, lon
        else:
            print(f"Unexpected filename format: {fname}")
            return None, None
    return None


def get_aerial_path(root_dir, lat, lon, zoom):
    lat_bin = int(float(lat))
    lon_bin = int(float(lon))
    return os.path.join(root_dir, f'{zoom}/{lat_bin}/{lon_bin}/{lat}_{lon}.jpg')


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
    

class RandomHorizontalShiftWithWrap:
    def __init__(self, shift_range):
        self.shift_range = shift_range

    def __call__(self, img):
        shift = np.random.uniform(-self.shift_range, self.shift_range) * img.width
        return ImageChops.offset(img, int(shift), 0)

class RandomRotationWithExpand:
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, img):
        angle = np.random.uniform(-self.degrees, self.degrees)
        return img.rotate(angle, resample=Image.BICUBIC, expand=True)