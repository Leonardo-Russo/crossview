import torch
import torch.nn as nn
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
        os.makedirs(save_path, exist_ok=True)
        file_name = f'epoch_{epoch + 1}_reconstruction.png'
        plt.savefig(os.path.join(save_path, file_name))
    else:
        plt.show()

    plt.close()



def save_dataset_samples(dataloader, save_path=None, num_images=16, title="Dataset Images"):
    """
    This function displays a grid of images from the provided DataLoader.
    
    Parameters:
        dataloader (DataLoader): The DataLoader from which to load the images.
        title (str): The title of the plot.
        num_images (int): The number of images to display in the grid.
    """
    # Attempt to get a batch of paired images
    try:
        aerial_images, ground_images = next(iter(dataloader))
    except ValueError as e:
        print("Failed to unpack images from the DataLoader:", e)
        return
    
    # Select a subset of images
    if aerial_images.size(0) > num_images:
        indices = torch.randperm(aerial_images.size(0))[:num_images]
        aerial_images = aerial_images[indices]
        ground_images = ground_images[indices]

    # Concatenate the images from the two domains for display
    images = torch.cat((aerial_images, ground_images), dim=0)

    # Make a grid from the images
    grid_img = make_grid(images, nrow=int(math.sqrt(images.size(0))), normalize=True)
    npimg = grid_img.numpy().transpose((1, 2, 0))

    # Create a filename from the title
    if save_path is None:
        save_path = title.lower().replace(" ", "_") + '.png'
    
    plt.figure(figsize=(15, 7.5))  # Adjust the size appropriately
    plt.imshow(npimg)
    plt.title(title)
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()




def combined_loss(output, target, alpha=1, beta=1):

    huber_loss = nn.HuberLoss()
    perceptual_loss = PerceptualLoss()

    return alpha * huber_loss(output, target) + beta * perceptual_loss(output, target)


def psnr(target, output, max_val=1.0):
    # target and output are torch.Tensors
    target_np = target.detach().cpu().numpy()
    output_np = output.detach().cpu().numpy()
    scores = [psnr_metric(t, o, data_range=max_val) for t, o in zip(target_np, output_np)]
    return np.mean(scores)



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
    

class CombinedLoss(nn.Module):
    def __init__(self, device):
        super(CombinedLoss, self).__init__()
        self.huber_loss = nn.HuberLoss()
        self.perceptual_loss = PerceptualLoss().to(device)
    
    def forward(self, outputs, targets):
        loss_huber = self.huber_loss(outputs, targets)
        loss_perceptual = self.perceptual_loss(outputs, targets)
        return loss_huber + loss_perceptual


class DINOv2Encoder(nn.Module):
    def __init__(self, out_features=100, model_name='dinov2_vits14_reg_lc'):
        super(DINOv2Encoder, self).__init__()
        
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

        self.adapter = nn.Sequential(
            nn.Linear(linear_features, (linear_features + out_features) // 2),      # floored average between input and output features
            nn.ELU(True),
            nn.Linear((linear_features + out_features) // 2, out_features)
        )

    def forward(self, x):
        x = self.vit(x)
        x = self.adapter(x)         # adapt the features to (out_features x 1)
        return x


class Encoder(nn.Module):

	def __init__(self, latent_dim):
		super(Encoder, self).__init__()

		self.cnn = nn.Sequential(
			nn.Conv2d(3, 128, 3, stride=2, padding=1),
			nn.BatchNorm2d(128),
			nn.ELU(True),
			nn.Conv2d(128, 256, 3, stride=2, padding=1),
			nn.BatchNorm2d(256),
			nn.ELU(True),
			nn.Conv2d(256, 512, 3, stride=2, padding=0),
			nn.BatchNorm2d(512),
			nn.ELU(True),
		)

		self.flatten = nn.Flatten(start_dim=1)

		self.fc = nn.Sequential(
			nn.Linear(373248, 128),
			nn.ELU(True),
			nn.Linear(128, latent_dim)
		)

	def forward(self, x):
		x = self.cnn(x)
		# print("enc: ", x.shape)
		x = self.flatten(x)
		x = self.fc(x)
		return x
    

class MLP(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dims, 512),
            nn.ELU(True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ELU(True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ELU(True),
            nn.BatchNorm1d(128),
            nn.Linear(128, output_dims)
        )

    def forward(self, x):
        return self.fc(x)
   

class Decoder(nn.Module):
    def __init__(self, input_dims=100, hidden_dims=512, output_channels=3, initial_size=7):
        super(Decoder, self).__init__()

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_channels = output_channels
        self.initial_size = initial_size
        
        self.fc = nn.Sequential(
			nn.Linear(input_dims, hidden_dims // 4),
			nn.BatchNorm1d(hidden_dims // 4),
			nn.ELU(True),
			nn.Linear(hidden_dims // 4, hidden_dims * initial_size * initial_size),
			nn.BatchNorm1d(hidden_dims * initial_size * initial_size),
			nn.ELU(True)
		)

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(hidden_dims, initial_size, initial_size))
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims, hidden_dims // 2, kernel_size=3, stride=2, padding=1, output_padding=1),            # 7x7 -> 14x14
            nn.BatchNorm2d(hidden_dims // 2),
			nn.ELU(True),
            nn.ConvTranspose2d(hidden_dims // 2, hidden_dims // 4, kernel_size=3, stride=2, padding=1, output_padding=1),       # 14x14 -> 28x28
            nn.BatchNorm2d(hidden_dims // 4),
			nn.ELU(True),
            nn.ConvTranspose2d(hidden_dims // 4, hidden_dims // 8, kernel_size=3, stride=2, padding=1, output_padding=1),       # 28x28 -> 56x56
            nn.BatchNorm2d(hidden_dims // 8),
			nn.ELU(True),
            nn.ConvTranspose2d(hidden_dims // 8, hidden_dims // 16, kernel_size=3, stride=2, padding=1, output_padding=1),      # 56x56 -> 112x112
            nn.BatchNorm2d(hidden_dims // 16),
			nn.ELU(True),
            nn.ConvTranspose2d(hidden_dims // 16, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1),       # 112x112 -> 224x224
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.unflatten(x)
        x = self.upsample(x)
        return x
    
    
class CustomDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.image_paths = [os.path.join(directory, fname) for fname in os.listdir(directory) if fname.endswith('.PNG')]    # lists all PNG files in the directory

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')  # Convert image to RGB
        
        if self.transform:
            image = self.transform(image)
        
        return image
    

class PairedImagesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Dataset to load paired images from given directories for aerial and ground images.
        Args:
            root_dir (str): Directory with all the images divided into 'aerial' and 'ground' subdirectories.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.aerial_dir = os.path.join(root_dir, 'aerial')
        self.ground_dir = os.path.join(root_dir, 'ground')
        self.transform = transform
        self.paired_filenames = [f for f in os.listdir(self.aerial_dir) if os.path.isfile(os.path.join(self.aerial_dir, f))]

        # Debug: Print out the number of files found
        # print(f"Number of paired files found: {len(self.paired_filenames)}")
        if len(self.paired_filenames) == 0:
            print(f"Check if the directory paths are correct and accessible: {self.aerial_dir}")
    
    def __len__(self):
        return len(self.paired_filenames)
    
    def __getitem__(self, idx):
        aerial_img_path = os.path.join(self.aerial_dir, self.paired_filenames[idx])
        ground_img_path = os.path.join(self.ground_dir, self.paired_filenames[idx].replace('drone', 'ground'))

        aerial_image = Image.open(aerial_img_path).convert('RGB')
        ground_image = Image.open(ground_img_path).convert('RGB')

        if self.transform:
            aerial_image = self.transform(aerial_image)
            ground_image = self.transform(ground_image)

        return aerial_image, ground_image