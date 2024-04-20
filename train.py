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



# Constants
image_channels = 3          # for RGB images
image_size = 224            # assuming square images
hidden_dims = 512           # hidden dimensions
n_encoded = 1028            # output size for the encoders
n_phi = 1000                 # size of phi
batch_size = 64
shuffle = True


# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

# Initialize Encoder and Decoder
encoder_A = DINOv2Encoder(out_features=n_encoded, model_name='dinov2_vitl14_reg_lc').to(device)
encoder_G = DINOv2Encoder(out_features=n_encoded, model_name='dinov2_vitl14_reg_lc').to(device)
mlp = MLP(input_dims=2*n_encoded, output_dims=n_phi).to(device)
decoder_A2G = Decoder(input_dims=n_phi+n_encoded, hidden_dims=hidden_dims, output_channels=3, initial_size=7).to(device)
decoder_G2A = Decoder(input_dims=n_phi+n_encoded, hidden_dims=hidden_dims, output_channels=3, initial_size=7).to(device)
# print(encoder_A, encoder_G, mlp, decoder_A2G, decoder_G2A)

# Optimizer and Loss Function
learning_rate = 1e-3
optimizer = optim.Adam(list(encoder_A.parameters()) + list(encoder_G.parameters()) + list(mlp.parameters()) + list(decoder_G2A.parameters()) + list(decoder_A2G.parameters()), lr=learning_rate)
criterion = nn.HuberLoss()

# Transformations
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.CenterCrop((image_size, image_size)),
    transforms.ToTensor()
])

# Define the Datasets
train_dataset = PairedImagesDataset('dataset/train', transform=transform)
val_dataset = PairedImagesDataset('dataset/val', transform=transform)

# Define the DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)


# ----- Training ----- #

def train(encoder_A, encoder_G, mlp, decoder_A2G, decoder_G2A, train_loader, val_loader, device, criterion, optimizer, epochs=10, save_path='untitled'):

    encoder_A.to(device)
    encoder_G.to(device)
    mlp.to(device)
    decoder_A2G.to(device)
    decoder_G2A.to(device)

    model_path = os.path.join('models', save_path)
    metrics_path = os.path.join('models', save_path, 'metrics')
    results_path = os.path.join('models', save_path, 'results')
    os.makedirs('models', exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(metrics_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    save_dataset_samples(train_dataloader, os.path.join(model_path, 'training_samples.png'), num_images=16, title='Training Samples')
    save_dataset_samples(val_dataloader, os.path.join(model_path, 'validation_samples.png'), num_images=16, title='Validation Samples')

      
    for epoch in range(epochs):
        encoder_A.train()
        encoder_G.train()
        mlp.train()
        decoder_A2G.train()
        decoder_G2A.train()
        
        total_loss = 0.0

        for images_A, images_G in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            images_A, images_G = images_A.to(device), images_G.to(device)
            optimizer.zero_grad()

            # Encode images A and B
            encoded_A = encoder_A(images_A)
            encoded_G = encoder_G(images_G)
            # print(f"Encoded A shape: {encoded_A.shape}")
            # print(f"Encoded G shape: {encoded_G.shape}")

            # Concatenate and process through MLP
            phi = mlp(torch.cat((encoded_A, encoded_G), dim=1))
            # print(f"Phi shape: {phi.shape}")
            # print(f"Concat Phi with Encoded G shape: {torch.cat((phi, encoded_G), dim=1).shape}")

            # Decode the MLP output into reconstructed images
            reconstructed_A = decoder_G2A(torch.cat((phi, encoded_G), dim=1))
            reconstructed_G = decoder_A2G(torch.cat((phi, encoded_A), dim=1))

            # Compute loss for both reconstructions
            loss_A = criterion(reconstructed_A, images_A)
            loss_G = criterion(reconstructed_G, images_G)
            total_loss = loss_A + loss_G

            # Backward and optimize
            total_loss.backward()
            optimizer.step()
            total_loss += total_loss.item()

        print(f'Epoch {epoch+1}/{epochs}: Training Loss = {total_loss/len(train_loader):.4f}')

        # Validate the Architecture
        validate(encoder_A, encoder_G, mlp, decoder_A2G, decoder_G2A, val_loader, criterion, epoch, results_path, device)
        
        # # Save the Models
        # torch.save(encoder_A.state_dict(), os.path.join(model_path, f'encoder_A_epoch_{epoch+1}.pth'))
        # torch.save(encoder_G.state_dict(), os.path.join(model_path, f'encoder_B_epoch_{epoch+1}.pth'))
        # torch.save(mlp.state_dict(), os.path.join(model_path, f'mlp_epoch_{epoch+1}.pth'))
        # torch.save(decoder_A2G.state_dict(), os.path.join(model_path, f'decoder_A2G_epoch_{epoch+1}.pth'))
        # torch.save(decoder_G2A.state_dict(), os.path.join(model_path, f'decoder_G2A_epoch_{epoch+1}.pth'))


def validate(encoder_A, encoder_B, mlp, decoder_A2G, decoder_G2A, loader, criterion, epoch, results_path, device):
    encoder_A.eval()
    encoder_B.eval()
    mlp.eval()
    decoder_A2G.eval()
    decoder_G2A.eval()
    total_val_loss = 0

    with torch.no_grad():
        for images_A, images_G in loader:
            images_A, images_G = images_A.to(device), images_G.to(device)

            encoded_A = encoder_A(images_A)
            encoded_G = encoder_G(images_G)
            phi = mlp(torch.cat((encoded_A, encoded_G), dim=1))
            reconstructed_A = decoder_G2A(torch.cat((phi, encoded_G), dim=1))
            reconstructed_G = decoder_A2G(torch.cat((phi, encoded_A), dim=1))

            loss_A = criterion(reconstructed_A, images_A)
            loss_G = criterion(reconstructed_G, images_G)
            total_loss = loss_A + loss_G

            total_val_loss += total_loss.item()

    # visualize_reconstruction(images_A, reconstructed_A, epoch, save_path=results_path)
    visualize_reconstruction(images_G, reconstructed_G, epoch, save_path=results_path)

    print(f'Validation Loss: {total_val_loss / len(loader):.4f}')


train(encoder_A, encoder_G, mlp, decoder_A2G, decoder_G2A, train_dataloader, val_dataloader, device, criterion, optimizer, epochs=10, save_path='lDINO + Huber + n_phi = 1000')