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
from utils import *


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

	def __init__(self, latent_dim=10):
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
     

class SparseEncoder(nn.Module):

	def __init__(self, latent_dim=50):
		super(SparseEncoder, self).__init__()

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
			nn.Linear(1024*7*7, 1000),
			nn.ELU(True),
			nn.Linear(1000, latent_dim)
		)

	def forward(self, x):
		x = self.cnn(x)
		# print("Encoder CNN Output Size: ", x.shape)
		x = self.flatten(x)
		x = self.fc(x)
          
		return x
     

class SparseEncoder(nn.Module):
    def __init__(self, latent_dim=1024, top_k=50):
        super(SparseEncoder, self).__init__()

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
            nn.ELU(True)
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.fc = nn.Sequential(
            nn.Linear(1024 * 7 * 7, 5000),
            nn.ELU(True),
            nn.Linear(5000, latent_dim)
        )

        self.top_k = top_k

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.fc(x)

        # Apply the top-k operation
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)  # Flatten to [batch_size, latent_dim]
        top_values, _ = torch.topk(x_flat, self.top_k, dim=1)
        threshold = top_values[:, -1].unsqueeze(-1)
        binary_flat = (x_flat >= threshold).float()

        x = binary_flat*x_flat.detach() + x_flat - x_flat.detach()

        # print("Sparse Encoder Output Size: ", x.shape)
        # print("Top Values: ", top_values[0, :10])
        # print("binary_flat shape: ", binary_flat.shape)
        # print("x_flat shape: ", x_flat.shape)
        # print("Threshold shape: ", threshold.shape)

        return x
   

class Decoder(nn.Module):
    def __init__(self, input_dims=100, hidden_dims=1024, output_channels=3, initial_size=7, image_size=224):
        super(Decoder, self).__init__()

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_channels = output_channels
        self.initial_size = initial_size
        self.image_size = image_size

        self.fc = nn.Sequential(
            nn.Linear(input_dims, input_dims*2),
            nn.ELU(True),
            nn.Linear(input_dims*2, hidden_dims * initial_size * initial_size)
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
            nn.ConvTranspose2d(hidden_dims // 16, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64x112x112 -> 4x224x224
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.unflatten(x)
        x = self.upsample(x)
        return x
    

class Attention(nn.Module):
    def __init__(self, input_dims=100, hidden_dims=512, output_channels=1, initial_size=7, image_size=224, attention_size=224//4):
        super(Attention, self).__init__()

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_channels = output_channels
        self.initial_size = initial_size
        self.image_size = image_size
        self.attention_size = attention_size
        self.bias = 10

        self.fc = nn.Sequential(
            nn.Linear(input_dims, attention_size * attention_size // 2),
            nn.ELU(True),
            nn.Linear(attention_size * attention_size // 2, attention_size * attention_size)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(output_channels, attention_size, attention_size))

    def forward(self, x):
        x = self.fc(x)
        x = self.unflatten(x)

        # Interpolate to the desired image size
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=True)

        # Reshape Attention Map and apply Softmax
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)                                                 # [batch_size, image_size^2]
        x_flat = torch.softmax(x_flat, dim=1) * self.image_size * self.image_size

        # Reshape back to image_size x image_size
        x = x_flat.view(batch_size, 1, self.image_size, self.image_size)

        return x
    

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
    def __init__(self, n_phi, n_encoded, hidden_dims, image_size, output_channels=3, pretrained=True, debug=False):
        super(CrossView, self).__init__()

        self.encoder_A = Encoder(latent_dim=n_encoded)
        self.encoder_G = Encoder(latent_dim=n_encoded)
        self.combiner = MLP(input_dims=2*n_encoded, output_dims=n_phi)
        self.mlp_A2G = MLP(input_dims=n_phi+n_encoded, output_dims=n_encoded)
        self.mlp_G2A = MLP(input_dims=n_phi+n_encoded, output_dims=n_encoded)
        self.decoder_A2G = Decoder(input_dims=n_encoded, hidden_dims=hidden_dims, output_channels=output_channels, initial_size=7)
        self.decoder_G2A = Decoder(input_dims=n_encoded, hidden_dims=hidden_dims, output_channels=output_channels, initial_size=7)
        
        self.image_size = image_size
        self.debug = debug

        if pretrained:
            encoder_A_path = os.path.join('pretrained_models', 'encoder_A.pth')
            encoder_G_path = os.path.join('pretrained_models', 'encoder_G.pth')
            decoder_A_path = os.path.join('pretrained_models', 'decoder_A.pth')
            decoder_G_path = os.path.join('pretrained_models', 'decoder_G.pth')
            self.encoder_A.load_state_dict(torch.load(encoder_A_path))
            self.encoder_G.load_state_dict(torch.load(encoder_G_path))
            self.decoder_A2G.load_state_dict(torch.load(decoder_G_path))
            self.decoder_G2A.load_state_dict(torch.load(decoder_A_path))
            freeze(self.encoder_A)
            freeze(self.encoder_G)
            freeze(self.decoder_A2G)
            freeze(self.decoder_G2A)

    
    def forward(self, images_A, images_G):

        # Encode images A and G
        encoded_A = self.encoder_A(images_A)
        encoded_G = self.encoder_G(images_G)

        # Concatenate and process through MLP
        phi = self.combiner(torch.cat((encoded_A, encoded_G), dim=-1))

        # Concatenate the encoded attended images with phi
        encoded_A_phi = self.mlp_A2G(torch.cat((phi, encoded_A), dim=1))   # -> encoded_G
        encoded_G_phi = self.mlp_G2A(torch.cat((phi, encoded_G), dim=1))

        # Decode the MLP output into reconstructed images
        reconstructed_A = self.decoder_G2A(encoded_G_phi)
        reconstructed_G = self.decoder_A2G(encoded_A_phi)

        # Print shapes for debugging
        if self.debug:
            print(f"Encoded A shape: {encoded_A.shape}, Encoded G shape: {encoded_G.shape}, "
                  f"Phi shape: {phi.shape}, Concat Phi with Encoded G shape: {torch.cat((phi, encoded_G), dim=1).shape}")
            
        return reconstructed_A, reconstructed_G, encoded_A, encoded_G, phi, encoded_A_phi, encoded_G_phi



## Top k Magic - Straight Trhough Gradient

class Attention(nn.Module):
    def __init__(self, input_dims=100, hidden_dims=512, output_channels=1, initial_size=7, image_size=224, attention_size=224//4):
        super(Attention, self).__init__()

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_channels = output_channels
        self.initial_size = initial_size
        self.image_size = image_size
        self.attention_size = attention_size
        self.bias = 10

        self.fc = nn.Sequential(
            nn.Linear(input_dims, input_dims),
            nn.ELU(True),
            nn.Linear(input_dims, hidden_dims * initial_size * initial_size)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(hidden_dims, initial_size, initial_size))

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims, hidden_dims // 2, kernel_size=3, stride=2, padding=1, output_padding=1),  # 512x7x7 -> 256x14x14
            nn.BatchNorm2d(hidden_dims // 2),
            nn.ELU(True),
            nn.ConvTranspose2d(hidden_dims // 2, hidden_dims // 4, kernel_size=3, stride=2, padding=1, output_padding=1),  # 256x14x14 -> 128x28x28
            nn.BatchNorm2d(hidden_dims // 4),
            nn.ELU(True),
            nn.ConvTranspose2d(hidden_dims // 4, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 128x28x28 -> 1x56x56
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.unflatten(x)
        x = self.upsample(x)
        # attention_map = x + self.bias
        attention_map = x

        # Reshape attention map for softmax
        batch_size = attention_map.size(0)
        attention_map_flat = attention_map.view(batch_size, -1)  # [batch_size, image_size^2]

        # Apply softmax
        attention_map_flat = torch.softmax(attention_map_flat, dim=1) * self.attention_size * self.attention_size

        # print("Attention Map Mean: ", attention_map.mean())
        # print("Attention Map Size: ", attention_map_flat.shape)
        # print("Attention Map: ", attention_map_flat[0, :10])

        # Reshape back to image_size x image_size
        attention_map = attention_map_flat.view(batch_size, 1, self.attention_size, self.attention_size)

        # Interpolate to the desired image size
        attention_map = F.interpolate(attention_map, size=(self.image_size, self.image_size), mode='bilinear', align_corners=True)

        # # Compute top 10% threshold
        # top_k = int(0.1 * attention_map.numel() / batch_size)
        # top_values, _ = torch.topk(attention_map.view(batch_size, -1), top_k, dim=1)
        # threshold = top_values[:, -1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # # Create binary map
        # binary_map = (attention_map >= threshold).float()

        # # Create the final attention map
        # attention_map = binary_map + attention_map - attention_map.detach()

        return attention_map