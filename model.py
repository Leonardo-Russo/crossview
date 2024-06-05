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
        self.mlp = MLP(input_dims=2*n_encoded, output_dims=n_phi)
        self.attention_A2G = Attention(input_dims=n_phi, hidden_dims=hidden_dims, output_channels=1, initial_size=7)
        self.attention_G2A = Attention(input_dims=n_phi, hidden_dims=hidden_dims, output_channels=1, initial_size=7)
        self.decoder_A2G = Decoder(input_dims=n_phi+n_encoded, hidden_dims=hidden_dims, output_channels=output_channels, initial_size=7)
        self.decoder_G2A = Decoder(input_dims=n_phi+n_encoded, hidden_dims=hidden_dims, output_channels=output_channels, initial_size=7)
        
        self.image_size = image_size
        self.debug = debug

        if pretrained:
            encoder_A_path = os.path.join('pretrained_models', 'encoder_A.pth')
            encoder_G_path = os.path.join('pretrained_models', 'encoder_G.pth')
            decoder_A_path = os.path.join('pretrained_models', 'decoder_A.pth')
            decoder_G_path = os.path.join('pretrained_models', 'decoder_G.pth')
            self.encoder_A.load_state_dict(torch.load(encoder_A_path))
            self.encoder_G.load_state_dict(torch.load(encoder_G_path))
            freeze(self.encoder_A)
            freeze(self.encoder_G)

            pretrained_decoder_A = Decoder(input_dims=n_encoded, hidden_dims=hidden_dims, output_channels=output_channels, initial_size=7)
            pretrained_decoder_G = Decoder(input_dims=n_encoded, hidden_dims=hidden_dims, output_channels=output_channels, initial_size=7)
            pretrained_decoder_A.load_state_dict(torch.load(decoder_A_path))
            pretrained_decoder_G.load_state_dict(torch.load(decoder_G_path))
            self.decoder_A2G = transfer_decoder(self.decoder_A2G, pretrained_decoder_G, n_encoded, hidden_dims, initial_size=7)
            self.decoder_G2A = transfer_decoder(self.decoder_G2A, pretrained_decoder_A, n_encoded, hidden_dims, initial_size=7)

    
    def forward(self, images_A, images_G):

        skip_attention = True

        # Encode images A and G
        encoded_A = self.encoder_A(images_A)
        encoded_G = self.encoder_G(images_G)

        # Concatenate and process through MLP
        phi = self.mlp(torch.cat((encoded_A, encoded_G), dim=-1))

        # Compute Attention Maps
        attention_A = self.attention_A2G(phi)
        attention_G = self.attention_G2A(phi)

        # Resize Attention Maps into Image Size Map
        attention_A = F.interpolate(attention_A, size=(self.image_size, self.image_size), mode='bilinear', align_corners=True)
        attention_G = F.interpolate(attention_G, size=(self.image_size, self.image_size), mode='bilinear', align_corners=True)

        if skip_attention:
            attention_A = torch.ones_like(attention_A)
            attention_G = torch.ones_like(attention_G)
        
        # Apply Attention Maps to Images
        attended_A = images_A * attention_A.expand_as(images_A)     # apply on all channels
        attended_G = images_G * attention_G.expand_as(images_G)     # apply on all channels

        # Encode the attended images
        encoded_attended_A = self.encoder_A(attended_A)
        encoded_attended_G = self.encoder_G(attended_G)

        # Decode the MLP output into reconstructed images
        reconstructed_A = self.decoder_G2A(torch.cat((phi, encoded_attended_G), dim=1))
        reconstructed_G = self.decoder_A2G(torch.cat((phi, encoded_attended_A), dim=1))

        # Print shapes for debugging
        if self.debug:
            print(f"Encoded A shape: {encoded_A.shape}, Encoded G shape: {encoded_G.shape}, "
                  f"Phi shape: {phi.shape}, Concat Phi with Encoded G shape: {torch.cat((phi, encoded_G), dim=1).shape}")
            
        return reconstructed_A, reconstructed_G, attention_A, attention_G, attended_A, attended_G
