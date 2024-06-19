import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import math
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import numpy as np
import random
from utils import *


class ViTEncoder(nn.Module):
    def __init__(self, out_features=100, model_name='dinov2_vits14_reg_lc'):
        super(ViTEncoder, self).__init__()
        
        self.vit = torch.hub.load('facebookresearch/dinov2', model_name)
        
        for param in self.vit.parameters():
            param.requires_grad = False  # freeze all parameters of the Vision Transformer

        linear_features = self.vit.linear_head.in_features
        self.vit.linear_head = nn.Identity()

        self.reducer = nn.Sequential(
            nn.Linear(linear_features, (linear_features + out_features) // 2),  # floored average between input and output features
            nn.ELU(True),
            nn.Linear((linear_features + out_features) // 2, out_features)
        )

    def forward(self, x):
        x = self.vit(x)
        x = self.reducer(x)
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
            nn.ELU(True)  # 3x224x224 -> 64x112x112 -> 128x56x56 -> 256x28x28 -> 512x14x14 -> 1024x7x7
        )

        self.flatten = nn.Flatten(start_dim=1)
        self.fc_mu = nn.Linear(1024 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(1024 * 7 * 7, latent_dim)

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, input_dims=100, hidden_dims=1024, output_channels=3, initial_size=7, image_size=224):
        super(Decoder, self).__init__()

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_channels = output_channels
        self.initial_size = initial_size
        self.image_size = image_size

        self.fc = nn.Sequential(
            nn.Linear(input_dims, input_dims * 2),
            nn.ELU(True),
            nn.Linear(input_dims * 2, hidden_dims * initial_size * initial_size)
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
    def __init__(self, input_dims=100, hidden_dims=512, output_channels=1, initial_size=7, image_size=224, attention_size=224 // 4):
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
        x_flat = x.view(batch_size, -1)  # [batch_size, image_size^2]
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


class CrossViewVAE(nn.Module):
    def __init__(self, n_phi, n_encoded, hidden_dims, image_size, output_channels=3, pretrained=True, debug=False):
        super(CrossViewVAE, self).__init__()

        self.encoder_A = Encoder(latent_dim=n_encoded)
        self.encoder_G = Encoder(latent_dim=n_encoded)
        self.combiner = MLP(input_dims=2 * n_encoded, output_dims=n_phi)
        self.mlp_A2G = MLP(input_dims=n_phi + n_encoded, output_dims=n_encoded)
        self.mlp_G2A = MLP(input_dims=n_phi + n_encoded, output_dims=n_encoded)
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

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, images_A, images_G):
        # Encode images A and G
        mu_A, logvar_A = self.encoder_A(images_A)
        mu_G, logvar_G = self.encoder_G(images_G)
        encoded_A = self.reparameterize(mu_A, logvar_A)
        encoded_G = self.reparameterize(mu_G, logvar_G)

        # Concatenate and process through MLP
        phi = self.combiner(torch.cat((encoded_A, encoded_G), dim=-1))

        # Concatenate the encoded attended images with phi
        encoded_A_phi = self.mlp_A2G(torch.cat((phi, encoded_A), dim=1))  # -> encoded_G
        encoded_G_phi = self.mlp_G2A(torch.cat((phi, encoded_G), dim=1))

        # Decode the MLP output into reconstructed images
        reconstructed_A = self.decoder_G2A(encoded_G_phi)
        reconstructed_G = self.decoder_A2G(encoded_A_phi)

        # Print shapes for debugging
        if self.debug:
            print(f"Encoded A shape: {encoded_A.shape}, Encoded G shape: {encoded_G.shape}, "
                  f"Phi shape: {phi.shape}, Concat Phi with Encoded G shape: {torch.cat((phi, encoded_G), dim=1).shape}")

        return reconstructed_A, reconstructed_G, mu_A, logvar_A, mu_G, logvar_G, encoded_A_phi, encoded_G_phi