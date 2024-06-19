import torch
import os
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import argparse
import matplotlib.pyplot as plt
from model import CrossView, Encoder
from dataset import CombinedPairedImagesDataset, sample_paired_images
from utils import visualize_reconstruction, ssim_loss, PerceptualLoss

def format_phi(phi_tensor):
    phi_list = phi_tensor.cpu().detach().numpy().flatten().tolist()
    return ', '.join(map(str, phi_list))

def interactive_test(model, test_loader, device):
    model.to(device)
    model.eval()

    # Choose an image index interactively
    image_index = int(input("Enter the image index to test (0 to {}): ".format(len(test_loader.dataset) - 1)))
    
    # Check if the image index is valid
    if image_index < 0 or image_index >= len(test_loader.dataset):
        print("Invalid image index. Please run the script again and provide a valid index.")
        return
    
    # Load the specified image
    with torch.no_grad():
        for idx, (images_A, images_G) in enumerate(test_loader):
            if idx == image_index:
                images_A, images_G = images_A.to(device), images_G.to(device)
                break
    
    # Forward Pass
    reconstructed_A, reconstructed_G, encoded_A, encoded_G, phi, encoded_A_phi, encoded_G_phi = model(images_A, images_G)
    
    # Visualize original and reconstructed images
    visualize_reconstruction(images_A, reconstructed_A, 0, save_path=None, num_images=1)
    visualize_reconstruction(images_G, reconstructed_G, 0, save_path=None, num_images=1)
    
    print("Phi values: ", format_phi(phi))
    
    # Interactive loop to modify phi and see results
    while True:
        action = input("\nPlease, enter an action code:\n1. Apply a Rotation to Aerial Image\n2. Set phi manually\n3. Exit\n\n").strip().lower()
        
        if action == '1':
            # Apply some transformation and see the new phi
            transform = transforms.RandomRotation(degrees=45)
            transformed_image = transform(images_A.cpu()).to(device)
            _, _, _, _, transformed_phi, _, _ = model(transformed_image, images_G)
            print("Transformed Phi values: ", format_phi(transformed_phi))
            visualize_reconstruction(transformed_image, reconstructed_A, 0, save_path=None, num_images=1)
        
        elif action == '2':
            new_phi = input("Enter new phi values (comma-separated): ")
            try:
                new_phi = torch.tensor([[float(x) for x in new_phi.split(',')]], device=device)
                if new_phi.shape != (1, n_phi):
                    raise ValueError
            except:
                print("Invalid format for phi. Please try again.")
                continue

            encoded_A_phi_new = model.mlp_A2G(torch.cat((new_phi, encoded_A), dim=1))
            encoded_G_phi_new = model.mlp_G2A(torch.cat((new_phi, encoded_G), dim=1))
            new_reconstructed_A = model.decoder_G2A(encoded_G_phi_new)
            new_reconstructed_G = model.decoder_A2G(encoded_A_phi_new)
            visualize_reconstruction(images_A, new_reconstructed_A, 0, save_path=None, num_images=1)
            visualize_reconstruction(images_G, new_reconstructed_G, 0, save_path=None, num_images=1)
        
        elif action == '3':
            break

        else:
            print("Invalid action. Please enter one of the given options.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a model interactively.')
    parser.add_argument('--save_path', '-p', type=str, default='huber_loss', help='Path to load the model and results')
    args = parser.parse_args()

    # Constants
    image_channels = 3
    image_size = 224
    aerial_scaling = 3
    hidden_dims = 512
    n_encoded = 1024
    n_phi = 10
    batch_size = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CrossView(n_phi, n_encoded, hidden_dims, image_size, output_channels=image_channels, pretrained=True).to(device)
    model_path = os.path.join('models', args.save_path, f'best_model_epoch_13.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded from ", model_path)

    transform_cutouts = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop((image_size, image_size)),
        transforms.ToTensor()
    ])

    transform_panos = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=image_size, scale=(0.8, 1.0)),
        transforms.ToTensor()
    ])

    transform_aerial = transforms.Compose([
        transforms.Resize((int(image_size * aerial_scaling), int(image_size * aerial_scaling))),
        transforms.RandomRotation(degrees=360),
        transforms.CenterCrop((image_size, image_size)),
        transforms.ToTensor()
    ])

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    test_filenames_panos, _ = sample_paired_images('/home/lrusso/cvusa', sample_percentage=0.2, split_ratio=0.1, groundtype='panos')
    test_filenames_cutouts, _ = sample_paired_images('/home/lrusso/cvusa', sample_percentage=0.2, split_ratio=0.1, groundtype='cutouts')

    test_dataset_combined = CombinedPairedImagesDataset(test_filenames_panos, test_filenames_cutouts, transform_aerial=transform_aerial, transform_panos=transform_panos, transform_cutouts=transform_cutouts)
    test_dataloader_combined = DataLoader(test_dataset_combined, batch_size=batch_size, shuffle=False, num_workers=8)

    interactive_test(model, test_dataloader_combined, device)
