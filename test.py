import torch
import os
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from model import CrossView, Encoder
from dataset import CombinedPairedImagesDataset, sample_paired_images
from utils import visualize_reconstruction, ssim_loss, PerceptualLoss
import glob

def format_phi(phi_tensor):
    phi_list = phi_tensor.cpu().detach().numpy().flatten().tolist()
    return ', '.join(map(str, phi_list))

def plot_phi_values(df, save_path):
    angle = df['Angle']
    for i in range(1, df.shape[1]):
        plt.figure()
        plt.plot(angle, df.iloc[:, i], label=f'phi_{i-1}')
        plt.xlabel('Rotation Angle (degrees)')
        plt.ylabel(f'phi_{i-1}')
        plt.title(f'Behavior of phi_{i-1} vs. Rotation Angle')
        plt.legend()
        plt.grid(True)
        plot_save_path = os.path.join(save_path, f'phi_{i-1}_vs_angle.png')
        plt.savefig(plot_save_path)
        plt.close()

def interactive_test(model, test_loader, device, save_path):
    model.to(device)
    model.eval()

    save_path = os.path.join('tests', save_path)

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
    
    print("Phi values:\n", format_phi(phi))
    
    # Interactive loop to modify phi and see results
    while True:
        action = input("\nPlease, enter an action code:\n1. Apply a Rotation to Aerial Image\n2. Set phi manually\n3. Apply phi from CSV to another image\n4. Exit\n\n").strip().lower()
        
        if action == '1':
            # Get the final angle of rotation from the user
            # final_angle = int(input("Enter the final angle of rotation (in degrees): "))
            final_angle = 360
            angle_step = 5
            phi_values = []
            output_dir = os.path.join(save_path, 'rotated_images')
            os.makedirs(output_dir, exist_ok=True)

            for angle in range(0, final_angle + 1, angle_step):
                transform = transforms.functional.rotate
                transformed_image = transform(images_A.cpu(), angle).to(device)
                transformed_reconstructed_A, _, _, _, transformed_phi, _, _ = model(transformed_image, images_G)
                phi_values.append([angle] + transformed_phi.cpu().detach().numpy().flatten().tolist())
                
                # Save the image
                save_path_image = os.path.join(output_dir, f'image_rotated_{angle}.png')
                visualize_reconstruction(transformed_image, transformed_reconstructed_A, 0, save_path=save_path_image, num_images=1)
                
            # Save phi values to a CSV file
            columns = ['Angle'] + [f'phi_{i}' for i in range(len(phi_values[0]) - 1)]
            df = pd.DataFrame(phi_values, columns=columns)
            df.to_csv(os.path.join(save_path, 'phi_values.csv'), index=False)
            print(f"Phi values saved to 'phi_values.csv' and images saved to '{output_dir}'.")

            # Plot the behavior of each component of phi as a function of the angle
            plots_dir = os.path.join(save_path, 'phi_plots')
            os.makedirs(plots_dir, exist_ok=True)
            plot_phi_values(df, plots_dir)
            print(f"Phi plots saved to '{plots_dir}'.")

        elif action == '2':
            new_phi = input("Enter new phi values (comma-separated): ")
            try:
                new_phi = torch.tensor([[float(x) for x in new_phi.split(',')]], device=device)
                if new_phi.shape != (1, n_phi):
                    raise ValueError("Phi tensor has incorrect shape.")
            except Exception as e:
                print(f"Invalid format for phi: {e}. Please try again.")
                continue

            encoded_A_phi_new = model.mlp_A2G(torch.cat((new_phi, encoded_A), dim=1))
            encoded_G_phi_new = model.mlp_G2A(torch.cat((new_phi, encoded_G), dim=1))
            new_reconstructed_A = model.decoder_G2A(encoded_G_phi_new)
            new_reconstructed_G = model.decoder_A2G(encoded_A_phi_new)
            visualize_reconstruction(images_A, new_reconstructed_A, 0, save_path=None, num_images=1)
            visualize_reconstruction(images_G, new_reconstructed_G, 0, save_path=None, num_images=1)

        elif action == '3':
            csv_path = input("Enter the path to the CSV file containing phi values: ")
            phi_df = pd.read_csv(csv_path)
            phi_values = phi_df.iloc[:, 1:].values

            image_index = int(input("Enter the image index to apply phi values (0 to {}): ".format(len(test_loader.dataset) - 1)))
            
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
            
            output_dir = os.path.join(save_path, 'phi_applied_images')
            os.makedirs(output_dir, exist_ok=True)

            for i, phi in enumerate(phi_values):
                phi_array = np.array(phi, dtype=np.float32)
                phi_tensor = torch.tensor([phi_array], device=device)
                encoded_A_phi_new = model.mlp_A2G(torch.cat((phi_tensor, encoded_A), dim=1))
                encoded_G_phi_new = model.mlp_G2A(torch.cat((phi_tensor, encoded_G), dim=1))
                new_reconstructed_A = model.decoder_G2A(encoded_G_phi_new)
                new_reconstructed_G = model.decoder_A2G(encoded_A_phi_new)
                
                # Save the images
                save_path_aerial = os.path.join(output_dir, f'reconstructed_aerial_{i}.png')
                save_path_ground = os.path.join(output_dir, f'reconstructed_ground_{i}.png')
                visualize_reconstruction(images_A, new_reconstructed_A, 0, save_path=save_path_aerial, num_images=1)
                visualize_reconstruction(images_G, new_reconstructed_G, 0, save_path=save_path_ground, num_images=1)
            
            print(f"Reconstructed images saved to '{output_dir}'.")

        elif action == '4':
            break

        else:
            print("Invalid action. Please enter one of the given options.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a model interactively.')
    parser.add_argument('--model_path', '-m', type=str, default='huber_loss', help='Path to load the model')
    parser.add_argument('--save_path', '-p', type=str, default='results', help='Path to save the results')
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

    # Load the Pretrained Model
    model_files = glob.glob(os.path.join('models', args.model_path, 'best_model_epoch_*.pth'))
    if not model_files:
        print("No model file found.")
        exit(1)
    model_files.sort()
    model_path = model_files[-1]
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from: {model_path}")

    transform_cutouts = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop((image_size, image_size)),
        transforms.ToTensor()
    ])

    transform_panos = transforms.Compose([
        transforms.RandomResizedCrop(size=image_size, scale=(0.8, 1.0)),
        transforms.ToTensor()
    ])

    transform_aerial = transforms.Compose([
        transforms.Resize((int(image_size * aerial_scaling), int(image_size * aerial_scaling))),
        transforms.CenterCrop((image_size, image_size)),
        transforms.ToTensor()
    ])

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    test_filenames_panos, _ = sample_paired_images('/home/lrusso/cvusa', sample_percentage=0.2, split_ratio=0.1, groundtype='panos')
    test_filenames_cutouts, _ = sample_paired_images('/home/lrusso/cvusa', sample_percentage=0.2, split_ratio=0.1, groundtype='cutouts')

    test_dataset_combined = CombinedPairedImagesDataset(test_filenames_panos, test_filenames_cutouts, transform_aerial=transform_aerial, transform_panos=transform_panos, transform_cutouts=transform_cutouts)
    test_dataloader_combined = DataLoader(test_dataset_combined, batch_size=batch_size, shuffle=False, num_workers=8)

    interactive_test(model, test_dataloader_combined, device, args.save_path)
