import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


model_save_path = 'models/ldm'

class Encoder(nn.Module):
  def __init__(self, latent_dim):
    super(Encoder, self).__init__()
    self.encoder = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
      nn.ReLU(),
      nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(128*32*32, latent_dim)
    )
  
  def forward(self, x):
    return self.encoder(x)
  
class Decoder(nn.Module):
  def __init__(self, latent_dim):
    super(Decoder, self).__init__()
    self.decoder = nn.Sequential(
      nn.Linear(latent_dim, 128*32*32),
      nn.ReLU(),
      nn.Unflatten(1, (128, 32, 32)),
      nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
      nn.ReLU(),
      nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
      nn.Sigmoid()
    )
  
  def forward(self, z):
    return self.decoder(z)


class DiffusionModel(nn.Module):
  def __init__(self, latent_dim):
    super(DiffusionModel, self).__init__()
    self.denoise_net = nn.Sequential(
      nn.Linear(latent_dim, 512),
      nn.ReLU(),
      nn.Linear(512, latent_dim)
    )
  
  def forward(self, z, t):
    t = t.view(-1, 1).expand_as(z)
    noise = torch.randn_like(z)
    z_noisy = z + t * noise
    denoised = self.denoise_net(z_noisy)
    return denoised

device = 'cuda' if torch.cuda.is_available() else 'cpu'
T = 10
num_epochs = 3

latent_dim = 256
encoder = Encoder(latent_dim)
decoder = Decoder(latent_dim)
diffusion_model = DiffusionModel(latent_dim)

autoencoder_params = list(encoder.parameters()) + list(decoder.parameters())
autoencoder_optimizer = optim.Adam(autoencoder_params, lr=1e-4)
diffusion_optimizer = optim.Adam(diffusion_model.parameters(), lr=1e-4)


# Define transformations for the training set
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize images
])

# Load the dataset (e.g., CIFAR-10)
train_dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)

# Create a DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

for epoch in range(num_epochs):
  for images, _labels in tqdm(train_loader):
    images = images.to(device)

    # Encoder
    latents = encoder(images)
    recon_images = decoder(latents)
    recon_loss = nn.MSELoss()(recon_images, images)


    # Diffusion forward
    t = torch.randint(0, T, (images.size(0),), device=device).float() / T
    noise_estimate = diffusion_model(latents, t)
    noise_loss = nn.MSELoss()(noise_estimate, latents)

    # Backpropagation for autoencoder
    autoencoder_optimizer.zero_grad()
    diffusion_optimizer.zero_grad()
    # recon_loss.backward()
    (recon_loss + noise_loss).backward()
    # noise_loss.backward(retain_graph=True)
    diffusion_optimizer.step()
    autoencoder_optimizer.step()

  torch.save({
    'encoder_state_dict': encoder.state_dict(),
    'decoder_state_dict': decoder.state_dict(),
    'diffusion_model_state_dict': diffusion_model.state_dict(),
    'autoencoder_optimizer_state_dict': autoencoder_optimizer.state_dict(),
    'diffusion_optimizer_state_dict': diffusion_optimizer.state_dict(),
    'epoch': epoch,
  }, os.path.join(model_save_path, f'ldm_checkpoint_epoch_{epoch}.pth'))

  print(f'Epoch {epoch} model saved!')

model_save_path = 'models/ldm/'
checkpoint_path = os.path.join(model_save_path, 'ldm_checkpoint_epoch_2.pth')
checkpoint = torch.load(checkpoint_path)

encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])
diffusion_model.load_state_dict(checkpoint['diffusion_model_state_dict'])
autoencoder_optimizer.load_state_dict(checkpoint['autoencoder_optimizer_state_dict'])
diffusion_optimizer.load_state_dict(checkpoint['diffusion_optimizer_state_dict'])

def generate_image(encoder, decoder, diffusion_model):
  encoder.to(device)
  decoder.to(device)
  diffusion_model.to(device)
  
  encoder.eval()
  decoder.eval()
  diffusion_model.eval()

  with torch.no_grad():
    z = torch.randn(1, latent_dim, device=device)
    for t in tqdm(reversed(range(T))):
      t = torch.tensor([t], device=z.device, dtype=z.dtype)
      z = diffusion_model(z, t / T)
    generated_image = decoder(z)
  
  return generated_image

import matplotlib.pyplot as plt
image = generate_image(encoder, decoder, diffusion_model)
plt.imshow(image.squeeze().cpu().numpy().transpose(1, 2, 0))
plt.show()