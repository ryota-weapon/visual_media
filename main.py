import torch
import os

# Define the paths and models
model_save_path = 'models/ldm/'
checkpoint_path = os.path.join(model_save_path, 'ldm_checkpoint_epoch_2.pth')

# Initialize the models and optimizers
encoder = Encoder(latent_dim)
decoder = Decoder(latent_dim)
diffusion_model = DiffusionModel(latent_dim)

autoencoder_optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))
diffusion_optimizer = torch.optim.Adam(diffusion_model.parameters())

# Load the saved checkpoint
checkpoint = torch.load(checkpoint_path)

# Load the state dictionaries into the models and optimizers
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])
diffusion_model.load_state_dict(checkpoint['diffusion_model_state_dict'])
autoencoder_optimizer.load_state_dict(checkpoint['autoencoder_optimizer_state_dict'])
diffusion_optimizer.load_state_dict(checkpoint['diffusion_optimizer_state_dict'])

# Optionally, set the models to evaluation mode if you are using them for inference
encoder.eval()
decoder.eval()
diffusion_model.eval()

# If resuming training, you might want to restore the epoch number
epoch = checkpoint['epoch']

print(f'Models and optimizers loaded from checkpoint at epoch {epoch}.')
