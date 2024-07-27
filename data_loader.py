import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import animation, rc

matplotlib.use('TkAgg')
matplotlib.interactive(False)

BATCH_SIZE = 100

trainval_data = MNIST("./data", 
                   train=True, 
                   download=True, 
                   transform=transforms.ToTensor())

train_size = int(len(trainval_data) * 0.8)
val_size = int(len(trainval_data) * 0.2)
train_data, val_data = torch.utils.data.random_split(trainval_data, [train_size, val_size])

train_loader = DataLoader(dataset=train_data,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=0)

val_loader = DataLoader(dataset=val_data,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=0)


class Encoder(nn.Module):
  def __init__(self, z_dim):
    super().__init__()
    self.lr = nn.Linear(28*28, 300)
    self.lr2 = nn.Linear(300, 100)
    self.lr_ave = nn.Linear(100, z_dim)
    self.lr_dev = nn.Linear(100, z_dim)
    self.relu = nn.ReLU()
  
  def forward(self, x):
    x = self.lr(x)
    x = self.relu(x)
    x = self.lr2(x)
    x = self.relu(x)
    ave = self.lr_ave(x)
    log_dev = self.lr_dev(x)

    ep = torch.randn(ave.size())
    z = ave + torch.exp(log_dev/2)*ep

    return z, ave, log_dev

class Decoder(nn.Module):
  def __init__(self, z_dim):
    super().__init__()
    self.lr = nn.Linear(z_dim, 100)
    self.lr2 = nn.Linear(100, 300)
    self.lr3 = nn.Linear(300, 28*28)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
  
  def forward(self, z):
    z = self.lr(z)
    z = self.relu(z)
    z = self.lr2(z)
    z = self.relu(z)
    z = self.lr3(z)
    z = self.sigmoid(z)

    return z

class VAE(nn.Module):
  def __init__(self, z_dim):
    super().__init__()
    self.encoder = Encoder(z_dim)
    self.decoder = Decoder(z_dim)
  
  def forward(self, x):
    z, ave, log_dev = self.encoder(x)
    x = self.decoder(z)
    return x, z, ave, log_dev
  

def criterion(recon_x, x, ave, log_dev):
  # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 28*28), reduction='sum')
  BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
  KLD = -0.5 * torch.sum(1 + log_dev - ave.pow(2) - log_dev.exp())
  return BCE + KLD



z_dim = 2
num_epochs = 1

#milestone: decrease the learning rate at the timestep in milestones
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE(z_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)

history  = {'train_loss': [], 'val_loss': [], "ave": [], "log_dev": [], "z": [], "labels": []}

for epoch in range(num_epochs):
  model.train()

  for i, (x, labels) in enumerate(train_loader):
    input = x.to(device).view(-1, 28*28).to(torch.float32)
    output, z, ave, log_dev = model(input)

    history['ave'].append(ave)
    history['log_dev'].append(log_dev)
    history['z'].append(z)
    history['labels'].append(labels)
    
    loss = criterion(output, input, ave, log_dev)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i+1) % 50 == 0:
      print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss:.4f}")
    history['train_loss'].append(loss)

  model.eval()
  with torch.no_grad():
    for i, (x, labels) in enumerate(val_loader):
      input = x.to(device).view(-1, 28*28).to(torch.float32)
      output, z, ave, log_dev = model(input)

      loss = criterion(output, input, ave, log_dev)
      history['val_loss'].append(loss)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {loss:.4f}")

  scheduler.step()

# loss with training data plot
train_loss_tensor = torch.stack(history["train_loss"])
train_loss_np = train_loss_tensor.to('cpu').detach().numpy().copy()
plt.plot(train_loss_np)
plt.show()
 
# loss with validation data plot
val_loss_tensor = torch.stack(history["val_loss"])
val_loss_np = val_loss_tensor.to('cpu').detach().numpy().copy()
plt.plot(val_loss_np)
plt.show()
