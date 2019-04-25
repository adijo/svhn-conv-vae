import torch

from torch import nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=100):
        return input.view(input.size(0), size, 1, 1)


class VAE(nn.Module):
    def __init__(self, device="cpu", image_channels=3, h_dim=2304, z_dim=100):
        super(VAE, self).__init__()
        self.device = device
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(z_dim, 1024, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        std = log_var.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size(), device=self.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, log_var = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

    def encode(self, x):
        h = self.encoder(x)
        z, mu, log_var = self.bottleneck(h)
        return z, mu, log_var

    def decode(self, z):
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, log_var = self.encode(x)
        z = self.decode(z)
        return z, mu, log_var


def loss_fn(recon_x, x, mu, log_var):
    MSE = F.mse_loss(recon_x, x, reduction="sum")
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return MSE + KLD, MSE, KLD

