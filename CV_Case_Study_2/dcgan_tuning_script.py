
import torch
from torch import nn, optim
import random

# Define Generator
class Generator(nn.Module):
    def __init__(self, z_dim, channels=3, features=64, slope=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, features * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features * 16),
            nn.LeakyReLU(slope, inplace=True),

            nn.ConvTranspose2d(features * 16, features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(slope, inplace=True),

            nn.ConvTranspose2d(features * 8, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(slope, inplace=True),

            nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(slope, inplace=True),

            nn.ConvTranspose2d(features * 2, features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.LeakyReLU(slope, inplace=True),

            nn.ConvTranspose2d(features, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

# Define Discriminator
class Discriminator(nn.Module):
    def __init__(self, channels=3, features=64, slope=0.2, dropout=False):
        super().__init__()
        layers = [
            nn.Conv2d(channels, features, 4, 2, 1, bias=False),
            nn.LeakyReLU(slope, inplace=True)
        ]
        if dropout: layers.append(nn.Dropout2d(0.3))
        layers += [
            nn.Conv2d(features, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(slope, inplace=True)
        ]
        if dropout: layers.append(nn.Dropout2d(0.3))
        layers += [
            nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(slope, inplace=True),
            nn.Conv2d(features * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
            nn.AdaptiveAvgPool2d((1, 1))
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).view(-1, 1)

# Hyperparameter space
param_space = {
    "z_dim": [64, 128],
    "features": [32, 64],
    "lr": [0.0002, 0.00005],
    "beta1": [0.5, 0.7],
    "slope": [0.1, 0.2, 0.3],
    "dropout": [True, False],
    "label_smoothing": [True, False]
}

def sample_hparams():
    return {k: random.choice(v) for k, v in param_space.items()}

def improved_tuning_loop(num_trials=5):
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(num_trials):
        h = sample_hparams()
        print(f"🔧 Trial {i+1}: {h}")

        G = Generator(h['z_dim'], features=h['features'], slope=h['slope']).to(device)
        D = Discriminator(features=h['features'], slope=h['slope'], dropout=h['dropout']).to(device)

        opt_G = optim.Adam(G.parameters(), lr=h['lr'], betas=(h['beta1'], 0.999))
        opt_D = optim.Adam(D.parameters(), lr=h['lr'], betas=(h['beta1'], 0.999))
        loss_fn = nn.BCELoss()

        batch_size = 32
        real_imgs = torch.randn(batch_size, 3, 128, 128).to(device)
        z = torch.randn(batch_size, h['z_dim'], 1, 1).to(device)

        fake_imgs = G(z).detach()
        real_labels = torch.full((batch_size, 1), 0.9 if h['label_smoothing'] else 1.0).to(device)
        fake_labels = torch.zeros((batch_size, 1)).to(device)

        # Train D
        D_real = D(real_imgs)
        D_fake = D(fake_imgs)
        loss_D = loss_fn(D_real, real_labels) + loss_fn(D_fake, fake_labels)
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # Train G
        z = torch.randn(batch_size, h['z_dim'], 1, 1).to(device)
        fake_imgs = G(z)
        output = D(fake_imgs)
        loss_G = loss_fn(output, real_labels)
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        h["D_loss"] = loss_D.item()
        h["G_loss"] = loss_G.item()
        results.append(h)

    return results

# Run it
if __name__ == "__main__":
    results = improved_tuning_loop()
    for r in results:
        print(r)
