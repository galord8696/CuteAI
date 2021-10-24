'''
    Adapted from RealPython.com for educational purposes
'''

import torch
from torch import nn

import math
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(111)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.model(x)
        return output

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
    
    def forward(self, x):
        output = self.model(x)
        return output

class SinGAN():
    def __init__(self, train_data_length, batch_size):
        self.train_data_length = train_data_length
        self.batch_size = batch_size
    
    def train(self, lr, num_epochs):
        train_data_length = self.train_data_length
        train_data = torch.zeros((train_data_length, 2))
        train_data[:, 0] = 2 * math.pi * torch.rand(train_data_length)
        train_data[:, 1] = torch.sin(train_data[:, 0])
        train_labels = torch.zeros(train_data_length)
        train_set = [
            (train_data[i], train_labels[i]) for i in range(train_data_length)
        ]

        batch_size = self.batch_size
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True
        )

        discriminator = Discriminator()
        generator = Generator()

        loss_function = nn.BCELoss()

        optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
        optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

        f = plt.figure()
        
        for epoch in range(num_epochs):
            for n, (real_samples, _) in enumerate(train_loader):
                # Data for training discriminator
                real_samples_labels = torch.ones((batch_size, 1))
                latent_space_samples = torch.randn((batch_size, 2))
                generated_samples = generator(latent_space_samples)
                
                generated_samples_labels = torch.zeros((batch_size, 1))
                all_samples = torch.cat((real_samples, generated_samples))
                all_samples_labels = torch.cat(
                    (real_samples_labels, generated_samples_labels)
                )
                
                # Training discriminator
                discriminator.zero_grad()
                output_discriminator = discriminator(all_samples)
                loss_discriminator = loss_function(
                    output_discriminator, all_samples_labels
                )
                loss_discriminator.backward()
                optimizer_discriminator.step()
                
                # Data for training generator
                latent_space_samples = torch.randn((batch_size, 2))
                
                # Training the generator
                generator.zero_grad()
                generated_samples = generator(latent_space_samples)
                output_discriminator_generated = discriminator(generated_samples)
                loss_generator = loss_function(
                    output_discriminator_generated, real_samples_labels
                )
                loss_generator.backward()
                optimizer_generator.step()
                
                if epoch % 10 == 0 and n == batch_size - 1:
                    # print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
                    # print(f"Epoch: {epoch} Loss G.: {loss_generator}")
                    # newGen = generated_samples.detach()
                    plt.clf()
                    newGen = generated_samples.detach()
                    plt.plot(newGen[:, 0], newGen[:, 1], ".")
                    plt.xticks([])
                    plt.yticks([])
                    plt.savefig("images/SinGan.png", dpi=37, transparent=True, bbox_inches='tight')
            

        newGen = generated_samples.detach()
        plt.plot(newGen[:, 0], newGen[:, 1], ".")
        plt.xticks([])
        plt.yticks([])
        plt.savefig("images/SinGan.png", dpi=37, transparent=True, bbox_inches='tight')
                    




if __name__ == '__main__':
    gan = SinGAN(1024, 32)
    gan.train(0.001, 100)