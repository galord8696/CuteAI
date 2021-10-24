'''
    Adapted from RealPython.com for educational purposes
'''

import torch
from torch import nn

import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

torch.manual_seed(111)

import warnings
warnings.filterwarnings("ignore")

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), 784)
        output = self.model(x)
        return output

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def forward(self, x):
        output = self.model(x)
        output = output.view(x.size(0), 1, 28, 28)
        return output

class mnistGAN():
    def __init__(self, batch_size):
        self.batch_size = batch_size
    
    def train(self, lr, num_epochs):
        device = ""
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

        train_set = torchvision.datasets.MNIST(
            root=".", train=True, download=True, transform=transform
        )

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True
        )

        discriminator = Discriminator().to(device=device)
        generator = Generator().to(device=device)
        
        loss_function = nn.BCELoss()

        optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
        optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

        f = plt.figure()
        
        for epoch in range(num_epochs):
            for n, (real_samples, mnist_labels) in enumerate(train_loader):
                # Data for training the discriminator
                real_samples = real_samples.to(device=device)
                real_samples_labels = torch.ones((self.batch_size, 1)).to(
                    device=device
                )
                latent_space_samples = torch.randn((self.batch_size, 100)).to(
                    device=device
                )
                generated_samples = generator(latent_space_samples)
                generated_samples_labels = torch.zeros((self.batch_size, 1)).to(
                    device=device
                )
                all_samples = torch.cat((real_samples, generated_samples))
                all_samples_labels = torch.cat(
                    (real_samples_labels, generated_samples_labels)
                )

                # Training the discriminator
                discriminator.zero_grad()
                output_discriminator = discriminator(all_samples)
                loss_discriminator = loss_function(
                    output_discriminator, all_samples_labels
                )
                loss_discriminator.backward()
                optimizer_discriminator.step()

                # Data for training the generator
                latent_space_samples = torch.randn((self.batch_size, 100)).to(
                    device=device
                )

                # Training the generator
                generator.zero_grad()
                generated_samples = generator(latent_space_samples)
                output_discriminator_generated = discriminator(generated_samples)
                loss_generator = loss_function(
                    output_discriminator_generated, real_samples_labels
                )
                loss_generator.backward()
                optimizer_generator.step()
                # print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
                # print(f"Epoch: {epoch} Loss G.: {loss_generator}")
                # newGen = generated_samples.detach()
                if n == self.batch_size - 1:
                    print("epoch: %d" % (epoch))
                    plt.clf()
                    generated_samples = generated_samples.cpu().detach()
                    for i in range(4):
                        ax = plt.subplot(4, 4, i + 1)
                        plt.imshow(generated_samples[i].reshape(28, 28), cmap="gray_r")
                        plt.xticks([])
                        plt.yticks([])
                    plt.savefig("images/mnistGan.png", dpi=37, transparent=True, bbox_inches='tight')
            
        plt.clf()
        generated_samples = generated_samples.cpu().detach()
        for i in range(16):
            ax = plt.subplot(4, 4, i + 1)
            plt.imshow(generated_samples[i].reshape(28, 28), cmap="gray_r")
            plt.xticks([])
            plt.yticks([])
        plt.savefig("images/mnistGan.png", dpi=37, transparent=True, bbox_inches='tight')
                    




if __name__ == '__main__':
    gan = mnistGAN(32)
    gan.train(0.0001, 3)