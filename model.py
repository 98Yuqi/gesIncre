import torch
import torch.nn as nn
from torchsummary import summary

def to_one_hot(labels: torch.Tensor, num_class: int):
    y = torch.zeros(labels.shape[0], num_class)
    for i, label in enumerate(labels):
        y[i, label] = 1
    return y

class antoencoder(nn.Module):
    def __init__(self):
        super(antoencoder, self).__init__()
        self.en = nn.Sequential(
            nn.Conv1d(2,32, kernel_size=3, stride=1 ,padding=1,bias=False),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32,64, kernel_size=3, stride=1,padding=1,bias=False),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64,128, kernel_size=3, stride=1,padding=1,bias=False),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2)
        )
        self.linear = nn.Linear(128*25,32)
        self.unLinear = nn.Sequential(
            nn.Linear(38, 128*25),
            nn.ReLU(True),
            nn.Unflatten(dim=1, unflattened_size=(128,25))
        )
        self.de = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128,64, kernel_size=3, stride=1,padding=1,bias=False),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(64,32, kernel_size=3, stride=1, padding=1,bias=False),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(32,1, kernel_size=3, stride=1, padding=1,bias=False),
            nn.ReLU(True)
        )
    def reparametrize(self, mu, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(std)  # simple from standard normal distribution
        z = mu + eps * std
        return z

    def decoder(self,z):
        t = self.unLinear(z)
        t = self.de(t)
        return t

    def forward(self, x,label):
        y = torch.argmax(label, dim=1).reshape((label.shape[0], 1, 1))
        y = torch.ones(x.shape) * y
        t = torch.cat((x, y), dim=1)

        # encode t=[20,2,200]
        out = self.en(t)
        out = out.view(out.size(0), -1)
        mu = self.linear(out)
        logvar = self.linear(out)
        z = self.reparametrize(mu,logvar)

        # decode z=[20,48]
        z = torch.cat((z, label), dim=1)
        recon = self.decoder(z)

        return recon, mu, logvar

if __name__ == '__main__':
    model = antoencoder().cuda()
    summary(model, (2, 200))

