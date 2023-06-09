import numpy as np
import quadprog
import torch
import matplotlib.pyplot as plt
from torch.nn import functional as F


def to_one_hot(labels: torch.Tensor, num_class: int):
    y = torch.zeros(labels.shape[0], num_class)
    for i, label in enumerate(labels):
        y[i, label] = 1
    return y

class CVAELearning(torch.nn.Module):
    def __init__(self, net):
        super(CVAELearning, self).__init__()
        self.model = net
        self.old_model = None
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def loss_function(self, data, recon, mu, logvar):
        recon_loss = F.mse_loss(recon, data, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld

    def before(self,cur_task):
        if self.old_model != None:
            z = torch.randn(20, 32)
            l = np.random.randint(cur_task, size=(20, 1))
            memory_label = to_one_hot(l, num_class=6)
            zl = torch.cat((z, memory_label), dim=1)
            memory = self.old_model.decoder(zl).detach()
            return memory, memory_label
        else:
            return None, None

    def train(self,data_loader,epoch,memory, memory_label):
        self.model.train()
        running_loss = 0.0
        for data_train, label_train in data_loader:
            data_train = data_train.type(torch.FloatTensor)
            label_train = label_train.type(torch.LongTensor)
            y = to_one_hot(label_train.reshape(-1, 1), num_class=6)

            self.optim.zero_grad()

            recon, mu, logvar = self.model(data_train, y)
            loss_recon = self.loss_function(data_train, recon, mu, logvar)
            if self.old_model == None:
                loss_all = loss_recon
            else:
                recon_mem, mu_mem, log_var_mem = self.model(memory, memory_label)
                recon_cyc, mu_cyc, log_var_cyc = self.model(recon, y)
                loss_mem = self.loss_function(memory, recon_mem, mu_mem, log_var_mem)
                loss_cyc = self.loss_function(recon, recon_cyc, mu_cyc, log_var_cyc)
                loss_all = loss_recon + loss_mem + loss_cyc

            loss2 = loss_all
            loss2.backward()
            running_loss += loss2.item()
            self.optim.step()
        msg = '[%d] AVG. loss: %.3f\n' % (epoch, running_loss / 100)
        print(msg)

    def eval(self,task):
        self.model.eval()
        filename = './model/CVAEincrement_%d_net.pkl' % (task)
        print(filename)
        torch.save(self.model, filename)
        self.old_model = torch.load(filename)
        self.old_model.eval()

        z = torch.randn(20, 32)
        # down = 2 * task
        # up = 2 * (task + 1)
        down = task
        up = task + 1
        l = np.random.randint(down, up, size=(20, 1))
        memory_label = to_one_hot(l, num_class=6)
        zl = torch.cat((z, memory_label), dim=1)
        recon = self.old_model.decoder(zl).detach()
        return recon, l

