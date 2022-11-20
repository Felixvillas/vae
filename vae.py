import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, device=torch.device('cpu')) -> None:
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.device = device
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2*output_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_dim)
        )

        self.optimizer = Adam(self.parameters(), lr=3e-4)

        self.to(device)
    
    def check(self, x):
        if isinstance(x, np.ndarray):
            return torch.as_tensor(x, dtype=torch.float32, device=self.device)
        elif isinstance(x, torch.Tensor):
            return x.to(dtype=torch.float32, device=self.device)
        else:
            raise NotImplementedError
    
    def encode(self, x):
        '''
        get z from x:(batch_size, input_dim)
        '''
        x = self.check(x)
        params = self.encoder(x)
        mus, log_sigmas = torch.split(params, [self.output_dim, self.output_dim], dim=-1)
        sigmas = torch.exp(log_sigmas)
        z = self.rp_sample(mus, sigmas)
        '''
        Using torch.distributions.Normal().rsample() is another way for Reparameter sampling
        z_list = [torch.distributions.Normal(mu, sigma).rsample() for mu, sigma in zip(torch.unbind(mus), torch.unbind(sigmas))]
        z = torch.stack(z_list)
        '''
        return z, mus, sigmas

    def rp_sample(self, mus, sigmas):
        '''
        Reparameter sampling
        z ~ N(mu, sigma) --> (z-mu)/sqrt(sigma) ~ N(0, 1)
        '''
        bs, output_dim = mus.size()
        epsilon = torch.distributions.Normal(
            self.check(torch.zeros(output_dim)), self.check(torch.ones(output_dim))
        ).sample([bs, ])
        z = mus + epsilon * torch.sqrt(sigmas)
        return z

    def decode(self, z):
        '''
        get x from z:(batch_size, output_dim)
        '''
        z = self.check(z)
        x = self.decoder(z)
        return x

    def kl_loss(self, mus, sigmas):
        '''
        # This method for computing kl_divergence is too slow as 
        # it compute kl_divergence between every N(mu, sigma) and N(0, 1) serially
        _, output_dim = mus.size()
        prior = torch.distributions.Normal(
            self.check(torch.zeros(output_dim)), self.check(torch.ones(output_dim))
        )
        posteriors = [torch.distributions.Normal(mu, sigma) for mu, sigma in zip(torch.unbind(mus), torch.unbind(sigmas))]
        losses = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        # loss_kl = torch.mean(torch.stack(losses))
        loss_kl = torch.mean(torch.sum(torch.stack(losses), dim=-1), dim=0)
        '''
        # loss_kl = 0.5 * torch.mean(torch.sum(mus ** 2 + sigmas - torch.log(sigmas) - 1, dim=-1), dim=0)
        priors = torch.distributions.Normal(
            self.check(torch.zeros_like(mus)), self.check(torch.ones_like(sigmas))
        )
        posteriors = torch.distributions.Normal(mus, sigmas)
        loss_kl = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(posteriors, priors), dim=-1), dim=0)
        return loss_kl
    
    def update(self, x):
        '''
        optimize model
        x: (batch_size, input_dim)
        '''
        x = self.check(x)
        z, mus, sigmas = self.encode(x)
        x_re = self.decode(z)
        loss_re = torch.mean(torch.sum((x_re - x) ** 2, dim=-1), dim=0)
        loss_kl = self.kl_loss(mus, sigmas)
        loss = loss_re + loss_kl
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), loss_kl.item()

    def generate(self, num_sample):
        '''
        generate x from z ~ N(0, 1)
        '''
        z = torch.distributions.Normal(
            self.check(torch.zeros(self.output_dim)), self.check(torch.ones(self.output_dim))
        ).sample([num_sample, ])
        x = self.decode(z)
        return x

class CVAE(VAE):
    def __init__(self, input_dim, hidden_size, output_dim, label_dim, device=torch.device('cpu')) -> None:
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.label_dim = label_dim
        self.device = device
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size), 
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size), 
            nn.Linear(hidden_size, 2*output_dim)
        )
        self.label_encoder = nn.Sequential(
            nn.Linear(label_dim, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size), 
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size), 
            nn.Linear(hidden_size, output_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2*output_dim, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size), 
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size), 
            nn.Linear(hidden_size, input_dim)
        )

        self.optimizer = Adam(self.parameters(), lr=3e-4)

        self.to(device)

    def encode(self, x, y):
        x = self.check(x)
        y = self.check(y)
        
        params = self.encoder(x)
        mus, log_sigmas = torch.split(params, [self.output_dim, self.output_dim], dim=-1)
        sigmas = torch.exp(log_sigmas)
        z = self.rp_sample(mus, sigmas)
        y_mus = self.label_encoder(y)
        return z, mus, sigmas, y_mus

    def decode(self, z, y_mus):
        z = self.check(z)
        y_mus = self.check(y_mus)
        z_y = torch.cat([z, y_mus], dim=-1)
        x = self.decoder(z_y)
        return x

    # def kl_loss(self, mus, sigmas, y_mus):
    #     _, output_dim = mus.size()
    #     priors = [torch.distributions.Normal(y_mu, self.check(torch.ones(output_dim))) for y_mu in torch.unbind(y_mus)]
    #     posteriors = [torch.distributions.Normal(mu, sigma) for mu, sigma in zip(torch.unbind(mus), torch.unbind(sigmas))]
    #     losses = [torch.distributions.kl.kl_divergence(post, prior) for post, prior in zip(posteriors, priors)]
    #     loss_kl = torch.mean(torch.sum(torch.stack(losses), dim=-1), dim=0)
    #     return loss_kl

    def update(self, x, y):
        x = self.check(x)
        y = self.check(y)
        z, mus, sigmas, y_mus = self.encode(x, y)
        x_re = self.decode(z, y_mus)
        loss_re = torch.mean(torch.sum((x_re - x) ** 2, dim=-1), dim=0)
        # loss_kl = self.kl_loss(mus, sigmas, y_mus)
        loss_kl = self.kl_loss(mus, sigmas)
        # loss = torch.mean(torch.sum((x_re - self.check(x)) ** 2, dim=-1), dim=0) + self.kl_loss(mus, sigmas)
        loss = loss_re + loss_kl
        params_grad = []
        for params in self.parameters():
            params_grad.append(
                torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), loss_kl.item()

    def generate(self, num_sample):
        '''
        generate x from z ~ N(0, 1) for every label
        '''
        prior = torch.distributions.Normal(
            self.check(torch.zeros(self.output_dim)), self.check(torch.ones(self.output_dim))
        )
        z_list = [prior.sample([num_sample, ]) for _ in range(self.label_dim)]
        y_list = [self.check(torch.eye(self.label_dim)[label].repeat([num_sample, 1])) for label in range(self.label_dim)]
        y = torch.stack(y_list) # (label_dim, num_sample, label_dim)
        z = torch.stack(z_list) # (label_dim, num_sample, output_dim)
        y_mu = self.label_encoder(y)
        x = self.decode(z, y_mu)
        return x

class CVAE2(VAE):
    def __init__(self, input_dim, hidden_size, output_dim, label_dim, device=torch.device('cpu')) -> None:
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.label_dim = label_dim
        self.device = device
        self.encoder = nn.Sequential(
            nn.Linear(input_dim+label_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2*output_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim+label_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_dim)
        )

        self.optimizer = Adam(self.parameters(), lr=3e-4)

        self.to(device)

    def encode(self, x, y):
        x = self.check(x)
        y = self.check(y)
        params = self.encoder(torch.cat([x, y], dim=-1))
        mus, log_sigmas = torch.split(params, [self.output_dim, self.output_dim], dim=-1)
        sigmas = torch.exp(log_sigmas)
        z = self.rp_sample(mus, sigmas)
        return z, mus, sigmas

    def decode(self, z, y):
        z = self.check(z)
        y = self.check(y)
        z_y = torch.cat([z, y], dim=-1)
        x = self.decoder(z_y)
        return x

    def update(self, x, y):
        x = self.check(x)
        y = self.check(y)
        z, mus, sigmas = self.encode(x, y)
        x_re = self.decode(z, y)
        loss_re = torch.mean(torch.sum((x_re - x) ** 2, dim=-1), dim=0)
        loss_kl = self.kl_loss(mus, sigmas)
        loss = loss_re + loss_kl
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), loss_kl.item()

    def generate(self, num_sample):
        '''
        generate x from z ~ N(0, 1) for every label
        '''
        prior = torch.distributions.Normal(
            self.check(torch.zeros(self.output_dim)), self.check(torch.ones(self.output_dim))
        )
        z_list = [prior.sample([num_sample, ]) for _ in range(self.label_dim)]
        y_list = [self.check(torch.eye(self.label_dim)[label].repeat([num_sample, 1])) for label in range(self.label_dim)]
        y = torch.stack(y_list) # (label_dim, num_sample, label_dim)
        z = torch.stack(z_list) # (label_dim, num_sample, output_dim)
        x = self.decode(z, y)
        return x
