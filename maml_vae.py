import torch
from torch.optim import Adam
import torch.nn.functional as F
from copy import deepcopy
from collections import OrderedDict
from torch.nn.utils.convert_parameters import parameters_to_vector

from vae import VAE
from utils import vector_to_parameters, maml_clip_grad_norm_

class My_MAML_VAE(VAE):
    def __init__(
        self, 
        input_dim, 
        hidden_size, 
        output_dim, 
        data_names,
        device=torch.device('cpu')
    ) -> None:
        super().__init__(input_dim, hidden_size, output_dim, device)
        self.lr = 1e-2
        self.max_norm = 0.25
        self.data_names = data_names
        self.copy_encoder = [deepcopy(self.encoder) for _ in data_names]
        self.copy_decoder = [deepcopy(self.decoder) for _ in data_names]
        theta_list = ['theta']
        theta_list.extend([f"theta{idx}" for idx, _ in enumerate(data_names)])
        self.encoder_theta = dict(zip(theta_list, [None for _ in theta_list]))
        self.decoder_theta = dict(zip(theta_list, [None for _ in theta_list]))
        self.ed2theta()

        self.to(device)

    def theta2thetap(self):
        for idx, _ in enumerate(self.data_names):
            self.encoder_theta[f"theta{idx}"] = self.encoder_theta['theta']
            self.decoder_theta[f"theta{idx}"] = self.decoder_theta['theta']

    def ed2theta(self):
        self.encoder_theta['theta'] = parameters_to_vector(self.encoder.parameters())
        self.decoder_theta['theta'] = parameters_to_vector(self.decoder.parameters())

    def theta2ed(self):
        vector_to_parameters(self.encoder_theta['theta'], self.encoder.parameters())
        vector_to_parameters(self.decoder_theta['theta'], self.decoder.parameters())
    
    def thetap2copy_ed(self):
        for idx, _ in enumerate(self.data_names):
            vector_to_parameters(self.encoder_theta[f"theta{idx}"], self.copy_encoder[idx].parameters())
            vector_to_parameters(self.decoder_theta[f"theta{idx}"], self.copy_decoder[idx].parameters())
            
    def check(self, x):
        if isinstance(x, list):
            return [self.check(x_i) for x_i in x]
        return super().check(x)

    def inner_encode(self, x):
        x = self.check(x)
        params = [encoder(x_i) for encoder, x_i in zip(self.copy_encoder, x)]
        mu_ls = [torch.split(param, [self.output_dim, self.output_dim], dim=-1) for param in params]
        mus = [mu_ls_i[0] for mu_ls_i in mu_ls]
        sigmas = [mu_ls_i[1].exp() for mu_ls_i in mu_ls]
        z = [self.rp_sample(mu, sigma) for mu, sigma in zip(mus, sigmas)]
        return z, mus, sigmas

    def inner_decode(self, z):
        z = self.check(z)
        x = [decoder(z_i) for decoder, z_i in zip(self.copy_decoder, z)]
        return x

    def inner_update(self, x):
        self.theta2thetap()
        x = self.check(x)
        z, mus, sigmas = tuple(zip(*[self.encode(x_i) for x_i in x]))
        x_re = [self.decode(z_i) for z_i in z]
        loss_re = [
            torch.mean(torch.sum((x_re_i - x_i) ** 2, dim=-1), dim=0) for x_re_i, x_i in zip(x_re, x)
        ]
        loss_kl = [
            self.kl_loss(mu, sigma) for mu, sigma in zip(mus, sigmas)
        ]
        loss = [loss_re_i + loss_kl_i for loss_re_i, loss_kl_i in zip(loss_re, loss_kl)]
        for idx, loss_i in enumerate(loss):
            # encoder
            grad_etheta = torch.autograd.grad(loss_i, self.encoder.parameters(), retain_graph=True)
            maml_clip_grad_norm_(grad_etheta, self.max_norm)
            grad_etheta_vector = parameters_to_vector(grad_etheta)
            self.encoder_theta[f"theta{idx}"] -= self.lr * grad_etheta_vector
            # decoder
            grad_dtheta = torch.autograd.grad(loss_i, self.decoder.parameters(), retain_graph=True)
            maml_clip_grad_norm_(grad_dtheta, self.max_norm)
            grad_dtheta_vector = parameters_to_vector(grad_dtheta)
            self.decoder_theta[f"theta{idx}"] -= self.lr * grad_dtheta_vector
        # load theta'p to copy encoder and decoder for outer update
        self.thetap2copy_ed()
        return [loss_re_i.item() for loss_re_i in loss_re], [loss_kl_i.item() for loss_kl_i in loss_kl]
        
    def outer_update(self, x):
        self.ed2theta()
        x = self.check(x)
        z, mus, sigmas = self.inner_encode(x)
        x_re = self.inner_decode(z)
        loss_re = [
            torch.mean(torch.sum((x_re_i - x_i) ** 2, dim=-1), dim=0) for x_re_i, x_i in zip(x_re, x)
        ]
        loss_kl = [
            self.kl_loss(mu, sigma) for mu, sigma in zip(mus, sigmas)
        ]
        loss = [loss_re_i + loss_kl_i for loss_re_i, loss_kl_i in zip(loss_re, loss_kl)]

        grad_ethetap_vector = 0
        grad_dthetap_vector = 0
        for idx, (loss_i, copy_encoder, copy_decoder) in enumerate(zip(loss, self.copy_encoder, self.copy_decoder)):
            '''
            这里没有考虑关于theta的二阶梯度
            '''
            # encoder
            grad_ethetap = torch.autograd.grad(loss_i, copy_encoder.parameters(), retain_graph=True)
            maml_clip_grad_norm_(grad_ethetap, self.max_norm)
            grad_ethetap_vector += parameters_to_vector(grad_ethetap)
            # decoder
            grad_dthetap = torch.autograd.grad(loss_i, copy_decoder.parameters(), retain_graph=True)
            maml_clip_grad_norm_(grad_dthetap, self.max_norm)
            grad_dthetap_vector += parameters_to_vector(grad_dthetap)

        self.encoder_theta['theta'] -= self.lr * (grad_ethetap_vector / (idx+1))
        self.decoder_theta['theta'] -= self.lr * (grad_dthetap_vector / (idx+1))
        self.theta2ed()

        return [loss_re_i.item() for loss_re_i in loss_re], [loss_kl_i.item() for loss_kl_i in loss_kl]

    def update(self, inner_xs, outer_xs):
        inner_loss_re, inner_loss_kl = self.inner_update(inner_xs)
        outer_loss_re, outer_loss_kl = self.outer_update(outer_xs)
        return inner_loss_re, inner_loss_kl, outer_loss_re, outer_loss_kl

class Simple_MAML_VAE(VAE):
    def __init__(
        self, 
        input_dim, 
        hidden_size, 
        output_dim, 
        data_names,
        device=torch.device('cpu')
    ) -> None:
        super().__init__(input_dim, hidden_size, output_dim, device)
        self.inner_lr = 3e-4
        self.max_grad_norm = 0.25
        self.data_names = data_names

        self.to(device)
                
    def check(self, x):
        if isinstance(x, list):
            return [self.check(x_i) for x_i in x]
        return super().check(x)


    def linear(self, x, w, b, is_relu):
        # x = torch.matmul(x, w.transpose(0, 1)) + b
        x = F.linear(x, w, b)
        if is_relu:
            x = F.relu(x)
        return x

    def functional_forward(self, x, params):
        # encode
        x_1 = self.linear(x, params.get('encoder.0.weight'), params.get('encoder.0.bias'), is_relu=True)
        x_2 = self.linear(x_1, params.get('encoder.2.weight'), params.get('encoder.2.bias'), is_relu=True)
        paras = self.linear(x_2, params.get('encoder.4.weight'), params.get('encoder.4.bias'), is_relu=False)
        # rsample
        mus, log_sigmas = torch.split(paras, [self.output_dim, self.output_dim], dim=-1)
        sigmas = torch.exp(log_sigmas)
        z = self.rp_sample(mus, sigmas)
        # decode
        z_1 = self.linear(z, params.get('decoder.0.weight'), params.get('decoder.0.bias'), is_relu=True)
        z_2 = self.linear(z_1, params.get('decoder.2.weight'), params.get('decoder.2.bias'), is_relu=True)
        x_re = self.linear(z_2, params.get('decoder.4.weight'), params.get('decoder.4.bias'), is_relu=False)
        # kl loss
        kl_loss = self.kl_loss(mus, sigmas)
        # reconstruct loss
        re_loss = torch.sum((x - x_re) ** 2, dim=-1).mean()

        return re_loss, kl_loss

    def update(self, inner_xs, outer_xs, inner_step=1):
        inner_xs = self.check(inner_xs)
        outer_xs = self.check(outer_xs)

        inner_losses_re = []
        inner_losses_kl = []
        outer_losses_re = []
        outer_losses_kl = []

        meta_loss = []
        for inner_x, outer_x in zip(inner_xs, outer_xs):
            # 每次循环对应一个task
            fast_weights = OrderedDict(self.named_parameters())
            for _ in range(inner_step):
                inner_loss_re, inner_loss_kl = self.functional_forward(inner_x, fast_weights)
                inner_loss = inner_loss_re + inner_loss_kl
                grads = torch.autograd.grad(inner_loss, fast_weights.values(), create_graph=True)
                # maml_clip_grad_norm_(grads, self.max_grad_norm)
                fast_weights = OrderedDict(
                    (name, param - self.inner_lr * grad) 
                    for (name, param), grad in zip(fast_weights.items(), grads)
                )

                # record loss
                inner_losses_re.append(inner_loss_re.item())
                inner_losses_kl.append(inner_loss_kl.item())
            outer_loss_re, outer_loss_kl = self.functional_forward(outer_x, fast_weights)
            outer_loss = outer_loss_re + outer_loss_kl
            meta_loss.append(outer_loss)
            
            # record loss
            outer_losses_re.append(outer_loss_re.item())
            outer_losses_kl.append(outer_loss_kl.item())

        loss = torch.stack(meta_loss).mean()
        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()
            
        return inner_losses_re, inner_losses_kl, outer_losses_re, outer_losses_kl