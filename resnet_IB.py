import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import torchvision
def KL_between_normals(q_distr, p_distr):
    mu_q, sigma_q = q_distr
    mu_p, sigma_p = p_distr
    k = mu_q.size(1)

    mu_diff = mu_p - mu_q
    mu_diff_sq = torch.mul(mu_diff, mu_diff)
    logdet_sigma_q = torch.sum(2 * torch.log(torch.clamp(sigma_q, min=1e-8)), dim=1)
    logdet_sigma_p = torch.sum(2 * torch.log(torch.clamp(sigma_p, min=1e-8)), dim=1)

    fs = torch.sum(torch.div(sigma_q ** 2, sigma_p ** 2), dim=1) + torch.sum(torch.div(mu_diff_sq, sigma_p ** 2), dim=1)
    two_kl = fs - k + logdet_sigma_p - logdet_sigma_q
    return two_kl * 0.5

class resnetIB(nn.Module):
    """
    Hydrogen IB model (FC).
    """
    def __init__(self, device):
        super(resnetIB, self).__init__()
        self.beta = 0.12
        self.z_dim = 256
        self.num_sample = 12
        self.device = device
        self.batch_size = 100
        self.prior = Normal(torch.zeros(1, self.z_dim).to(self.device),torch.ones(1, self.z_dim).to(self.device))
        self.feature_net = torchvision.models.resnet34(pretrained= True)
        self.Encoder = nn.Sequential(nn.Linear(in_features=512, out_features=1024),
                                     nn.ReLU(),
                                     nn.Linear(in_features=1024, out_features=1024),
                                     nn.ReLU(),
                                     nn.Linear(in_features=1024, out_features=2 * self.z_dim))

        self.Decoder = nn.Sequential(nn.Linear(in_features=self.z_dim, out_features=10))
        self.feature_net.fc = self.Encoder

    def gaussian_noise(self, num_samples, K):
        return torch.normal(torch.zeros(*num_samples, K), torch.ones(*num_samples, K)).to(self.device)

    def sample_prior_Z(self, num_samples):
        return self.gaussian_noise(num_samples=num_samples, K=self.dimZ)

    def encoder_result(self, x):
        encoder_output = self.feature_net(x)
        mean_C = encoder_output[:, :self.z_dim]
        std_C = torch.nn.functional.softplus(encoder_output[:, self.z_dim:])
        return mean_C, std_C

    def sample_Z(self, num_samples, x):
        mean_C, std_C = self.encoder_result(x)
        return mean_C, std_C, mean_C + std_C * self.gaussian_noise(num_samples=(num_samples, self.batch_size), K=self.z_dim)
               # mean_S, std_S, mean_S + std_S * self.gaussian_noise(num_samples=(num_samples, batch_size), K=self.z_dim)

    def get_logits(self, x):
        #encode
        mean_C, std_C, z_C = self.sample_Z(num_samples=self.num_sample, x=x)

        #decode
        y_logits = self.Decoder(z_C)
        z_C = torch.mean(z_C, dim=0)

        return mean_C, std_C, y_logits, z_C
        # mean_C (100, 256)
        # y_logits (100, 256)
        # y_logits (12, 100, 10)

    def forward(self, x):
        mean_C, std_C, y_logits, z_C = self.get_logits(x)
        y_pre = torch.mean(y_logits, dim=0)
        return y_pre
        # y_pre (100, 10)

    def batch_loss(self, x_batch, y_batch, num_samples):
        # compute I(X,T)
        prior_Z_distr = torch.zeros(self.batch_size, self.z_dim).to(self.device), torch.ones(self.batch_size, self.z_dim).to(self.device)
        mean, std, y_logits, z_C =  self.get_logits(x_batch)
        enc_dist = mean, std
        I_X_T_bound = torch.mean(KL_between_normals(enc_dist, prior_Z_distr)) / math.log(2)

        # compute I(Y,T)
        # y_logits (12, 100, 10)
        loss_func = nn.CrossEntropyLoss(reduce=False)
        y_logits = y_logits.permute(1, 2, 0)    # y_logits (100, 10, 12)
        y_label = y_batch[:, None].expand(-1, num_samples)      # y_label (100, 12)
        cross_entropy_loss = loss_func(y_logits, y_label)
        cross_entropy_loss_montecarlo = torch.mean(cross_entropy_loss, dim=-1)
        I_Y_T_bound = torch.mean(cross_entropy_loss_montecarlo, dim=0) / math.log(2)


        # compute Loss
        loss = I_Y_T_bound + self.beta*I_X_T_bound

        return loss, I_X_T_bound, math.log(10, 2) - I_Y_T_bound

    def give_beta(self, beta):
        self.beta = beta


class EMA(nn.Module):
    def __init__(self, mu):
        super(EMA, self).__init__()
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def forward(self, name, x):
        assert name in self.shadow
        new_average = self.mu * x + (1.0 - self.mu) * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average
