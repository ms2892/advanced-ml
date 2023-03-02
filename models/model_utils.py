
import numpy as np
import torch
import torch.nn.functional as F

class BBB_Hyper(object):

    def __init__(self, ):
        self.dataset = 'mnist'  # mnist || cifar10 || fmnist

        self.lr = 1e-4
        self.momentum = 0.95
        self.hidden_units = 400
        self.mixture = True
        self.pi = 0.75
        self.s1 = float(np.exp(-8))
        self.s2 = float(np.exp(-1))
        self.rho_init = -8
        self.multiplier = 1.

        self.max_epoch = 600
        self.n_samples = 1
        self.n_test_samples = 10
        self.batch_size = 125
        self.eval_batch_size = 1000


GAUSSIAN_SCALER = 1. / np.sqrt(2.0 * np.pi)
def gaussian(x, mu, sigma):
    bell = torch.exp(- (x - mu) ** 2 / (2.0 * sigma ** 2))
    return GAUSSIAN_SCALER / sigma * bell


def mixture_prior(input, pi, s1, s2):
    prob1 = pi * gaussian(input, 0., s1)
    prob2 = (1. - pi) * gaussian(input, 0., s2)
    return torch.log(prob1 + prob2)


def log_gaussian(x, mu, sigma):
    return float(-0.5 * np.log(2 * np.pi) - np.log(np.abs(sigma))) - (x - mu) ** 2 / (2 * sigma ** 2)

def log_gaussian_rho(x, mu, rho):
    return float(-0.5 * np.log(2 * np.pi)) - rho - (x - mu) ** 2 / (2 * torch.exp(rho) ** 2)


def probs(model, hyper, data, target):
    s_log_pw, s_log_qw, s_log_likelihood = 0., 0., 0.
    for _ in range(hyper.n_samples):
        output = torch.log(model(data))

        sample_log_pw, sample_log_qw = model.get_lpw_lqw()
        sample_log_likelihood = -F.nll_loss(output, target, reduction='sum') * hyper.multiplier

        s_log_pw += sample_log_pw / hyper.n_samples
        s_log_qw += sample_log_qw / hyper.n_samples
        s_log_likelihood += sample_log_likelihood / hyper.n_samples

    return s_log_pw, s_log_qw, s_log_likelihood

def ELBO(l_pw, l_qw, l_likelihood, beta):
    kl = beta * (l_qw - l_pw)
    return kl - l_likelihood