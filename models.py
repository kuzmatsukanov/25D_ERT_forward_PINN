import sys
#from logger import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
from siren_pytorch import SirenNet

import math

class FCNet(nn.Module):
    """Fully connected net [input_dim, hidden_dim, output_dim]"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers):
        """
        Args:
            input_dim (int): number of input dimensions
            hidden_dim (int): number of hidden dimensions
            output_dim (int): number of output dimensions
            num_hidden_layers (int): number of hidden layers
        """
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)])  # more layers with 'num_hidden_layers' neurons each
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self._initialize_weights()

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        for layer in self.fc_layers:
            x = torch.tanh(layer(x))
        x = self.fc_out(x)
        return x

    def _initialize_weights(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)

        for layer in self.fc_layers:
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

        torch.nn.init.xavier_uniform_(self.fc_out.weight)
        torch.nn.init.zeros_(self.fc_out.bias)

class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

def siren_init(layer, w0, c=6.0):
    with torch.no_grad():
        num_in = layer.in_features
        bound = math.sqrt(c / num_in) / w0
        layer.weight.uniform_(-bound, bound)

class SirenSubnet(nn.Module):
    def __init__(self,
                 dim_in: int,
                 dim_hidden: int,
                 dim_out: int,
                 num_layers: int,
                 w0_initial: float = 30.,
                 w0: float = 1.0,
                 use_skip: bool = True):
        super().__init__()
        self.use_skip = use_skip

        # first layer (inject high-freq)
        self.first_lin = nn.Linear(dim_in, dim_hidden)
        siren_init(self.first_lin, w0_initial)
        self.first_act = Sine(w0_initial)

        # remaining hidden layers
        self.hidden = nn.ModuleList()
        for _ in range(num_layers - 1):
            lin = nn.Linear(dim_hidden, dim_hidden)
            siren_init(lin, w0)
            act = Sine(w0)
            self.hidden.append(nn.ModuleList([lin, act]))

        # final linear to output_dim
        self.final = nn.Linear(dim_hidden, dim_out)
        nn.init.xavier_uniform_(self.final.weight)

    def forward(self, x):
        # first layer
        h = self.first_act(self.first_lin(x))

        # hidden + skip
        for lin, act in self.hidden:
            h_prev = h
            h = act(lin(h))
            if self.use_skip:
                h = h + h_prev

        # to output
        return self.final(h)

class MSSiren(nn.Module):
    r"""
    Multi‑Scale SIREN (MS‑SIREN)
    φ(xyz) = Σ_k  f_k(a_k · xyz)          # Eq. (12) in Huang et al (2022).
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers, scale_factors = (1, 2, 4, 8), w0_initial = 30.0, w0 = 1.0, use_skip=True):
        """
        Args:
            input_dim (int): number of input dimensions
            hidden_dim (int): number of hidden dimensions
            output_dim (int): number of output dimensions
            num_hidden_layers (int): number of hidden layers
            scale_factors (tuple of float): Multiplicative factors aₖ applied to the input before it is fed into the k‑th sub‑net.
                Choose a geometric progression to cover the frequency spectrum you expect in the solution.
            w0_initial (float): frequency factor w₀ for the *first* layer of every sub‑net. A higher value injects high‑frequency features.
            w0 (float): frequency factor for all subsequent layers in every sub‑net.0
        """
        super().__init__()
        self.scale_factors = scale_factors
        self.use_skip = use_skip

        self.subnets = nn.ModuleList([
            SirenSubnet(dim_in = input_dim,     #SirenNet
                     dim_hidden = hidden_dim,
                     dim_out = output_dim,
                     num_layers  = num_hidden_layers,
                     w0_initial  = w0_initial,
                     w0          = w0,       # all other layers
                     use_skip    = self.use_skip)
            for _ in scale_factors
        ])

    def forward(self, xzk):
        """
        Args:
            xz (torch.Tensor): collocation points with shape [B, D=2]
        """
        outs = [net(a * xzk) for a, net in zip(self.scale_factors, self.subnets)] # scale input → run subnet
        return torch.stack(outs, 0).sum(0)  # element‑wise Σ

class ModelSingleElectrode(nn.Module):
    def __init__(self, latent_dim, num_hidden_layers_tr, num_hidden_layers_br, scale_factors=(1, 2, 4, 8), use_skip=True):
        super(ModelSingleElectrode, self).__init__()
        # Set architecture
        # Tr(x,z) * Br(k)
        # trunk: MS-SIREN. branch: FCNet
        latent_dim = latent_dim
        self.trunk_net = MSSiren(input_dim=2, hidden_dim=latent_dim, output_dim=latent_dim, num_hidden_layers=num_hidden_layers_tr, scale_factors=scale_factors, use_skip=use_skip)
        self.branch_net = FCNet(input_dim=1, hidden_dim=latent_dim, output_dim=latent_dim, num_hidden_layers=num_hidden_layers_br)

    def forward(self, xzk):
        """
        It accepts not k but log10(k)
        Args:
            xzk: (torch.Tensor): of shape [B, 3]. xz are collocation points. NOT k but log10k is a wavenumber
        Returns:
            torch.Tensor: shape [B, 1], φ₁(x, z, logk) predicted potential
        """
        assert xzk.dim() == 2 and xzk.size(-1) == 3, "xzk must be [B,3]"
        #B, _ = xzk.shape

        # Split
        xz   = xzk[..., :2]                 # [B,2]
        logk = xzk[...,  2:3].detach()      # [B,1] no grad wrt k

        trunk_output = self.trunk_net(xz)  # [B, L]  # Encode the coordinates xz (points where to evaluate solution)
        branch_output = self.branch_net(logk)  # [B, L]  # Encode the logk value

        assert trunk_output.dim() == 2
        assert branch_output.dim() == 2
        assert trunk_output.shape == branch_output.shape

        u_pred = (trunk_output * branch_output).sum(dim=-1, keepdim=True)  # [B, 1]
        return u_pred

    def log_initialization(self, logger):
        logger.info(
            "Model initialized:\n"
            f"  - Architecture:\n"
            f"  {self}\n"
            f"  - Parameters:\n"
            f"  -- model.trunk_net.scale_factors: {self.trunk_net.scale_factors}\n"
            f"  -- model.trunk_net.use_skip: {self.trunk_net.use_skip}"
        )
