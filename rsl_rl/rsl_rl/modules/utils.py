import torch
import torch.nn as nn
import torch.nn.functional as F

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None

class L2Norm(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.normalize(x, p=2.0, dim=-1)

class SimNorm(nn.Module):
    """
    Simplicial normalization.
    Adapted from https://arxiv.org/abs/2204.00616.
    """

    def __init__(self):
        super().__init__()
        self.dim = 8  # for latent dim 512

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)

    def __repr__(self):
        return f"SimNorm(dim={self.dim})"

class MLP(nn.Module):
    def __init__(self, dims, activation='elu', last_activation=False):
        super().__init__()
        activation = get_activation(activation)
        layers = []
        last_dim = dims[0]
        for h_dim in dims[1:-1]:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(activation)
            last_dim = h_dim
        layers.append(nn.Linear(last_dim, dims[-1]))
        if last_activation:
            layers.append(activation)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class Experts(nn.Module):
    def __init__(self,
                 expert_num,
                 input_dim,
                 backbone_hidden_dims,
                 expert_hidden_dim,
                 output_dim,
                 activation='elu',
    ):
        super().__init__()
        self.expert_num = expert_num
        self.output_dim = output_dim

        self.backbone = MLP([input_dim, *backbone_hidden_dims, expert_num * expert_hidden_dim], activation, last_activation=True)
        self.experts = nn.Conv1d(
            in_channels=expert_num*expert_hidden_dim,
            out_channels=expert_num*output_dim,
            kernel_size=1,
            groups=expert_num,
        )
    
    def forward(self, x):
        shared_features = self.backbone(x).unsqueeze(-1)  # (B, expert_num * expert_hidden_dim, 1)
        expert_outs = self.experts(shared_features).squeeze(-1)  # (B, expert_num * output_dim)
        expert_outs = expert_outs.reshape(-1, self.expert_num, self.output_dim)
        return expert_outs

class MoE(nn.Module):
    def __init__(self,
                 expert_num,
                 input_dim,
                 hidden_dims,
                 output_dim,
                 activation='elu',
    ):
        super().__init__()

        # Expert networks
        self.experts = Experts(
            expert_num=expert_num,
            input_dim=input_dim,
            backbone_hidden_dims=hidden_dims[:-1],
            expert_hidden_dim=hidden_dims[-1],
            output_dim=output_dim,
            activation=activation,
        )
        
        # Gating network
        self.gating_network = nn.Sequential(
            MLP([input_dim, *hidden_dims[:-1], expert_num], activation),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        weights = self.gating_network(x)  # (B, expert_num)
        expert_outs = self.experts(x)  # (B, expert_num, output_dim)
        output = torch.sum(weights.unsqueeze(-1) * expert_outs, dim=1)  # (B, output_dim)
        return output, weights

class StudentMoEEncoder(nn.Module):
    def __init__(
        self,
        expert_num,
        input_dim,
        hidden_dims,
        output_dim,
        activation='elu',
        norm_type='l2norm',
    ):
        super().__init__()
        self.norm_layer = L2Norm() if norm_type == 'l2norm' else SimNorm()
        self.moe = MoE(
            expert_num=expert_num,
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
        )
    
    def forward(self, obs):
        latent, weights = self.moe(obs)
        latent = self.norm_layer(latent)
        return latent, weights

