"""
SimbaV2 actor and critic networks for SAC.

Reference: Lee et al., "Hyperspherical Normalization for Scalable Deep Reinforcement Learning", ICML 2025
(https://proceedings.mlr.press/v267/lee25u.html). Equation, section and table numbers below refer to that paper.
Details the paper leaves open follow the authors' JAX code (https://github.com/dojeon-ai/SimbaV2).

Continuous actions follow the paper. Discrete actions use SAC-Discrete with the paper's |A| x n_atoms critic
output (Eq. 14): the critic takes only the observation and predicts a return distribution for every action.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.spaces import Box, Discrete

from torch_ac.utils import RunningMeanStd

EPS = 1e-8


def l2normalize(x, dim=-1):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(EPS)


class Scaler(nn.Module):
    """Learnable element-wise scale. The parameter starts at `scale` but acts as if it started at `init` (Eq. 21),
    so `scale` sets its learning rate independently of the optimizer's."""

    def __init__(self, dim, init, scale):
        super().__init__()
        self.scaler = nn.Parameter(torch.full((dim,), float(scale)))
        self.forward_scaler = init / scale

    def forward(self, x):
        return self.scaler * self.forward_scaler * x


class HyperDense(nn.Linear):
    """Bias-free linear layer whose weight vectors lie on the unit hypersphere (Section 4.4).

    Weights are initialized orthogonally and must be re-projected with `project()` after every optimizer step (Eq. 20).
    """

    def __init__(self, in_dim, out_dim):
        super().__init__(in_dim, out_dim, bias=False)
        nn.init.orthogonal_(self.weight)
        self.project()

    @torch.no_grad()
    def project(self):
        # One unit-norm weight vector per output feature
        self.weight.copy_(l2normalize(self.weight, dim=1))


class HyperEmbedder(nn.Module):
    """Input embedding: Shift + l2-Norm (Eq. 9), then Linear + Scaler + l2-Norm (Eq. 10)."""

    def __init__(self, in_dim, hidden_dim, scaler_init, scaler_scale, c_shift):
        super().__init__()
        self.c_shift = c_shift
        self.w = HyperDense(in_dim + 1, hidden_dim)
        self.scaler = Scaler(hidden_dim, scaler_init, scaler_scale)

    def forward(self, x):
        x = torch.cat([x, torch.full_like(x[..., :1], self.c_shift)], dim=-1)
        x = l2normalize(x)
        return l2normalize(self.scaler(self.w(x)))


class HyperLERPBlock(nn.Module):
    """Feature encoding block: inverted-bottleneck MLP + l2-Norm (Eq. 11), then LERP + l2-Norm (Eq. 12)."""

    expansion = 4

    def __init__(self, hidden_dim, scaler_init, scaler_scale, alpha_init, alpha_scale):
        super().__init__()
        self.w1 = HyperDense(hidden_dim, self.expansion * hidden_dim)
        # (sqrt(2/d_h) / sqrt(4) = sqrt(2/(4 d_h)), see Table 3)
        self.scaler = Scaler(self.expansion * hidden_dim,
                             scaler_init / math.sqrt(self.expansion), scaler_scale / math.sqrt(self.expansion))
        self.w2 = HyperDense(self.expansion * hidden_dim, hidden_dim)
        self.alpha = Scaler(hidden_dim, alpha_init, alpha_scale)

    def forward(self, x):
        # EPS keeps the ReLU output away from the zero vector
        h = F.relu(self.scaler(self.w1(x))) + EPS
        h = l2normalize(self.w2(h))
        return l2normalize(x + self.alpha(h - x))


class HyperHead(nn.Module):
    """Output prediction (Eq. 14): Linear + Scaler, then Linear. The final bias is from the official code."""

    def __init__(self, hidden_dim, out_dim, scaler_init, scaler_scale):
        super().__init__()
        self.w1 = HyperDense(hidden_dim, hidden_dim)
        self.scaler = Scaler(hidden_dim, scaler_init, scaler_scale)
        self.w2 = HyperDense(hidden_dim, out_dim)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        return self.w2(self.scaler(self.w1(x))) + self.bias


class SimbaV2Encoder(nn.Sequential):
    """Input embedding followed by `num_blocks` LERP blocks, initialized as in Table 3."""

    def __init__(self, in_dim, hidden_dim, num_blocks, c_shift):
        scaler = math.sqrt(2 / hidden_dim)
        alpha_init, alpha_scale = 1 / (num_blocks + 1), 1 / math.sqrt(hidden_dim)
        super().__init__(
            HyperEmbedder(in_dim, hidden_dim, scaler, scaler, c_shift),
            *[HyperLERPBlock(hidden_dim, scaler, scaler, alpha_init, alpha_scale) for _ in range(num_blocks)]
        )


def output_head(hidden_dim, out_dim):
    # Output scaler (sqrt(2/d_h), sqrt(2/d_h)) as in Table 3 (the official code uses (1, 1))
    scaler = math.sqrt(2 / hidden_dim)
    return HyperHead(hidden_dim, out_dim, scaler, scaler)


def project_weights(module):
    """Projects every HyperDense weight of `module` back onto the unit hypersphere (Eq. 20)."""
    for m in module.modules():
        if isinstance(m, HyperDense):
            m.project()


class SimbaV2Network(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_blocks, c_shift):
        super().__init__()
        self.encoder = SimbaV2Encoder(in_dim, hidden_dim, num_blocks, c_shift)
        self.head = output_head(hidden_dim, out_dim)

    def forward(self, x):
        return self.head(self.encoder(x))


class SimbaV2Actor(nn.Module):
    """Discrete actions: `forward` returns the log-probabilities of every action.
    Continuous actions: tanh-squashed Gaussian with separate mean and log-std heads, sampled with `sample`."""

    log_std_min = -10.0
    log_std_max = 2.0

    def __init__(self, obs_dim, action_space, hidden_dim, num_blocks, c_shift):
        super().__init__()
        self.discrete = isinstance(action_space, Discrete)
        self.encoder = SimbaV2Encoder(obs_dim, hidden_dim, num_blocks, c_shift)

        if self.discrete:
            self.logits = output_head(hidden_dim, action_space.n)
        elif isinstance(action_space, Box):
            assert (action_space.low == -1).all() and (action_space.high == 1).all()
            action_dim = action_space.shape[0]
            self.mean = output_head(hidden_dim, action_dim)
            self.log_std = output_head(hidden_dim, action_dim)
        else:
            raise NotImplementedError(f"Unsupported action_space type: {action_space}")

    def forward(self, obs):
        z = self.encoder(obs)
        if self.discrete:
            return F.log_softmax(self.logits(z), dim=-1)

        # Squash the log-std into [log_std_min, log_std_max]
        log_std = torch.tanh(self.log_std(z))
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (1 + log_std)
        return self.mean(z), log_std

    def sample(self, obs):
        """Reparameterized continuous action in [-1, 1] and its log-probability."""
        mean, log_std = self(obs)
        noise = torch.randn_like(mean)
        u = mean + log_std.exp() * noise
        action = torch.tanh(u)
        # log N(u; mean, std) - log|d tanh(u) / du|, where log(1 - tanh(u)^2) = 2 (log 2 - u - softplus(-2u))
        log_prob = -0.5 * noise.pow(2) - log_std - 0.5 * math.log(2 * math.pi)
        log_prob = log_prob - 2 * (math.log(2) - u - F.softplus(-2 * u))
        return action, log_prob.sum(-1)

    def project_weights(self):
        project_weights(self)


class SimbaV2Critic(nn.Module):
    """Ensemble of `num_critics` categorical Q-networks over `num_bins` returns in [v_min, v_max] (Eqs. 13-16).

    `forward` stacks the members' outputs along a leading dimension E:
        continuous: forward(obs, action) -> q (E, N), log_probs (E, N, num_bins)
        discrete:   forward(obs)         -> q (E, N, |A|), log_probs (E, N, |A|, num_bins)
    """

    def __init__(self, obs_dim, action_space, hidden_dim, num_blocks, c_shift, num_bins, v_min, v_max, num_critics):
        super().__init__()
        self.discrete = isinstance(action_space, Discrete)
        self.num_bins = num_bins
        self.v_min = v_min
        self.v_max = v_max

        if self.discrete:
            in_dim, out_dim = obs_dim, action_space.n * num_bins
        else:
            in_dim, out_dim = obs_dim + action_space.shape[0], num_bins
        self.nets = nn.ModuleList([SimbaV2Network(in_dim, out_dim, hidden_dim, num_blocks, c_shift)
                                   for _ in range(num_critics)])
        self.register_buffer("bin_values", torch.linspace(v_min, v_max, num_bins))

    def forward(self, obs, action=None):
        x = obs if self.discrete else torch.cat([obs, action], dim=-1)
        logits = torch.stack([net(x) for net in self.nets])
        if self.discrete:
            logits = logits.view(*logits.shape[:-1], -1, self.num_bins)

        log_probs = F.log_softmax(logits, dim=-1)
        q = (log_probs.exp() * self.bin_values).sum(-1)
        return q, log_probs

    def project_weights(self):
        project_weights(self)


class SimbaV2SACModel(nn.Module):
    """SAC actor and critic ensemble with the running observation statistics they share (RSNorm, Eq. 4)."""

    recurrent = False

    def __init__(self, obs_space, action_space, rm_update_algo,
                 actor_hidden_dim=128, actor_num_blocks=1,
                 critic_hidden_dim=512, critic_num_blocks=2, num_critics=2,
                 num_bins=101, g_max=5.0, c_shift=3.0):
        super().__init__()
        self.rm_update_algo = rm_update_algo
        self.discrete = isinstance(action_space, Discrete)

        obs_dim = int(np.prod(obs_space["image"]))
        if rm_update_algo != "no_rm":
            obs_dim += obs_space["rm_state"]
        self.obs_dim = obs_dim

        self.obs_rms = RunningMeanStd(obs_dim)
        self.actor = SimbaV2Actor(obs_dim, action_space, actor_hidden_dim, actor_num_blocks, c_shift)
        self.critic = SimbaV2Critic(obs_dim, action_space, critic_hidden_dim, critic_num_blocks, c_shift,
                                    num_bins, -g_max, g_max, num_critics)

    def obs_vector(self, obs):
        """Raw network input: the flattened env features, followed by the RM belief (tdm/naive/ibu) or the
        ground-truth RM state (oracle)."""
        x = obs.image.flatten(start_dim=1)
        if self.rm_update_algo == "no_rm":
            return x
        elif self.rm_update_algo in ["tdm", "ibu", "naive"]:
            return torch.cat((x, obs.rm_belief), dim=1)
        elif self.rm_update_algo == "oracle":
            return torch.cat((x, obs.rm_state), dim=1)
        else:
            raise NotImplementedError()
