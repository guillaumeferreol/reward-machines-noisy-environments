import torch

from torch_ac.utils.dictlist import DictList


class ReplayBuffer:
    """A FIFO buffer of (obs, action, reward, next_obs, done) transitions, sampled uniformly.

    Everything is stored as tensors on `device`, so sampling does not copy data between devices.
    """

    def __init__(self, capacity, obs_dim, action_shape, action_dtype, device=None):
        self.capacity = capacity
        self.device = device
        self.size = 0
        self.pos = 0

        self.obs = torch.zeros(capacity, obs_dim, device=device)
        self.next_obs = torch.zeros(capacity, obs_dim, device=device)
        self.actions = torch.zeros((capacity,) + tuple(action_shape), dtype=action_dtype, device=device)
        self.rewards = torch.zeros(capacity, device=device)
        self.dones = torch.zeros(capacity, device=device)

    def __len__(self):
        return self.size

    def add(self, obs, action, reward, next_obs, done):
        """Adds a batch of N transitions (every argument has N as its first dimension)."""
        n = obs.shape[0]
        idx = (self.pos + torch.arange(n, device=self.device)) % self.capacity
        self.obs[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_obs[idx] = next_obs
        self.dones[idx] = done

        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return DictList({
            "obs": self.obs[idx],
            "action": self.actions[idx],
            "reward": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "done": self.dones[idx],
        })
