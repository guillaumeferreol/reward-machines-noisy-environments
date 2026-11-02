import torch
import torch.nn as nn


class RunningMeanStd(nn.Module):
    """Running mean and variance of a stream of batches (parallel algorithm, as in gymnasium).

    The statistics are buffers, so they are saved and restored with the owning module's state_dict.
    """

    def __init__(self, shape=(), epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float64))
        self.register_buffer("var", torch.ones(shape, dtype=torch.float64))
        self.register_buffer("count", torch.tensor(1e-4, dtype=torch.float64))

    @torch.no_grad()
    def update(self, x):
        """Updates the statistics with a batch `x` of shape (N, *shape)."""
        x = x.to(torch.float64)
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        m2 = self.var * self.count + batch_var * batch_count + delta.pow(2) * self.count * batch_count / tot_count

        self.mean.add_(delta * batch_count / tot_count)
        self.var.copy_(m2 / tot_count)
        self.count.copy_(tot_count)

    def normalize(self, x):
        return ((x - self.mean) / torch.sqrt(self.var + self.epsilon)).to(x.dtype)
