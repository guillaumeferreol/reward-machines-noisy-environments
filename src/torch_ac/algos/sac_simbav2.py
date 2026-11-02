"""
Soft Actor-Critic with the SimbaV2 architecture (Lee et al., "Hyperspherical Normalization for Scalable Deep
Reinforcement Learning", ICML 2025, https://proceedings.mlr.press/v267/lee25u.html). Equation numbers refer to the
paper; details it leaves open follow the official code (https://github.com/dojeon-ai/SimbaV2).
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_ac.algos.ppo import PPOAlgo
from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList, ParallelEnv, ReplayBuffer, RunningMeanStd


class RewardScaler(nn.Module):
    """Reward bounding and scaling (Eqs. 17-19).

    Tracks the discounted return G of every env, its running variance and its running maximum |G|, and divides
    rewards by max(sqrt(var(G) + eps), G_max_seen / g_max) so the critic's targets stay within [-g_max, g_max].
    """

    def __init__(self, num_procs, discount, g_max, epsilon=1e-8):
        super().__init__()
        self.discount = discount
        self.g_max = g_max
        self.epsilon = epsilon
        self.return_rms = RunningMeanStd()
        self.register_buffer("return_max", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("returns", torch.zeros(num_procs), persistent=False)

    @torch.no_grad()
    def update(self, reward, done):
        self.returns.mul_(self.discount).add_(reward)
        self.return_rms.update(self.returns)
        self.return_max.copy_(torch.maximum(self.return_max, self.returns.abs().max()))
        # G restarts from 0 with every episode
        self.returns.mul_(1 - done.float())

    def denominator(self):
        return torch.maximum(torch.sqrt(self.return_rms.var + self.epsilon), self.return_max / self.g_max).float()

    def forward(self, reward):
        return reward / self.denominator()


def project_onto_bins(probs, target_values, bin_values):
    """Categorical projection (Bellemare et al., 2017): moves the mass `probs[..., j]`, located at
    `target_values[..., j]`, onto the two nearest of the fixed, evenly spaced `bin_values`."""
    v_min, v_max, num_bins = bin_values[0], bin_values[-1], bin_values.numel()
    b = (target_values - v_min) / ((v_max - v_min) / (num_bins - 1))
    # Clamping the bin position (rather than the value) also catches GPU rounding just past the last bin
    b = b.clamp(0, num_bins - 1)
    lower, upper = b.floor(), b.ceil()
    # When b falls exactly on a bin, all the mass goes to `lower`
    m_lower = probs * (upper + (lower == upper).float() - b)
    m_upper = probs * (b - lower)

    projected = torch.zeros_like(probs)
    projected.scatter_add_(-1, lower.long(), m_lower)
    projected.scatter_add_(-1, upper.long(), m_upper)
    return projected


class SACSimbaV2Algo:
    """Soft Actor-Critic with SimbaV2 networks, a categorical critic and reward scaling.

    Like PPOAlgo, `collect_experiences` runs `num_frames_per_proc` steps in every environment and returns them so
    the detector can be trained on that rollout (`update_detector_parameters`). Unlike PPO, every transition also goes
    into a replay buffer, and `utd_ratio` SAC updates per frame are interleaved with the environment steps once the
    buffer holds `learning_starts` transitions.
    """

    def __init__(self, envs, model, detectormodel, device=None, num_frames_per_proc=100, discount=0.99,
                 batch_size=256, utd_ratio=2.0, buffer_size=10**6, learning_starts=5000,
                 lr=1e-4, lr_end=3e-5, lr_decay_steps=10**6, tau=0.005,
                 temperature_init=0.01, target_entropy=None, g_max=5.0, random_warmup=True,
                 detector_epochs=4, detector_batch_size=100, detector_lr=0.0003, detector_recurrence=4,
                 preprocess_obss=None, rm_update_algo="tdm"):
        """
        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        model : SimbaV2SACModel
            provides `obs_vector`, `obs_rms`, `actor` and `critic` (see src/simbav2_model.py)
        detectormodel : torch.Module
        num_frames_per_proc : int
            the number of frames collected by every process between two detector updates
        discount : float
            the discount for future rewards
        batch_size : int
            the number of transitions in each SAC update
        utd_ratio : float
            the number of SAC updates per collected frame (update-to-data ratio)
        buffer_size : int
            the capacity of the replay buffer
        learning_starts : int
            the number of transitions in the buffer before the SAC updates start
        lr, lr_end, lr_decay_steps : float, float, int
            the learning rate of the actor, critic and temperature decays linearly from `lr` to `lr_end`
            over `lr_decay_steps` updates
        tau : float
            the momentum of the target critic
        temperature_init : float
            the initial entropy temperature
        target_entropy : float
            the entropy the temperature is tuned to reach
        g_max : float
            the bound of the critic's return support [-g_max, g_max] after reward scaling
        random_warmup : bool
            act uniformly at random until the buffer holds `learning_starts` transitions (otherwise use the policy)
        """

        # Store parameters
        self.env = ParallelEnv(envs)
        self.model = model
        self.detectormodel = detectormodel
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.batch_size = batch_size
        self.utd_ratio = utd_ratio
        self.learning_starts = learning_starts
        self.tau = tau
        self.target_entropy = target_entropy
        self.random_warmup = random_warmup
        self.detector_epochs = detector_epochs
        self.detector_batch_size = detector_batch_size
        self.detector_recurrence = detector_recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.rm_update_algo = rm_update_algo
        self.use_rm_belief = (self.rm_update_algo in ["tdm", "naive", "ibu"])
        self.action_space = envs[0].action_space

        # Control parameters (oracle and no_rm do not use the detector)
        if self.use_rm_belief:
            assert self.detectormodel.recurrent or self.detector_recurrence == 1
            assert self.num_frames_per_proc % self.detector_recurrence == 0
            assert self.detector_batch_size % self.detector_recurrence == 0

        # Configure models
        self.model.to(self.device)
        self.model.train()
        self.target_critic = copy.deepcopy(self.model.critic)
        self.target_critic.requires_grad_(False)
        self.log_temperature = torch.tensor(np.log(temperature_init), dtype=torch.float, device=self.device,
                                            requires_grad=True)
        self.detectormodel.to(self.device)
        self.detectormodel.train()

        self.optimizers = {
            "actor": torch.optim.Adam(self.model.actor.parameters(), lr),
            "critic": torch.optim.Adam(self.model.critic.parameters(), lr),
            "temperature": torch.optim.Adam([self.log_temperature], lr),
        }
        self.schedulers = {
            name: torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=lr_end / lr,
                                                    total_iters=lr_decay_steps)
            for name, optimizer in self.optimizers.items()
        }
        if not self.detectormodel.params_free:
            self.optimizer_detector = torch.optim.Adam(self.detectormodel.parameters(), detector_lr)
        else:
            self.optimizer_detector = None

        # Store helpers values
        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs
        self.batch_num = 0
        self.update_budget = 0.

        # Replay buffer: raw observations, normalized when sampled with the current statistics
        if self.model.discrete:
            self.buffer = ReplayBuffer(buffer_size, self.model.obs_dim, (), torch.long, self.device)
        else:
            self.buffer = ReplayBuffer(buffer_size, self.model.obs_dim, self.action_space.shape, torch.float,
                                       self.device)
        self.reward_scaler = RewardScaler(self.num_procs, discount, g_max).to(self.device)
        # Transitions of the last step, completed by the next observation at the next step
        self.pending = None

        # Initialize experience values (kept for the detector)
        shape = (self.num_frames_per_proc, self.num_procs)

        self.obs = self.env.reset()
        self.obss = [None] * (shape[0])
        if self.detectormodel.recurrent:
            self.detector_memory = torch.zeros(shape[1], self.detectormodel.memory_size, device=self.device)
            self.detector_memories = torch.zeros(*shape, self.detectormodel.memory_size, device=self.device)
        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)

        # Initialize log values
        self.log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_num_frames = [1] * self.num_procs

    # The detector is trained on the latest rollout exactly as in PPO
    update_detector_parameters = PPOAlgo.update_detector_parameters
    _get_batches_starting_indexes = PPOAlgo._get_batches_starting_indexes

    def compute_rm_belief(self, preprocessed_obs):
        """Returns the RM state belief of the detector for the current observations, and its next memory."""
        detector_memory = None
        if self.detectormodel.recurrent:
            detector_belief, detector_memory = self.detectormodel(preprocessed_obs,
                                                                  self.detector_memory * self.mask.unsqueeze(1))
        else:
            detector_belief = self.detectormodel(preprocessed_obs)

        # tdm: softmax the RM state probabilities
        if self.rm_update_algo == "tdm":
            detector_belief = F.softmax(detector_belief, dim=1)

        # naive: use the event predictions to update RM env
        if self.rm_update_algo == "naive":
            detector_belief = self.env.update_rm_beliefs((detector_belief > 0).cpu().numpy())
            detector_belief = torch.tensor(np.array(detector_belief), device=self.device, dtype=torch.float)

        # ibu: use the event predictions to update RM env
        if self.rm_update_algo == "ibu":
            detector_belief = self.env.update_rm_beliefs(detector_belief.cpu().numpy())
            detector_belief = torch.tensor(np.array(detector_belief), device=self.device, dtype=torch.float)

        return detector_belief, detector_memory

    def sample_actions(self, x):
        """Actions for the raw observation vectors `x`, in the format stored in the replay buffer."""
        if self.random_warmup and len(self.buffer) < self.learning_starts:
            if self.model.discrete:
                return torch.randint(self.action_space.n, (self.num_procs,), device=self.device)
            return 2 * torch.rand((self.num_procs,) + self.action_space.shape, device=self.device) - 1

        obs = self.model.obs_rms.normalize(x)
        if self.model.discrete:
            return torch.distributions.Categorical(logits=self.model.actor(obs)).sample()
        return self.model.actor.sample(obs)[0]

    def store_transitions(self, x, action, reward, terminated, truncated, next_x):
        # ParallelEnv resets an env as soon as its episode ends, so `next_x` is then the first observation of the next
        # episode. That is harmless after a termination (the target does not bootstrap), but after a time-limit
        # truncation the final observation needed to bootstrap is lost, so those transitions are dropped.
        keep = terminated | ~truncated
        self.buffer.add(x[keep], action[keep], reward[keep], next_x[keep], terminated[keep].float())

    def collect_experiences(self):
        """Steps the environments for `num_frames_per_proc` frames, filling the replay buffer and performing
        `utd_ratio` SAC updates per frame after the warmup.

        Returns
        -------
        exps : DictList
            The observations, detector memories and masks of the rollout, for training the detector. The k-th block
            of consecutive `self.num_frames_per_proc` frames contains data obtained from the k-th environment.
        logs : dict
            Episode statistics and the average SAC losses over the updates of this rollout.
        """

        update_logs = []

        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction

            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)

            with torch.no_grad():
                # If necessary, add rm_belief to the observation
                if self.use_rm_belief:
                    detector_belief, detector_memory = self.compute_rm_belief(preprocessed_obs)
                    preprocessed_obs.rm_belief = detector_belief

                x = self.model.obs_vector(preprocessed_obs)
                self.model.obs_rms.update(x)

                if self.pending is not None:
                    self.store_transitions(*self.pending, x)

                action = self.sample_actions(x)

            obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
            reward = torch.tensor(reward, device=self.device, dtype=torch.float)
            terminated = torch.tensor(terminated, device=self.device)
            truncated = torch.tensor(truncated, device=self.device)
            done = terminated | truncated

            self.reward_scaler.update(reward, done)
            self.pending = (x, action, reward, terminated, truncated)

            # Update experiences values

            self.obss[i] = self.obs
            self.obs = obs
            if self.use_rm_belief and self.detectormodel.recurrent:
                self.detector_memories[i] = self.detector_memory
                self.detector_memory = detector_memory
            self.masks[i] = self.mask
            self.mask = 1 - done.float()

            # Update log values

            self.log_episode_return += reward
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for j, done_ in enumerate(done.tolist()):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[j].item())
                    self.log_num_frames.append(self.log_episode_num_frames[j].item())

            self.log_episode_return *= self.mask
            self.log_episode_num_frames *= self.mask

            # Update the actor and critic

            if len(self.buffer) >= self.learning_starts:
                self.update_budget += self.utd_ratio * self.num_procs
                while self.update_budget >= 1:
                    update_logs.append(self.update_ac_parameters())
                    self.update_budget -= 1

        # Define experiences for the detector (same layout as BaseAlgo.collect_experiences)

        exps = DictList()
        exps.obs = self.preprocess_obss([self.obss[i][j]
                                         for j in range(self.num_procs)
                                         for i in range(self.num_frames_per_proc)], device=self.device)
        if self.detectormodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.detector_memory = self.detector_memories.transpose(0, 1).reshape(-1, *self.detector_memories.shape[2:])
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        logs = {
            "return_per_episode": self.log_return[-keep:],
            # No reward shaping: the reshaped return is the return
            "reshaped_return_per_episode": self.log_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames,
            "num_updates": len(update_logs),
            "buffer_size": len(self.buffer),
            "temperature": self.log_temperature.exp().item(),
            "reward_scale": self.reward_scaler.denominator().item(),
        }
        for key in ["entropy", "q_value", "actor_loss", "critic_loss"]:
            logs[key] = torch.stack([l[key] for l in update_logs]).mean().item() if update_logs else float("nan")

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, logs

    def update_ac_parameters(self):
        """One SAC update on a batch sampled from the replay buffer: actor, then temperature, then critic
        (whose target uses the updated actor and temperature), then target critic, as in the official code."""

        batch = self.buffer.sample(self.batch_size)
        obs = self.model.obs_rms.normalize(batch.obs)
        next_obs = self.model.obs_rms.normalize(batch.next_obs)
        reward = self.reward_scaler(batch.reward)
        not_done = 1 - batch.done
        bin_values = self.model.critic.bin_values

        # Actor: minimize E[temperature * log pi(a|s) - Q(s, a)] (Eq. 1), with the minimum over the critics

        temperature = self.log_temperature.exp().detach()
        if self.model.discrete:
            # The critic is not updated before its own loss, so its output for all actions at s serves both losses
            critic_q, critic_log_probs = self.model.critic(obs)
            log_pi = self.model.actor(obs)
            pi = log_pi.exp()
            q = critic_q.detach().min(0).values
            actor_loss = (pi * (temperature * log_pi - q)).sum(-1).mean()
            entropy = -(pi * log_pi).sum(-1).mean()
        else:
            action, log_pi = self.model.actor.sample(obs)
            q = self.model.critic(obs, action)[0].min(0).values
            actor_loss = (temperature * log_pi - q).mean()
            entropy = -log_pi.mean()

        self.optimizers["actor"].zero_grad()
        actor_loss.backward(inputs=list(self.model.actor.parameters()))
        self.optimizers["actor"].step()
        self.model.actor.project_weights()

        # Temperature: move the policy entropy towards the target entropy

        temperature_loss = self.log_temperature.exp() * (entropy.detach() - self.target_entropy)
        self.optimizers["temperature"].zero_grad()
        temperature_loss.backward()
        self.optimizers["temperature"].step()

        # Critic: cross-entropy between the predicted return distribution and the projected soft Bellman target

        with torch.no_grad():
            temperature = self.log_temperature.exp()
            if self.model.discrete:
                # Target = sum_a' pi(a'|s') Proj[r + discount * (z - temperature * log pi(a'|s'))] under Z(s', a')
                next_log_pi = self.model.actor(next_obs)
                next_q, next_log_probs = self.target_critic(next_obs)
                next_log_probs = self._min_q_distribution(next_q, next_log_probs)
                target_values = (reward[:, None, None] + self.discount * not_done[:, None, None]
                                 * (bin_values - temperature * next_log_pi[..., None]))
                target_probs = project_onto_bins(next_log_probs.exp(), target_values, bin_values)
                target_probs = (next_log_pi.exp()[..., None] * target_probs).sum(1)
            else:
                next_action, next_log_pi = self.model.actor.sample(next_obs)
                next_q, next_log_probs = self.target_critic(next_obs, next_action)
                next_log_probs = self._min_q_distribution(next_q, next_log_probs)
                target_values = (reward[:, None] + self.discount * not_done[:, None]
                                 * (bin_values - temperature * next_log_pi[:, None]))
                target_probs = project_onto_bins(next_log_probs.exp(), target_values, bin_values)

        if self.model.discrete:
            # Select the buffer actions: (E, B, |A|, ...) -> (E, B, ...)
            index = batch.action[None, :, None].expand(critic_q.shape[0], -1, 1)
            q = critic_q.gather(2, index).squeeze(2)
            log_probs = critic_log_probs.gather(2, index[..., None].expand(-1, -1, -1, self.model.critic.num_bins))
            log_probs = log_probs.squeeze(2)
        else:
            q, log_probs = self.model.critic(obs, batch.action)
        # Sum over the critics of the mean over the batch
        critic_loss = -(target_probs * log_probs).sum(-1).mean(-1).sum()

        self.optimizers["critic"].zero_grad()
        critic_loss.backward()
        self.optimizers["critic"].step()
        self.model.critic.project_weights()

        # Target critic: Polyak averaging

        with torch.no_grad():
            torch._foreach_lerp_(list(self.target_critic.parameters()), list(self.model.critic.parameters()), self.tau)

        for scheduler in self.schedulers.values():
            scheduler.step()

        return {
            "entropy": entropy.detach(),
            "q_value": q.mean().detach(),
            "actor_loss": actor_loss.detach(),
            "critic_loss": critic_loss.detach(),
        }

    @staticmethod
    def _min_q_distribution(q, log_probs):
        """For each sample (and action), the return distribution of the critic with the lowest Q-value.
        q: (E, ...), log_probs: (E, ..., num_bins) -> (..., num_bins)"""
        index = q.argmin(0, keepdim=True)[..., None].expand(1, *log_probs.shape[1:])
        return log_probs.gather(0, index).squeeze(0)

    def state_dict(self):
        """Training state besides the model's own state_dict (the replay buffer is not saved)."""
        return {
            "target_critic": self.target_critic.state_dict(),
            "log_temperature": self.log_temperature.detach().clone(),
            "reward_scaler": self.reward_scaler.state_dict(),
            "optimizers": {name: optimizer.state_dict() for name, optimizer in self.optimizers.items()},
            "schedulers": {name: scheduler.state_dict() for name, scheduler in self.schedulers.items()},
        }

    def load_state_dict(self, state):
        self.target_critic.load_state_dict(state["target_critic"])
        with torch.no_grad():
            self.log_temperature.copy_(state["log_temperature"])
        self.reward_scaler.load_state_dict(state["reward_scaler"])
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state["optimizers"][name])
        for name, scheduler in self.schedulers.items():
            scheduler.load_state_dict(state["schedulers"][name])
