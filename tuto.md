# Tutorial: Reward Machines in Noisy Environments

How to launch trainings, which environments are available, how to plot results, and how to visualize agents.
The commands below were run end to end on short runs. The exceptions are the interactive window scripts
(`*_manual_agent.py`, `mujoco_visualize.py`) and the toy gold-mining script, which are marked as untested.

## 0. Before you start

Run everything from the repo root with the environment set up:

```bash
cd /home/gferreol/Documents/Reward_Machines/reward-machines-noisy-environments
. setup.sh        # PYTHONPATH, MuJoCo libs, GLEW preload, and the `reward` env on PATH
```

Paths like `src/envs/rm_files` and `storage/` are relative, so running from anywhere else breaks.

Extra packages needed beyond `requirements.txt`: `tqdm` (data collection), `matplotlib` and `pandas` (plotting).
For GIF export from `minigrid_visualize.py` you also need `pip install array2gif`.

## 1. The idea in one paragraph

A **Reward Machine (RM)** is a finite-state machine whose transitions fire on propositions (`a`, `b`, `c`, ...) and emit
rewards. Normally you know exactly which propositions are true at each step. This repo handles the case where you don't,
because the sensors are noisy or the view is partial. A learned **detector** (abstraction model) guesses the propositions
or the RM state, and the PPO policy acts on that guess.

## 2. Environments

| ID | Type | Task | Events | RM file |
|---|---|---|---|---|
| `Kitchen-v2` | MiniGrid, 9x9, 400 steps | Three locked doors split the kitchen from the outside. Do the 3 chores, which are floor tiles at random positions, then touch the target square at (1,1). | `a`,`b`,`c` chores, `d` target | `k1.txt` |
| `Traffic-v0` | MiniGrid 1-D corridor, 100 steps | Cross an intersection with a stochastic light that the agent may not be able to see. | `c`,`d`,`t` | `f1.txt` |
| `Colour-v0` | Safety Gym point robot (MuJoCo), 2000 steps | Pillars get random colours. A mission string says which colour to touch first, then to enter the portal, without touching the others. | `a`,`b`,`c` pillars, `d` hazard/portal | `p1.txt` |
| gold mining | Tabular toy problem | See [toy_experiments/README.md](toy_experiments/README.md) | n/a | n/a |

- The registered IDs and their RM files are in [src/utils/env.py](src/utils/env.py) (`key2rmfiles`).
- Adding your own environment means registering it, adding a `letter_types` list and a `get_events()` method, and writing an
  RM file in `src/envs/rm_files/`.
- The RM files use the syntax `(from, to, 'boolean formula', ConstantRewardFunction(r))`. The first line is the initial
  state and `[N]` is the terminal state.

## 3. Algorithms (`--rm-update-algo`)

| Value | Meaning |
|---|---|
| `oracle` | Ground-truth labels. This is the upper bound. |
| `no_rm` | No RM information, so it is plain recurrent PPO. |
| `naive` | Thresholds the detector's predictions of the propositions and steps the RM with them. |
| `ibu` | Independent Belief Updating: a belief over RM states, updated from per-step proposition probabilities. |
| `tdm` | Temporal Dependency Modelling: the detector predicts the RM state belief from the whole history. This is the paper's best method and the default. |

Implementations are in [src/rm_wrappers.py](src/rm_wrappers.py). Detector architectures are in
[src/detector_model.py](src/detector_model.py).

## 4. Training an agent

```bash
python scripts/train_agent.py --env Kitchen-v2 --rm-update-algo tdm --seed 1 \
       --frames 2000000 --procs 8 --hidden-size 64
```

Main flags (defaults in brackets):

- **General:**
  - `--algo` [ppo] picks the learning algorithm: `ppo`, or `sac` for SAC + SimbaV2 (see 4.1). The defaults below are
    PPO's.
  - `--env` is required.
  - `--seed` [1].
  - `--frames` [5e7].
  - `--procs` [1], the number of parallel environments.
  - `--frames-per-proc` [100], the rollout length.
  - `--log-interval` [1].
  - `--save-interval` [20], measured in updates, so a checkpoint is written every 20 x `procs` x `frames-per-proc` frames.
  - `--wandb` turns on Weights & Biases (off by default).
- **PPO:** `--lr 3e-4`, `--discount 0.99`, `--gae-lambda 0.95`, `--entropy-coef 0.001`, `--epochs 4`, `--batch-size 100`,
  `--clip-eps 0.2`.
- **Memory:**
  - `--recurrence` [4] adds an LSTM to the policy. Any value above 1 does this.
  - `--detector-recurrence` [4] adds an LSTM to the detector.
  - `tdm` requires the recurrent detector, and so do `naive`/`ibu` on MiniGrid.
- **Detector:** `--detector-epochs`, `--detector-batch-size`, `--detector-lr`.
- **Resume or warm start:**
  - `--load-model <dir>` loads a checkpoint from a directory that contains `status.pt`.
  - Re-running the same command also resumes automatically, because it finds `status.pt` in the run directory.

Examples:

```bash
# Colour-v0, with 4 parallel envs
python scripts/train_agent.py --env Colour-v0 --rm-update-algo tdm --procs 4 --frames 5000000

# Baselines for comparison
python scripts/train_agent.py --env Kitchen-v2 --rm-update-algo oracle --hidden-size 64
python scripts/train_agent.py --env Kitchen-v2 --rm-update-algo no_rm  --hidden-size 64
python scripts/train_agent.py --env Kitchen-v2 --rm-update-algo naive  --hidden-size 64

# Several seeds
for s in 1 2 3; do python scripts/train_agent.py --env Kitchen-v2 --rm-update-algo tdm --seed $s --hidden-size 64 & done; wait
```

**Where output goes:** `storage/<algo>[-r]-<env>-seed<N>/train/`. The `-r` appears when `--recurrence > 1`, which is the
default.

- `log.csv`: metrics, one row per update.
- `log.txt`: the same as the console output.
- `status.pt`: policy, detector and optimizers.
- `config.pickle`: the arguments.

**Reading the console line**, for example `rR:μσmM ... | F:μσmM ... | dLtest ... | dtop1Acctest ...`:

- `rR` is the reward statistics (mean, std, min, max) over finished episodes.
- `F` is the episode length.
- `H`, `V`, `pL`, `vL` are entropy, value, policy loss and value loss.
- `dLtrain`, `dLtest` and `dtop1Acc*` are the detector's loss and top-1 accuracy.
- Return columns are placeholders (0 reward, 1 frame) until an episode finishes in that update. In `Colour-v0` the episodes
  are 2000 steps long, so you'll see zeros for a while.

**Resuming vs. fresh runs:** if `storage/<run>/train/status.pt` already exists for a given env, algo and seed, training
continues from it and appends to its `log.csv`. Use another seed or delete the folder for a fresh run.

### 4.1 Off-policy alternative: SAC + SimbaV2

`--algo sac` makes [scripts/train_agent.py](scripts/train_agent.py) train Soft Actor-Critic with the SimbaV2 networks
([Lee et al., ICML 2025](https://proceedings.mlr.press/v267/lee25u.html)) instead of PPO. It takes the same environments,
`--rm-update-algo` values and detectors, and writes the same kind of logs.

```bash
python scripts/train_agent.py --algo sac --env Kitchen-v2 --rm-update-algo tdm --seed 1 --frames 1000000
python scripts/train_agent.py --algo sac --env Colour-v0 --rm-update-algo tdm --procs 4
```

`--frames`, `--batch-size`, `--lr`, `--discount` and `--save-interval` get SAC defaults when you don't set them (listed
below). The PPO-only flags (`--epochs`, `--clip-eps`, `--gae-lambda`, `--entropy-coef`, `--value-loss-coef`,
`--recurrence`, `--hidden-size`, `--optim-*`) are ignored.

How it differs from PPO:

- Every step goes into a replay buffer (kept on the GPU). After 5000 frames of random actions, the agent makes
  `--utd-ratio` gradient updates per frame, between environment steps.
- The detector is still trained on the latest `--frames-per-proc` x `--procs` frames, as in PPO. The RM belief stored in
  the buffer is the one computed at the time of the step. It is not recomputed when the detector improves.
- The policy has no memory, so there is no `--recurrence` flag. It sees the RM belief (`tdm`/`naive`/`ibu`) or the true
  RM state (`oracle`) as input, like PPO. With `no_rm` the agent therefore has no memory at all.
- Kitchen-v2 and Traffic-v0 have discrete actions and use SAC-Discrete. This extension is not part of the paper. Colour-v0 uses the paper's
  continuous tanh-Gaussian policy.
- Transitions cut by the time limit are not stored. `ParallelEnv` resets the env before the final observation can be
  kept, so these transitions can't be bootstrapped correctly.

SAC flags (defaults in brackets, taken from Table 3 of the paper):

- **SAC:**
  - `--batch-size` [256], `--utd-ratio` [2], `--buffer-size` [1e6], `--learning-starts` [5000].
  - `--lr` [1e-4] decays linearly to `--lr-end` [3e-5] over the run.
  - `--tau` [0.005], `--temperature-init` [0.01].
  - `--target-entropy-scale` [0.5] gives a target entropy of -0.5|A| for continuous actions and 0.5 log|A| for discrete
    ones.
  - `--cdq` / `--no-cdq` turns two critics with clipped double Q on or off. It is on by default because the RM tasks
    terminate.
- **Discount:** by default `--discount` uses the paper's heuristic clip((T/5 - 1)/(T/5), 0.95, 0.995), where T is the
  episode length. That gives 0.9875 for Kitchen-v2, 0.95 for Traffic-v0 and 0.995 for Colour-v0. Pass `--discount 0.99`
  to match PPO.
- **Networks:**
  - Actor: `--actor-hidden-size` [128], `--actor-blocks` [1].
  - Critic: `--critic-hidden-size` [512], `--critic-blocks` [2], `--num-bins` [101].
  - Inputs and outputs: `--g-max` [5] bounds the critic's support, and `--c-shift` [3] is the input shift constant.
- **Run length:** `--frames` [1e6].
- **Rollouts:** `--frames-per-proc` [100] only sets how often the detector trains and a log line is written.
  `--save-interval` [100] counts those rollouts.

**Speed:** one update takes about 11 ms on the RTX 1000 laptop GPU while other jobs are running, so the defaults give
around 40 frames per second. `--utd-ratio 1`, `--no-cdq` or a smaller `--critic-hidden-size` each roughly halve that cost.
The buffer for Kitchen-v2 uses about 1.3 GB of GPU memory, and `--buffer-size` lowers it.

**Output:** runs go to `storage/sac-<algo>-<env>-seed<N>/train/`. `log.csv` keeps the PPO columns where they mean the
same thing and adds `temperature`, `q_value`, `actor_loss`, `critic_loss` and `reward_scale`. Because the run name starts
with `sac-<algo>`, PPO and SAC can share one plot, for example
`python scripts/plot_results.py --env Kitchen-v2 --algos tdm sac-tdm`.

**Resuming:** re-running the same command continues from `status.pt`. The replay buffer is not saved, so it is refilled
with the loaded policy before updates start again. The visualization scripts (section 7) only load PPO models for now.

## 5. Pretraining the detector offline (optional)

Instead of learning the detector only from RL interaction data, you can collect labelled data and pretrain it.

```bash
# 1. Collect episodes with a semi-random policy and ground-truth labels.
#    It writes ./collect_<env>.pt in the current directory, so rename it each time.
python scripts/collect_offline_data.py --env Kitchen-v2 --episodes 500 --seed 1
mv collect_Kitchen-v2.pt data_train.pt
python scripts/collect_offline_data.py --env Kitchen-v2 --episodes 100 --seed 2
mv collect_Kitchen-v2.pt data_test.pt

# 2. Pretrain. The model name MUST be "<algo>-<env>-pretrained" for train_agent.py to find it.
python scripts/pretrain_detector.py --env Kitchen-v2 --rm-update-algo tdm --use-mem \
       --train-data-path data_train.pt --test-data-path data_test.pt \
       --epochs 60 --batch-size 50 --model tdm-Kitchen-v2-pretrained

# 3. Train the agent. It auto-loads storage/tdm-Kitchen-v2-pretrained/status.pt if present
python scripts/train_agent.py --env Kitchen-v2 --rm-update-algo tdm --hidden-size 64
```

- Pass `--use-mem` so the pretrained detector matches the recurrent one `train_agent.py` builds by default.
- `--batch-size` must be no larger than the number of episodes.
- Pretraining also saves the lowest-test-loss weights to `storage/<model>/best/status.pt`. `train_agent.py` loads only the
  final `status.pt`, so copy `best/status.pt` over it if you prefer the best checkpoint.
- Steps 1 and 2 were tested on small samples. Step 3 picking up the pretrained weights was not tested.

## 6. Plotting results

[scripts/plot_results.py](scripts/plot_results.py) reads `storage/*/train/log.csv`, groups runs that differ only by seed, and
plots mean +/- std.

```bash
python scripts/plot_results.py --env Kitchen-v2 --metric rreturn_mean --smooth 20          # opens a window
python scripts/plot_results.py --env Kitchen-v2 --metric detector_top1_accuracy_test --out acc.png
python scripts/plot_results.py --env Colour-v0 --algos tdm oracle no_rm --out colour.png
```

Useful `--metric` values:

- `rreturn_mean` (the reward statistic the console prints) and `return_mean`.
- `num_frames_mean` (episode length).
- `detector_loss_test`, `detector_top1_accuracy_test`.
- `entropy`, `value_loss`.

The x-axis is frames. Notes:

- `--smooth` is a window in logged updates, not frames. With 1 process and 100 frames per update, 20 updates is only 2000
  frames, so the curve is spiky. Try `--smooth 300` or more for long runs.
- Updates where no episode finished log placeholder zeros (see section 4), which pulls the mean towards 0.
- A run that was interrupted and resumed leaves repeated header rows and replayed frame counts in `log.csv`. The plot script
  drops the header rows and keeps the latest row for each frame count.

For other analysis you can read `log.csv` directly with `pandas.read_csv`. For live curves, add
`--wandb` to training.

## 7. Visualizing a trained agent

**MiniGrid** (`Kitchen-v2`, `Traffic-v0`):

```bash
python scripts/minigrid_visualize.py --env Kitchen-v2 --rm-update-algo tdm \
       --model storage/tdm-r-Kitchen-v2-seed1/train --use-mem --use-mem-detector --episodes 5
```

- `--pause` sets the delay between actions.
- `--no-render` runs headless and just prints returns.
- `--gif name` saves a GIF, but it needs `pip install array2gif`.
- The window shows the agent's RM-state belief.

**MuJoCo** (`Colour-v0`):

```bash
python scripts/mujoco_visualize.py --env Colour-v0 --rm-update-algo tdm \
       --model storage/tdm-r-Colour-v0-seed1/train --use-mem --use-mem-detector
```

This is untested. The window needs the GLEW preload that `setup.sh` already sets.

**Two caveats in `minigrid_visualize.py`:**

1. It hard-codes `hidden_size=64`, while `train_agent.py` defaults to 128. A mismatch gives a "size mismatch" error. Train
   with `--hidden-size 64`, as in the examples above, or change the script.
2. On a GPU machine it crashes with "tensors on cuda:0 and cpu" in `Agent.analyze_feedback`. Run it with
   `CUDA_VISIBLE_DEVICES=""` to use the CPU, which works. Putting `masks` on `self.device` in
   [src/utils/agent.py](src/utils/agent.py) would fix it properly.

## 8. Playing the environments yourself

```bash
python scripts/minigrid_manual_agent.py --env-id Kitchen-v2    # keyboard control
python scripts/mujoco_manual_agent.py  --env Colour-v0          # keyboard control in the MuJoCo viewer
```

These are untested (they need a window and keyboard input). The point of them is to see the mission and the event letters
first-hand.

## 9. Toy gold-mining experiment

```bash
PYTHONPATH=toy_experiments python toy_experiments/run_gold_mining_exps.py
```

This is untested, and the script looks unfinished. It uses a variable `i` with no loop around it, and it writes into
`toy_experiments/data/policy_learning/` and `toy_experiments/data/internal_rewards/`, which you'd need to create. Wrap the two
blocks in `for i in range(N):`. The algorithms are in [toy_experiments/gold_mining_algos.py](toy_experiments/gold_mining_algos.py)
(`QLIndependentBelief` and others), and `reward_type="predicted"` trains without ground-truth rewards.

## 10. Typical workflow

1. Play the environment manually (section 8) to understand the task.
2. Train `oracle` and `no_rm` as upper and lower baselines (section 4).
3. Train `tdm`, `naive` and `ibu` over 3 or more seeds (section 4).
4. Optionally pretrain detectors (section 5).
5. Plot with `plot_results.py` (section 6) and replay a checkpoint (section 7).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `GLEW initialization error: Missing GL version` | Source `setup.sh` so the GLEW/GL `LD_PRELOAD` is set. |
| `mujoco_py` fails to compile with Cython errors | `pip install "Cython<3"` |
| `patchelf` not found while building mujoco_py | `pip install patchelf` |
| `module 'numpy' has no attribute 'complex'` | Already patched in the local `../safety-gym` clone (`np.complex` -> `complex`). Redo this if you re-clone it. |
| `ModuleNotFoundError: No module named 'gym'` | The local `../safety-gym` clone must import `gymnasium` instead. Already patched in `engine.py`, `suite.py` and `random_agent.py`. |
| gymnasium warnings about `Env.seed`, float64 or observation bounds | Harmless. |
