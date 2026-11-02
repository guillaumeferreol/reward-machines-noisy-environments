"""
Plot learning curves from the training logs written by train_agent.py.

Each run lives in storage/<algo>[-r]-<env>-seed<N>/train/log.csv. Runs that differ only by seed
are grouped together and plotted as mean +- std across seeds.

Example:
    python scripts/plot_results.py --env Kitchen-v2 --metric return_mean --smooth 20 --out kitchen.png
    python scripts/plot_results.py --env Kitchen-v2 --metric detector_top1_accuracy_test
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import matplotlib
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True, help="environment name, e.g. Kitchen-v2")
parser.add_argument("--storage", default="storage", help="directory containing the runs")
parser.add_argument("--metric", default="return_mean",
                    help="column of log.csv to plot (e.g. return_mean, rreturn_mean, num_frames_mean, "
                         "detector_loss_test, detector_top1_accuracy_test, entropy)")
parser.add_argument("--smooth", type=int, default=10, help="rolling-window size, in logged updates")
parser.add_argument("--algos", nargs="*", default=None, help="only plot these algos (e.g. tdm naive)")
parser.add_argument("--out", default=None, help="save the figure here instead of opening a window")
args = parser.parse_args()

if args.out:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Run directories are named "{algo}{-r if recurrent}-{env}-seed{N}"
pattern = re.compile(rf"^(?P<algo>.+?)(?P<rec>-r)?-{re.escape(args.env)}-seed(?P<seed>\d+)$")

groups = defaultdict(list)
for log_path in sorted(glob.glob(os.path.join(args.storage, "*", "train", "log.csv"))):
    run_name = os.path.basename(os.path.dirname(os.path.dirname(log_path)))
    match = pattern.match(run_name)
    if match is None:
        continue
    algo = match["algo"] + (" (recurrent)" if match["rec"] else "")
    if args.algos and match["algo"] not in args.algos:
        continue
    if os.path.getsize(log_path) == 0:
        continue
    df = pd.read_csv(log_path)
    # train_agent.py appends to log.csv, so a restarted run repeats the header row...
    df = df[df["update"] != "update"].apply(pd.to_numeric)
    # ...and a run resumed from its last checkpoint replays the updates logged after that checkpoint.
    # Keep the latest occurrence of each frame count (the resumed run overwrites the replayed segment).
    df = df.drop_duplicates("frames", keep="last").sort_values("frames")
    groups[algo].append(df.set_index("frames")[args.metric])

if not groups:
    raise SystemExit(f"No runs found for env {args.env} in {args.storage}/")

fig, ax = plt.subplots(figsize=(7, 4.5))
for algo, series in sorted(groups.items()):
    # Align seeds on a common frame grid
    frames = np.unique(np.concatenate([s.index.values for s in series]))
    mat = np.stack([s.reindex(frames).interpolate(limit_direction="both").values for s in series])
    mat = pd.DataFrame(mat.T).rolling(args.smooth, min_periods=1).mean().values.T
    mean, std = mat.mean(0), mat.std(0)
    ax.plot(frames, mean, label=f"{algo} (n={len(series)})")
    ax.fill_between(frames, mean - std, mean + std, alpha=0.2)

ax.set_xlabel("frames")
ax.set_ylabel(args.metric)
ax.set_title(args.env)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()

if args.out:
    fig.savefig(args.out, dpi=150)
    print("Saved", args.out)
else:
    plt.show()
