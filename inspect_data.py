import numpy as np
import torch
import matplotlib.pyplot as plt
from .models import PedPred3
from torch.utils.data import DataLoader
from collections import namedtuple
from .config import cfg
from .dataset import FileListDataset, GridFromH5Dataset, SeqDataset
from .grid import Grid

# DATA LOADING
def get_data(*mode, n_in, n_out, local_leak: dict = {}, num_workers = 0, pin_memory =False, drop_last = True, prefetch_factor=2):
    mode = mode or ('train', 'valid')
    dataset, _, subset = cfg.dataset.partition(':')
    resolution = cfg.resolution
    period = cfg.period
    kernel = cfg.kernel
    # nin = cfg.nin
    # nout = cfg.nout
    nin = n_in
    nout = n_out
    batch = 1
    # batch = cfg.batch

    r = 1 / resolution
    r = int(r) if r.is_integer() else None  # don't raise exception until usage

    if dataset == 'alex':
        assert not subset
        grid = Grid((-6, -6), 0, (12 * r, 12 * r), resolution)

        data = {mode:
            DataLoader(
                SeqDataset(
                    GridFromH5Dataset(
                        f'data/alex1_{mode}.h5',
                        grid, period,
                        kernel=kernel,
                        random_rotation=(mode == 'train'),
                    ),
                    nin, nout,
                ),
                batch,
                shuffle=(mode == 'train'),
                generator=torch.default_generator,

            )
            for mode in mode
        }

    elif dataset == 'atc':
        # grids must have shape divisible by 4
        grids = {
            'corridor': Grid(origin=(38.2789, -15.8076), theta=2.5647, shape=(36 * r, 12 * r), resolution=resolution),
            'no_walls': Grid(origin=(-3.0431, 4.3197), theta=2.4128, shape=(16 * r, 12 * r), resolution=resolution),
            'all': Grid(origin=(55.3890, -8.2735), theta=2.6970, shape=(88 * r, 36 * r), resolution=resolution),
        }
        grid = grids[subset]

        data = {mode:
            DataLoader(
                FileListDataset(
                    f'data/sunday_atc_{mode}.lst',
                    lambda file:
                    SeqDataset(
                        GridFromH5Dataset(
                            (file.parent /'ATC'/'Sundays' / file.stem).with_suffix('.h5'),
                            grid, period,
                            kernel=kernel,
                            random_rotation=False,
                            attach_point_data=(mode == 'test'),
                            attach_point_time=(mode == 'test'),
                        ),
                        nin, nout,
                    )
                    ,
                ),
                batch,
                shuffle=(mode == 'train'),
                generator=torch.default_generator,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
                prefetch_factor=prefetch_factor if num_workers > 0 else None
            )
            for mode in mode
        }

    else:
        raise ValueError(f"Invalid {dataset=}.")

    for key in local_leak:
        local_leak[key] = locals()[key]

    data = namedtuple('DataLoaderSet', data)(**data)
    if len(data) == 1: data = data[0]
    return data


# # Load data
# data = get_data('test', n_in=1, n_out=1)
#
# # Accumulators
# feature_values = [[] for _ in range(4)]  # 0=density, 1=vel_x, 2=vel_y, 3=variance
#
# for x_batch, _ in data:
#     x_np = x_batch.squeeze(0).squeeze(0).numpy()  # (4,36,12)
#     for f in range(4):
#         feature_values[f].extend(x_np[f].flatten())
#
# # Convert to numpy arrays
# feature_values = [np.array(vals) for vals in feature_values]
#
# # Print summary stats
# feature_names = ["Density", "Velocity X", "Velocity Y", "Variance"]
# print(f"{'Feature':<15}{'Min':>10}{'Max':>10}{'Mean':>10}{'Std':>10}")
# print("-" * 55)
# for name, vals in zip(feature_names, feature_values):
#     print(f"{name:<15}{np.min(vals):>10.4f}{np.max(vals):>10.4f}{np.mean(vals):>10.4f}{np.std(vals):>10.4f}")
#
# # Plot histograms
# fig, axes = plt.subplots(2, 2, figsize=(10, 8))
# axes = axes.flatten()
# for idx, (name, vals) in enumerate(zip(feature_names, feature_values)):
#     axes[idx].hist(vals, bins=50, color='skyblue', edgecolor='black')
#     axes[idx].set_title(f"{name} Distribution")
#     axes[idx].set_xlabel("Value")
#     axes[idx].set_ylabel("Frequency")
# plt.tight_layout()
# plt.show()
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

# ==== CONFIG ====
SAVE_DIR = "./EnKF_results/FeatureStats"
os.makedirs(SAVE_DIR, exist_ok=True)

DENS_THR = 1e-6          # density threshold marking "active" cells
BINS = 60                 # histogram bins

# Load your test split exactly like EnKF uses
data = get_data('test2', n_in=1, n_out=1)

# Accumulators (active cells only)
vals_active = { 'density': [], 'vx': [], 'vy': [], 'var': [] }
active_fraction_per_step = []

num_steps = 0
for x_batch, _ in data:
    # x: (1, 1, 4, H, W) -> (4, H, W)
    x = x_batch.squeeze(0).squeeze(0).numpy()
    density = x[0]
    vx = x[1]
    vy = x[2]
    var = x[3]

    mask = density > DENS_THR
    active_fraction = mask.mean()
    active_fraction_per_step.append(active_fraction)

    if np.any(mask):
        vals_active['density'].append(density[mask].ravel())
        vals_active['vx'].append(vx[mask].ravel())
        vals_active['vy'].append(vy[mask].ravel())
        vals_active['var'].append(var[mask].ravel())

    num_steps += 1

# Concatenate across steps
for k in vals_active:
    if len(vals_active[k]) > 0:
        vals_active[k] = np.concatenate(vals_active[k], axis=0)
    else:
        vals_active[k] = np.array([])

feature_names = ["density", "vx", "vy", "var"]

def summarize(arr):
    if arr.size == 0:
        return dict(min=np.nan, max=np.nan, mean=np.nan, std=np.nan, p5=np.nan, p95=np.nan, iqr=np.nan, mad=np.nan)
    p5, p95 = np.percentile(arr, [5, 95])
    iqr = np.percentile(arr, 75) - np.percentile(arr, 25)
    # robust std approx using MAD
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    robust_std = 1.4826 * mad  # ~N(0,1) consistency
    return dict(
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
        p5=float(p5),
        p95=float(p95),
        iqr=float(iqr),
        mad=float(robust_std),
    )

print("Active-cell (density > {:.1e}) stats over the test set:".format(DENS_THR))
print(f"{'Feature':<10} {'min':>8} {'max':>8} {'mean':>8} {'std':>8} {'p5':>8} {'p95':>8} {'iqr':>8} {'robust_std':>12}")
print("-" * 90)

stats = {}
for name in feature_names:
    s = summarize(vals_active[name])
    stats[name] = s
    print(f"{name:<10} {s['min']:>8.4f} {s['max']:>8.4f} {s['mean']:>8.4f} {s['std']:>8.4f} "
          f"{s['p5']:>8.4f} {s['p95']:>8.4f} {s['iqr']:>8.4f} {s['mad']:>12.4f}")

active_fraction_per_step = np.array(active_fraction_per_step, dtype=float)
print("\nActive cell fraction per step: mean={:.4f}, std={:.4f}, min={:.4f}, max={:.4f}".format(
    np.nanmean(active_fraction_per_step),
    np.nanstd(active_fraction_per_step),
    np.nanmin(active_fraction_per_step),
    np.nanmax(active_fraction_per_step),
))

# ---- Histograms for active values ----
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes = axes.ravel()
for ax, name in zip(axes, feature_names):
    arr = vals_active[name]
    if arr.size > 0:
        ax.hist(arr, bins=BINS, edgecolor='black')
        ax.set_title(f"{name} (active cells)")
        ax.set_xlabel("value")
        ax.set_ylabel("count")
    else:
        ax.text(0.5, 0.5, "No active cells", ha='center', va='center')
        ax.set_title(f"{name} (active cells)")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "active_feature_histograms.png"))
plt.close()

# ================= Suggested noise scales =================
# Heuristics:
# - Use robust_std (MAD-based) on active cells to avoid being dominated by outliers.
# - Observation noise std ~ 0.25 * robust_std  (sensor is usually more accurate than the natural variability)
# - Process noise std     ~ 0.15 * robust_std  (model error per step; smaller than obs)
# Clamp to small floors to avoid zeros.
def suggest_std(robust_std, factor, floor=1e-6):
    if np.isnan(robust_std) or robust_std == 0:
        return floor
    return max(factor * robust_std, floor)

obs_factor = 0.25
proc_factor = 0.15

obs_std = (
    suggest_std(stats['density']['mad'], obs_factor),
    suggest_std(stats['vx']['mad'],      obs_factor),
    suggest_std(stats['vy']['mad'],      obs_factor),
    suggest_std(stats['var']['mad'],     obs_factor),
)

proc_std = (
    suggest_std(stats['density']['mad'], proc_factor),
    suggest_std(stats['vx']['mad'],      proc_factor),
    suggest_std(stats['vy']['mad'],      proc_factor),
    suggest_std(stats['var']['mad'],     proc_factor),
)

print("\nSuggested per-feature stds (based on active-cell robust std):")
print("  Observation std (density, vx, vy, var): ({:.3e}, {:.3e}, {:.3e}, {:.3e})".format(*obs_std))
print("  Process std     (density, vx, vy, var): ({:.3e}, {:.3e}, {:.3e}, {:.3e})".format(*proc_std))

# Also output a few nearby options (×0.5, ×1, ×2) for quick grid search seeds
scale_set = [0.5, 1.0, 2.0]
obs_grid = [tuple(s*o for o in obs_std) for s in scale_set]
proc_grid = [tuple(s*p for p in proc_std) for s in scale_set]

print("\nSeed grids you can feed into your EnKF search:")
print("  OBS grid:")
for g in obs_grid:
    print("   ", "({:.3e}, {:.3e}, {:.3e}, {:.3e})".format(*g))
print("  PROC grid:")
for g in proc_grid:
    print("   ", "({:.3e}, {:.3e}, {:.3e}, {:.3e})".format(*g))

# Save grids for later import (optional)
with open(os.path.join(SAVE_DIR, "suggested_noises.txt"), "w") as f:
    f.write("OBS seed grid:\n")
    for g in obs_grid:
        f.write("({:.6e}, {:.6e}, {:.6e}, {:.6e})\n".format(*g))
    f.write("\nPROC seed grid:\n")
    for g in proc_grid:
        f.write("({:.6e}, {:.6e}, {:.6e}, {:.6e})\n".format(*g))
