import torch
import h5py
import numpy as np
from tqdm import tqdm
from ..config import cfg
from ..dataset import GridFromH5Dataset, SeqDataset
# from ..flow_interpolation.models_gpytorch import ConstrGPModel, HYPS, optimize_traj
from ..grid import Grid, GridData
import os


directory = 'Final planning results/planning results beta=0.0001-all/planning_bold-yeti_best_1000_E_A8.h5'
def get_dataset(step=1, **kwargs):
	resolution = cfg.resolution
	r = 1/resolution
	grid = Grid(origin=( 38.2789,-15.8076), theta=2.5647, shape=(36*r,12*r), resolution=resolution)
	period = cfg.period
	dataset = GridFromH5Dataset(
		'data/ATC/atc-20130707.h5',
		grid, period*step,
		kernel=cfg.kernel,
		random_rotation=False,
		normalise_period=period,
		**kwargs
	)
	return dataset

def compute_collisions(h5_filename, collision_threshold=0.5):
    origin = torch.tensor([38.2789, -15.8076])
    # Open H5 file directly
    with h5py.File(h5_filename, 'r') as file:
        expected_costs = {}
        actual_costs = {}
        paths = {}

        for method in file:
            group = file[method]
            expected_costs[method] = group.attrs['expected_cost']
            actual_costs[method] = group.attrs['actual_cost']
            ps = torch.as_tensor(group['path_ps'])
            ts = torch.as_tensor(group['path_ts'])
            print(f"Method: {method}, ps shape: {ps.shape}")  # <--- Print ps
            # print(ps)
            paths[method] = (ps, ts)

    method_order = ['static', 'once', 'online', 'linear', 'actual']
    paths = {method: paths[method] for method in method_order if method in paths}

    # Get dataset (for actual pedestrian data)
    dataset = get_dataset(step=1, attach_point_data=True)

    start_time_idx = int(10+80)  # As seen in your plotting functions

    # Store collision counts per method
    collisions = {}

    for method, (ps, ts) in paths.items():
        num_collisions = 0

        for position, t in zip(ps, ts):
            dataset_idx = start_time_idx + int(t.item())

            if dataset_idx >= len(dataset):
                continue

            grid_frame = dataset[dataset_idx][0].data if hasattr(dataset[dataset_idx][0], 'data') else dataset[dataset_idx][0]



            # print(position)
            pos = (position - origin)
            x, y = pos
            # print(pos)
            x_int, y_int = int(round(x.item())), int(round(y.item()))
            H, W = grid_frame.shape

            x_int = max(0, min(H - 1, x_int))
            y_int = max(0, min(W - 1, y_int))

            # Access the grid cell corresponding to the position
            cell_value = grid_frame[x_int, y_int]

            # If the density or value of the cell exceeds the threshold, count it as a collision
            if cell_value > collision_threshold:
                num_collisions += 1

        collisions[method] = num_collisions

    return collisions

# Example usage:

# Example usage:
if __name__ == "__main__":
    # Set directory containing H5 files
    directory = 'Final planning results/planning results beta=0.0001-all'

    # List all files in the directory
    files = os.listdir(directory)

    # Initialize a dictionary to accumulate collision counts per method
    total_collisions = {method: 0 for method in ['static', 'once', 'online', 'linear', 'actual']}
    file_count = 0

    for file_name in files:
        file_path = os.path.join(directory, file_name)

        # Check if the file is an HDF5 file
        if file_name.endswith('.h5'):
            collisions = compute_collisions(file_path)  # Pass the full path
            print(f"Collision counts for file {file_name}:")
            for method, count in collisions.items():
                print(f"{method}: {count}")
                total_collisions[method] += count
            file_count += 1

    # Compute and print the average collision counts per method
    if file_count > 0:
        print("\nAverage collision counts per method over all files:")
        for method, total_count in total_collisions.items():
            avg_count = total_count / file_count
            print(f"{method}: {avg_count:.2f}")
    else:
        print("No files processed.")
