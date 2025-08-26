from collections import defaultdict, namedtuple
from torch.utils.data import DataLoader
import itertools
import matplotlib.pyplot as plt
import time
import numpy as np
from itertools import islice
import torch
from torch.func import jacrev
# Local
from .config_jac import cfg
from .dataset import FileListDataset, GridFromH5Dataset, SeqDataset
from .grid import Grid
from .models import *
from .test import get_my_model
import gc
import psutil
import torch

def get_data(*mode, local_leak: dict = {}, num_workers = 0, pin_memory =False, drop_last = True, prefetch_factor=2):
    mode = mode or ('train', 'valid')
    dataset, _, subset = cfg.dataset.partition(':')
    resolution = cfg.resolution
    period = cfg.period
    kernel = cfg.kernel
    nin = 5
    # cfg.nin
    nout = 5
    # cfg.nout
    batch = cfg.batch

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
                            (file.parent / 'ATC' /'Sundays' / file.stem).with_suffix('.h5'),
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
                num_workers= num_workers,
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


class Jacobian:
    def __init__(self, model, n_in):

        self.model = model.eval()  # Use the trained model in evaluation mode
        self.n_in = cfg.nin
    def diag_jacobian(self, f_flat, x_flat):
        J = jacrev(f_flat)(x_flat)  # J shape: [N, N]

        # Extract main diagonal
        # jacobian_diag = torch.diagonal(J)

        # Reshape to original input
        return J


    def compute(self, input_sample):

        input_sample = input_sample.requires_grad_(True)
        horizon = input_sample.shape[1]
        with torch.no_grad():
            input_sample = input_sample.detach().requires_grad_()

        def single_pass(x):
            output = self.model(x, horizon=input_sample.shape[1])
            return output
        start = time.time()
        jacobian = torch.autograd.functional.jacobian(single_pass, input_sample, vectorize=True )
        end = time.time()
        print(f"full jacobian computational time= ", end-start)
        print(f"Jacobian shape: {jacobian.shape}")
        return jacobian

    def banded_jacobian(self, input_sample, band_width=3):
        input_sample = input_sample.requires_grad_(True)
        horizon = input_sample.shape[1]

        def single_pass(x):
            return self.model(x, horizon=horizon)

        output = single_pass(input_sample)
        batch_size = input_sample.shape[0]
        input_flat = input_sample.view(batch_size, -1)  # shape: [batch, N]
        output_flat = output.view(batch_size, -1)  # shape: [batch, N]
        N = input_flat.shape[1]
        diag_entries = {offset: [] for offset in range(-band_width, band_width + 1)}

        for batch_idx in range(batch_size):
            for i in range(N):
                grad_output = torch.zeros_like(output_flat[batch_idx])
                grad_output[i] = 1.0

                grad_input = torch.autograd.grad(
                    outputs=output_flat[batch_idx],
                    inputs=input_flat[batch_idx],
                    grad_outputs=grad_output,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True
                )[0]

                if grad_input is None:
                    continue

                # Collect only band entries
                for offset in range(-band_width, band_width + 1):
                    j = i + offset
                    if 0 <= j < N:
                        val = grad_input[j].item()
                        if abs(val) > 1e-12:
                            diag_entries[offset].append((batch_idx, i, j, val))

        return diag_entries

    def get_diagonal_jacobian_entries(self, x, max_offset=3):
        self.model.eval()
        x = x.requires_grad_(True)
        output = self.model(x, horizon=input_sample.shape[1])

        output_flat = output.reshape(-1)
        input_flat = x.reshape(-1)
        N = input_flat.shape[0]

        diag_entries = {offset: [] for offset in range(-max_offset, max_offset + 1)}

        for out_idx in range(N):
            if x.grad is not None:
                x.grad.zero_()

            grad_output = torch.zeros_like(output_flat)
            grad_output[out_idx] = 1.0

            grad = torch.autograd.grad(outputs=output_flat, inputs=x, grad_outputs=grad_output, retain_graph=True)[0]
            grad_flat = grad.reshape(-1)

            for offset in range(-max_offset, max_offset + 1):
                in_idx = out_idx - offset
                if 0 <= in_idx < N:
                    val = grad_flat[in_idx].item()
                    if abs(val) > 1e-8:
                        diag_entries[offset].append((out_idx, in_idx, val))

        return diag_entries

def run(input_sample, model, jacobian_calculator):

    pred = model(input_sample, horizon=input_sample.shape[1])
    # jacobian = jacobian_calculator.compute(input_sample)
    jacobian = None
    input_sample = input_sample.detach().requires_grad_(True)
    x_flat = input_sample.view(-1)

    def f_flat(x_flat):
        x_shaped = x_flat.view_as(input_sample)
        output = model(x_shaped, horizon=input_sample.shape[1])
        return output.view(-1)

    print('start computing jac')
    start = time.time()
    diag_jacobian = jacobian_calculator.diag_jacobian(f_flat, x_flat)
    end = time.time()
    print('compute whole thing using jrec', end - start)

    return diag_jacobian, jacobian, pred

# if __name__ == '__main__':
#     model_name = 'apt-oyster_best'
#     data = get_data('test')
#     # Indices = [0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48]
#     Indices = [0, 3]
#     max_index = max(Indices)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = PedPred3()
#     checkpoint = torch.load('apt-ibex_train_model_28D.pth', map_location=device)
#     model.load_state_dict(checkpoint['model'])
#     print("Switching to evaluation mode...")
#     model.eval()
#
#     jacobian_calculator = Jacobian(model, n_in=cfg.nin)
#     for i, (input_sample, target) in islice(enumerate(data), max_index + 1):
#         if i in Indices:
#             # input_sample = input_sample[0]
#             # print(input_sample[0][0].shape)
#             diag_A, A, pred = run(input_sample[0][0].view(1, 1,4,36,12), model, jacobian_calculator)
#             diag_A, A, pred = run(input_sample, model, jacobian_calculator)
#             # torch.save(diag_A, f"RAS/Iterative_jac/A_pedpred3_t{i}.pt")
#             del diag_A, A, pred
#             torch.cuda.empty_cache()
#             import gc
#             gc.collect()
#             # Extract diagonals: main (0), 1 to 3 above and below
#             # band_diags = {}
#             # for offset in range(-3, 4):  # from -3 (lower) to +3 (upper)
#             #     diag = torch.diagonal(diag_A, offset=offset)
#             #     band_diags[offset] = diag
#             #     print(f"Diagonal offset {offset}: length={diag.numel()}, non-zeros={torch.count_nonzero(diag).item()}")

def get_memory_usage():
    return psutil.Process().memory_info().rss / (1024 ** 2)  # in MB

def benchmark(input_sample, label, model, jacobian_calculator, runs=2):
    times = []
    mem_usages = []

    for _ in range(runs):
        gc.collect()
        torch.cuda.empty_cache()
        mem_before = get_memory_usage()
        start = time.time()

        diag_A, _, _ = run(input_sample, model, jacobian_calculator)

        end = time.time()
        mem_after = get_memory_usage()
        times.append(end - start)
        mem_usages.append(mem_after - mem_before)

        del diag_A

    print(f"\n--- Benchmark: {label} ---")
    print(f"Average time over {runs} runs: {np.mean(times):.4f} s")
    print(f"Average memory usage: {np.mean(mem_usages):.2f} MB")
    return times, mem_usages

if __name__ == '__main__':
    model_name = 'apt-oyster_best'
    data = get_data('test')
    Indices = [0,1,2]  # Use a single batch for fair benchmarking
    max_index = max(Indices)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PedPred3()
    checkpoint = torch.load('apt-ibex_train_model_28D.pth', map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    jacobian_calculator = Jacobian(model, n_in=cfg.nin)

    for i, (input_sample, target) in islice(enumerate(data), max_index + 1):
        if i in Indices:
            # small_input = input_sample[0][0].view(1,1, 4, 36, 12)
            large_input = input_sample  # [10, 4, 36, 12]

            print(f"Running 100 runs for timing and memory comparison...\n")

            # benchmark(small_input, "SMALL input (1x4x36x12)", model, jacobian_calculator)
            benchmark(large_input, "LARGE input (10x4x36x12)", model, jacobian_calculator)
