# Standard Library
from collections import defaultdict, namedtuple
import sys
import time
from typing import Callable
import os
# PyPi
import matplotlib as mpl
from torch import optim
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# Local
from .config import cfg
from .dataset import FileListDataset, GridFromH5Dataset, SeqDataset
from .grid import Grid
from .metrics import Metrics
from .models import *
from .saving_loading import TrainingState
from .tools import CatchSignal
from .tools import tqdm
from .tools.torch import cuda_context

loss_metric = cfg.loss
loss_fun = Metrics[loss_metric]


def get_data(*mode, local_leak: dict = {}, num_workers = 0, pin_memory =True, drop_last = True, prefetch_factor=2):
    mode = mode or ('train', 'valid')
    dataset, _, subset = cfg.dataset.partition(':')
    resolution = cfg.resolution
    period = cfg.period
    kernel = cfg.kernel
    nin = cfg.nin
    nout = cfg.nout
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
                            (file.parent /'ATC' /'Sundays' / file.stem).with_suffix('.h5'),
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

def get_trainstate(file_glob):
    model = PedPred2()

    # # precipitation now-casting config
    # optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.999), amsgrad=True)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    # pytorch defaults
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), amsgrad=True)
    # todo: try adagrad maybe
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10, cooldown=0, threshold=0)

    # load state
    state = TrainingState(file_glob, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)
    return state


class EDException(Exception):
    """Early Death"""

    def __init__(self, name=None, age=None, cause=None):
        self.name = name
        self.age = age
        self.cause = cause

    def __str__(self):
        lines = ["R I P"]
        if self.name:   lines += [self.name]
        if self.age:    lines += [f"age {self.age}"]
        if self.cause:  lines += [f"died with {self.cause}"]
        lines += [time.strftime('%e %B, %Y')]
        return '\n'.join(f'{line:^25s}' for line in lines)


def train(state: TrainingState, data: DataLoader, loss_fun: Callable):
    state.model.train()

    # go through data
    losses = []
    data = tqdm(data, desc=f"{state.name} training, epoch {state.epochs:,}")
    for i, (input, target) in enumerate(data):
        # run the model
        pred = state.model(input, horizon=target.shape[1])
        loss = loss_fun(pred, target)
        # print(f"{state.name} training, epoch {state.epochs:,} {100*i//len(data):3}%, steps {state.steps:,}, loss {loss}")
        data.set_postfix(loss=float(loss))
        if not loss.isfinite():
            if state.steps < 18:
                raise EDException(name=state.name, age=state.steps, cause=f"{loss=:g}")
            print(f"!!! {loss=:g}, skipping iteration")
            continue

        # gradient
        loss.backward()
        clip_grad_value_(state.model.parameters(), 1e3)  # clip infinities
        grad_norm = clip_grad_norm_(state.model.parameters(), 1)  # clip norm
        if not grad_norm.isfinite():
            if state.steps < 18:
                raise EDException(name=state.name, age=state.steps, cause=f"{grad_norm=:g}")
            print(f"!!! {grad_norm=:g}, skipping iteration")
            continue

        # record loss
        losses.append(float(loss))
        state.writer.add_scalar('train iteration loss', loss, state.steps)
        state.writer.add_scalar('grad norm', grad_norm, state.steps)

        # optimize
        state.optimizer.step()
        state.optimizer.zero_grad()
        state.steps += 1

    # log loss
    loss = sum(losses) / len(losses)
    state.writer.add_scalar('training loss', loss, state.steps)
    # print(f"{state.name} training, epoch {state.epochs:,} {100:3}%, steps {state.steps:,}, average loss {loss}")
    data.set_postfix(average_loss=float(loss))
    save_loss_to_file(state, loss, is_training=True)
    state.epochs += 1
    state.writer.add_scalar('epochs', state.epochs, state.steps)

    return loss


def validate(state, data, loss_metric: str):
    state.model.eval()
    with torch.no_grad():

        # look at current weights
        for name, param in state.model.named_parameters():
            if not param.isfinite().all():
                raise ValueError(f"Parameter {name} has non-finite values.")
            state.writer.add_histogram(f'weights/{name}', param, state.steps)

        # go through data
        losses = []
        all_metrics = defaultdict(list)
        data = tqdm(data, desc=f"{state.name} validating, epoch {state.epochs:,}")
        for i, (input, target) in enumerate(data):
            # run the model
            pred = state.model(input, horizon=target.shape[1])
            metrics = Metrics(pred, target)
            loss = metrics[loss_metric]
            data.set_postfix(loss=float(loss))

            # introspection
            # image/hist introspection for the first few
            if i in range(0, len(data), len(data) // 8):
                if state.epochs == 1:
                    state.writer.add_figure(f'plot input/{i}', input[0].plot(), state.steps)
                    state.writer.add_figure(f'plot target/{i}', target[0].plot(), state.steps)
                state.writer.add_figure(f'plot prediction/{i}', pred[0].plot(), state.steps)

            # TODO get rid of this?
            # for metric in metrics.images:
            # 	value: Tensor = metrics[metric]
            # 	isfinite = value.isfinite()
            # 	if not isfinite.any(): continue
            # 	state.writer.add_histogram(f'hist {metric}/{i}', value, state.steps)
            # 	finite = value[isfinite]
            # 	mn = finite.min()  # minimum
            # 	rn = finite.max() - mn  # range
            # 	iB,iT,iC,iH,iW = range(5)
            # 	sB,sT,sC,sH,sW = value.shape
            # 	image = value.sub(mn).div(rn).permute(iB,iC,iH,iT,iW).reshape(1,sB*sC*sH,sT*sW)
            # 	state.writer.add_image(f'image {metric}/{i}', image, state.steps)

            # record loss and other metrics
            losses.append(loss)
            for metric in metrics.scalars:
                all_metrics[metric].append(metrics[metric])

        # log loss and metrics
        loss = sum(losses) / len(losses)
        state.writer.add_scalar('validation loss', loss, state.steps)
        for metric, values in all_metrics.items():
            value = sum(values) / len(values)
            state.writer.add_scalar(f'{metric}', value, state.steps)

        # adjust learning rate
        if state.lr_scheduler is not None:
            if isinstance(state.lr_scheduler, ReduceLROnPlateau):  # adaptive
                state.lr_scheduler.step(loss)
            else:
                state.lr_scheduler.step()
            for i, param_group in enumerate(state.optimizer.param_groups):
                state.writer.add_scalar(f'learning rate/{i}', param_group['lr'], state.steps)

        save_loss_to_file(state, loss, is_training=False)
        # this may trigger a state.save(best=True)
        state.loss = loss


        return loss


def fit(file_glob=None, max_epochs=15):
    with cuda_context():
        # train_data = torch.load('train_data.pth')
        # val_data = torch.load('val_data.pth')
        data = get_data()
        state = get_trainstate(file_glob)

        try:
            with CatchSignal() as stop:
                # main loop
                while not stop:
                    if state.epochs >= max_epochs:

                        print(f"Reached maximum epochs: {max_epochs}. Stopping training.")
                        break
                    try:
                        train(state, data.train, loss_fun)
                        validate(state, data.valid, loss_metric)

                    except EDException as e:
                        print(e)  # memorial
                        state = get_trainstate(file_glob)  # born again!

        except Exception as e:
            state.save(err=True)
            raise e
        except KeyboardInterrupt as e:
            state.save(err=True)
            raise e
        else:
            state.save()

def save_loss_to_file(state, loss, is_training=True):
    """Save the average loss for the current epoch to a file."""
    log_file = os.path.join(state.dir, "loss_log.txt")
    mode = 'a' if os.path.exists(log_file) else 'w'  # Append if file exists, else create new

    with open(log_file, mode) as f:
        if is_training:
            f.write(f"Epoch {state.epochs}: Training Loss = {loss}\n")
        else:
            f.write(f"Epoch {state.epochs}: Validation Loss = {loss}\n")
def main():
    file_glob = sys.argv[1] if len(sys.argv) > 1 else None
    fit(file_glob)


if __name__ == '__main__':
    mpl.use('Agg')
    main()