# Standard Library
from itertools import count, islice
from math import ceil, floor

# PyPI
import h5py
import torch
from torch.utils.data import DataLoader
import matplotlib as mpl

# Local pedpred
from ..config import cfg
from ..dataset import GridFromH5Dataset, SeqDataset
# from ..flow_interpolation.models_gpytorch import ConstrGPModel, HYPS, optimize_traj
from ..grid import Grid, GridData
from ..models import PedPred3, PedPred
from ..saving_loading import TrainingState
from ..test import get_model
from ..tools import tqdm
from ..train import get_data

# Local planning
from .dijk import dijkstra
from .graph import InfiniteGraph, Predictor

default_horizon = 50  # steps = 30s iff Config.period = 0.5


class Planner:
    def __init__(self, *, n, model, grid, period, dataset, static_dataset=None):
        # can (and should) share graph, but careful:
        # - start with different nodes
        # - clear predictor
        self.graph = InfiniteGraph(n, grid.shape, Predictor(model))

        self.dataset = dataset
        self.static_dataset = static_dataset

    # # self.linear_A = torch.load('average_jacobian.pt')
    # # self.linear_B = torch.load('estimated_bias.pt')
    # self.linear_A = torch.load('average_jacobian.pt').squeeze(0).squeeze(4).reshape(17280, 17280)
    # self.linear_b = torch.load('estimated_bias.pt').view(1, -1)
    #
    # # Verify shapes
    # assert self.linear_A.shape == (17280, 17280)
    # assert self.linear_b.shape == (1, 17280)

    def init_predictor(self, data_iter):
        self.graph.predictor.clear()  # start from scratch
        self.graph.predictor.linear = False
        for obs in islice(data_iter, 10 + 160):
            self.graph.predictor.inform(obs)

    def init_predictor_linear(self, data_iter):
        self.graph.predictor.clear()  # start from scratch
        self.graph.predictor.linear = True
        for obs in islice(data_iter, 10 + 160):
            self.graph.predictor.inform_linear(obs)

    @staticmethod
    def goal2waypoints(goal_node):
        waypoints = [goal_node]
        while (prev_node := waypoints[-1].parent) is not None:
            waypoints.append(prev_node)
        waypoints.reverse()
        return waypoints

    def plan_static(self, start_p, goal_p, *, horizon=default_horizon, verbose=False):
        assert self.static_dataset is not None

        data = torch.as_tensor(self.static_dataset)
        C, H, W = data.shape
        data = data.expand(horizon, C, H, W)  # make large time dimension
        self.graph.predictor.clear()
        self.graph.predictor.data = data

        goal_idx = self.graph.nearest(goal_p)
        goal_condition = lambda node: (node.idx % node.graph.n) == goal_idx

        start_idx = self.graph.nearest([0, *start_p])
        start_node = self.graph.Node(start_idx, cost=0)

        self.graph.horizon = horizon
        goal_node = dijkstra(start_node, goal_condition, verbose=verbose)
        waypoints = self.goal2waypoints(goal_node)

        return waypoints

    def plan_once(self, start_p, goal_p, *, horizon=default_horizon, verbose=False):
        data_iter = iter(self.dataset)
        self.init_predictor(data_iter)
        goal_idx = self.graph.nearest(goal_p)
        # self.graph.goal = self.graph.position(goal_idx)  # for A*

        goal_condition = lambda node: (node.idx % node.graph.n) == goal_idx

        start_idx = self.graph.nearest([0, *start_p])
        start_node = self.graph.Node(start_idx, cost=0)

        self.graph.horizon = horizon
        goal_node = dijkstra(start_node, goal_condition, verbose=verbose)
        waypoints = self.goal2waypoints(goal_node)

        return waypoints

    def plan_online(self, start_p, goal_p, *, horizon=default_horizon, verbose=False):
        data_iter = iter(self.dataset)
        self.init_predictor(data_iter)
        goal_idx = self.graph.nearest(goal_p)
        self.graph.horizon = horizon
        # self.graph.goal = self.graph.position(goal_idx)  # for A*

        goal_condition = lambda node: (node.idx % node.graph.n) == goal_idx

        start_idx = self.graph.nearest([0, *start_p])
        start_node = self.graph.Node(start_idx, cost=0)

        # all_paths = []

        last_node = start_node

        for i in tqdm(count(), 'plan_online'):
            goal_node = dijkstra(last_node, goal_condition, verbose=(verbose >= 2))
            waypoints = self.goal2waypoints(goal_node)
            # all_paths.append(waypoints)

            for node in waypoints:
                time_idx = node.idx // node.graph.n
                if time_idx > i:
                    # this is the furthest the robot gets to within this time idx
                    break
            else:
                # found goal within this time idx
                pass
            last_node = node

            if verbose:
                print(f"{i=}, {last_node.time=}, {goal_node.time=}, {last_node.cost=}, {goal_node.cost=}")

            if goal_condition(last_node):
                break
            # print('i before inform_online= ', i)
            self.graph.predictor.inform_online(next(data_iter), i)

        return waypoints

    def plan_linear(self, start_p, goal_p, *, horizon=default_horizon, verbose=False):
        data_iter = iter(self.dataset)
        self.init_predictor_linear(data_iter)
        goal_idx = self.graph.nearest(goal_p)
        self.graph.horizon = horizon
        goal_condition = lambda node: (node.idx % node.graph.n) == goal_idx

        start_idx = self.graph.nearest([0, *start_p])
        start_node = self.graph.Node(start_idx, cost=0)

        # all_paths = []

        last_node = start_node

        for i in tqdm(count(), 'plan_linear'):
            goal_node = dijkstra(last_node, goal_condition, verbose=(verbose >= 2))
            waypoints = self.goal2waypoints(goal_node)
            # all_paths.append(waypoints)

            for node in waypoints:
                time_idx = node.idx // node.graph.n
                if time_idx > i:
                    # this is the furthest the robot gets to within this time idx
                    break
            else:
                # found goal within this time idx
                pass
            last_node = node

            if verbose:
                print(f"{i=}, {last_node.time=}, {goal_node.time=}, {last_node.cost=}, {goal_node.cost=}")

            if goal_condition(last_node):
                break

            self.graph.predictor.inform_online_linear(next(data_iter), i)

        return waypoints

    def plan_actual(self, start_p, goal_p, *, horizon=default_horizon, verbose=False):
        data_iter = iter(self.dataset)
        self.init_predictor(data_iter)  # to waste the first 10 warmup observations

        data = []
        for obs in islice(data_iter, horizon):
            data.append(obs)
        data = torch.stack(data)
        self.graph.predictor.clear()
        self.graph.predictor.data = data

        goal_idx = self.graph.nearest(goal_p)
        goal_condition = lambda node: (node.idx % node.graph.n) == goal_idx

        start_idx = self.graph.nearest([0, *start_p])
        start_node = self.graph.Node(start_idx, cost=0)

        self.graph.horizon = horizon
        goal_node = dijkstra(start_node, goal_condition, verbose=verbose)
        waypoints = self.goal2waypoints(goal_node)

        return waypoints


def get_dataset(step=1, **kwargs):
    resolution = cfg.resolution
    r = 1 / resolution
    grid = Grid(origin=(38.2789, -15.8076), theta=2.5647, shape=(36 * r, 12 * r), resolution=resolution)
    period = cfg.period
    dataset = GridFromH5Dataset(
        'data/ATC/atc-20130707.h5',
        grid, period * step,
        kernel=cfg.kernel,
        random_rotation=False,
        normalise_period=period,
        **kwargs
    )
    return dataset


def plot_paths(paths, dataset, step=1):
    """ Paths is a list. A path is (ps,ts). """
    max_t = max(path[1][-1] for path in paths.values())
    # print(max_t)
    stop_t = ceil(max_t / step)
    # print(stop_t)
    # stop_t = 12  # todo debug

    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
    T = 4
    B= 1
    # B = (stop_t - 1) // T + 1  # height, previously "batch"
    H, W = dataset.grid.shape
    print(' dataset.grid.shape', dataset.grid.shape)
    figsize = 1 / 10  # per grid cell (was 3 per image)
    fig = plt.figure(figsize=((T + 0.3) * W * figsize + 0.1, B * H * figsize + 0.2), tight_layout=True)
    axs = ImageGrid(fig, 111, nrows_ncols=(B, T), axes_pad=(0.05, 0.10), share_all=True
                    )
    # cax = axs.cbar_axes[0]
    start_t = 12 / step
    assert start_t.is_integer()
    start_t = int(start_t)
    data = dataset[start_t:start_t + T * B]
    # print('start_t:start_t+T*B', start_t, start_t+T*B)
    data = GridData(data)
    # arr = dataset[10][0].data  # Replace `.data` with the correct attribute if needed
    # print('Number of nonzero values:', (arr != 0).sum())
    # print('actual data in plot' , dataset[10][0])
    # fig = data.plot(fig=fig, axs=axs, cax=cax)
    fig = data.plot(fig=fig, axs=axs, show_colorbar=False)

    colors = dict(  # also defines draw order
        # static='purple',
        # actual='black',
        # online='green',
        # once='blue',
        # linear='black',
        # online_stefan = 'red',
        # once_stefan = 'orange',
    )
    for i in range(stop_t):
        ax = fig.axes[i]
        for method in colors:
            # print('methods', method)
            if method not in paths: continue
            # print(method)
            path = paths[method]
            color = colors[method]
            plot_kwargs = dict(color=color, linewidth=5, markersize=10)
            ps, ts = path
            ts = ts / step  # plot multiple time steps together
            ps = ps - 0.5  # grid is pixel centered
            xs, ys = ps.unbind(dim=-1)
            h_full = ax.plot(ys, xs, '-', **plot_kwargs, alpha=1 / 3)
            current = (ts >= i) & (ts < i + 1)
            ps = ps[current, :]
            xs, ys = ps.unbind(dim=-1)
            h_curr = ax.plot(ys, xs, '.-', **plot_kwargs, markevery=[0, -1])

    return fig


def plan2h5(model_name, n):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = get_model(model_name)
    model = PedPred3()
    checkpoint = torch.load('apt-ibex_train_model_28D.pth', map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    model2 = PedPred()
    model2.load_state_dict(torch.load('corridor_train28d_Stefan_8epoch.pth', map_location=device))
    model2.eval()
    # r: int = 1  # cells per meter
    # grid = Grid((-6, -6), 0, (12 * r, 12 * r), 1 / r)
    # period = 0.5
    # dataset = 'data/alex1'
    #
    # with h5py.File(f'{dataset}_static.h5','r') as h5:
    # 	static_data = torch.as_tensor(h5['data'])
    # dataset = GridFromH5Dataset(f'{dataset}_valid.h5', grid, period, random_rotation=False)

    dataset = get_dataset()
    grid = dataset.grid
    period = dataset.period
    static_data = None  # ignore for now

    planner = Planner(n=n, model=model, grid=grid, period=period, dataset=dataset, static_dataset=static_data)
    planner2 = Planner(n=n, model=model2, grid=grid, period=period, dataset=dataset, static_dataset=static_data)

    H, W = grid.shape
    headings = dict(
        SE=([0, 0], [H, W]),
        SW=([0, W], [H, 0]),
        NE=([H, 0], [0, W]),
        NW=([H, W], [0, 0]),
        S=([0, W], [H, W]),
        N=([H, W], [0, W]),
        E=([H, 0], [H, W]),
        W=([H, W], [H, 0]),
        S2=([0, W/2], [H/2, W/2]),
        N2=([H, W/2], [0, W]),
        E2=([H, 0], [H, W/2]),
        W2=([H, W/2], [H/2, 0]),
        S3=([0, W], [H/2, W]),
        N3=([H/2, W], [0, W]),
        E3=([H/2, 0], [H, W/2]),
        W3=([H, W/2], [H, 0]),
    )
    for heading in tqdm(headings, "headings"):
        start_p, goal_p = headings[heading]

        all_waypoints = dict(
            once_stefan=planner2.plan_once(start_p, goal_p),
            online_stefan=planner2.plan_online(start_p, goal_p),
            once=planner.plan_once(start_p, goal_p),
            online=planner.plan_online(start_p, goal_p),
            linear=planner.plan_linear(start_p, goal_p)
        )
        horizon = ceil(max(waypoints[-1].time for waypoints in all_waypoints.values()))
        all_waypoints.update(
            # 	# static=planner.plan_static(start_p, goal_p, horizon=horizon),
            actual=planner.plan_actual(start_p, goal_p, horizon=horizon),
        )

        expected_costs = {name: waypoints[-1].cost for name, waypoints in all_waypoints.items()}
        # overwrite predictions with real data (as done by plan_actual?)
        planner.graph.predictor.data = dataset[10 + 160:10 + horizon + 160]
        # clear all recorded costs (except for start nodes)
        for waypoints in all_waypoints.values():
            for node in waypoints[1:]:
                node._cost = None
        # NOW you can calculate actual costs
        actual_costs = {name: waypoints[-1].cost for name, waypoints in all_waypoints.items()}

        # save paths
        paths = {}
        for method, waypoints in all_waypoints.items():
            ps = []
            ts = []
            for node in waypoints:
                ps.append(node.position)
                ts.append(node.time)
            ps = torch.stack(ps)
            ts = torch.stack(ts)
            paths.update({method: (ps, ts)})

        with h5py.File(f'planning_all_{n}_{heading}_A32.h5', 'x') as h5:
            for method in paths:
                g = h5.create_group(method)
                g.attrs['expected_cost'] = expected_costs[method]
                g.attrs['actual_cost'] = actual_costs[method]
                ps, ts = paths[method]
                g.create_dataset(f'path_ps', data=ps.cpu().numpy())
                g.create_dataset(f'path_ts', data=ts.cpu().numpy())

    return


def load_h5_old(filename):
    """ to paths """
    paths = []
    with h5py.File(filename, 'r') as h5:
        for i in range(4):
            ps = torch.as_tensor(h5[f'path_{i}_ps'])
            ts = torch.as_tensor(h5[f'path_{i}_ts'])
            path = (ps, ts)
            paths.append(path)
    return paths


def load_h5(filename, costs=False):
    """ to paths """
    expected_costs = {}
    actual_costs = {}
    paths = {}
    with h5py.File(filename, 'r') as h5:
        for method in h5:
            g = h5[method]
            expected_costs[method] = g.attrs['expected_cost']
            actual_costs[method] = g.attrs['actual_cost']
            ps = torch.as_tensor(g['path_ps'])
            ts = torch.as_tensor(g['path_ts'])
            paths[method] = (ps, ts)

    method_order = ['once_stefan', 'online_stefan', 'once', 'online', 'actual']
    paths = {method: paths[method] for method in method_order if method in paths}
    if costs:
        return paths, expected_costs, actual_costs
    return paths


def plot_h5_multi_horizon(filename, max_horizon=10):
    for horizon in range(1, max_horizon + 1):
        paths = load_h5(filename)

        # Slice paths to only include up to the desired horizon
        new_paths = {}
        for method, (ps, ts) in paths.items():
            mask = ts < horizon
            new_paths[method] = (ps[mask], ts[mask])

        step = 1
        dataset = get_dataset(step, attach_point_data=True)

        # plot and save
        f = plot_paths(new_paths, dataset, step=step)
        f.savefig(filename.replace('.h5', f'_horizon{horizon}_{step}.pdf'))
        f.savefig(filename.replace('.h5', f'_horizon{horizon}_{step}.png'))
        plt.close(f)


def plot_h5(filename,i):
    paths = load_h5(filename)

    step = 12
    dataset = get_dataset(step, attach_point_data=True)

    f = plot_paths(paths, dataset, step=step)
    f.savefig(filename.replace('.h5', f'_{step}_{i}.pdf'))
    return


def sync_path_to_period(period, ps, ts):
    import numpy as np
    t = np.arange(ts[-1] + period, step=period)
    x = np.interp(t, ts, ps[:, 0])
    y = np.interp(t, ts, ps[:, 1])
    return x, y


def animate_h5(filename):
    paths = load_h5(filename)

    step = 1 / 5
    data = get_dataset(step, attach_point_data=True)
    step_period = data.period

    robot = sync_path_to_period(step_period, *paths['online'])
    # N = 1372  # ??
    N = len(robot[0])

    gd = GridData(data[:N])
    gd.plot_anim(period=step_period, gif=f'{filename}.gif', video=f'{filename}.mp4', robot=robot)


def smoov(filename, modelname):
    from matplotlib import pyplot as plt
    paths, expected_costs, actual_costs = load_h5(filename, costs=True)
    path = paths['once']
    expected_cost = expected_costs['once']
    actual_cost = actual_costs['once']
    ps, ts = path
    import numpy as np
    period = 0.1
    t = np.arange(ts[-1] + period, step=period)
    x = np.interp(t, ts, ps[:, 0])
    y = np.interp(t, ts, ps[:, 1])
    traj = np.stack([t, x, y], axis=-1)
    traj = torch.as_tensor(traj).float()

    data = get_dataset(attach_point_data=True)
    predictor = Predictor(model=get_model(modelname))
    predictor.inform(data[:10])
    predictor.predict(default_horizon)
    smoover = ConstrGPModel(predictor.data, *HYPS)

    invasiveness = smoover.invasiveness(traj)

    traj = optimize_traj(traj, smoover, plot=True)
    return traj


"""Kavi"""


def draw_path3D():
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    from mpl_toolkits import mplot3d
    from pedpred.grid import Grid, GridData
    from pedpred.dataset import GridFromH5Dataset

    mpl.use('qt5agg')
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    model_name = 'live-jaguar'
    n = 10000
    heading = 'SW'
    filename = f'planning_{model_name}_{n}_{heading}.h5'
    paths = load_h5(filename)

    ps, ts = paths['once']
    xs, ys = ps.unbind(dim=-1)

    r: int = 1  # cells per meter
    grid = Grid((-6, -6), 0, (12 * r, 12 * r), 1 / r)
    period = 0.5
    dataset = 'data/alex1'
    step = 2
    dataset = GridFromH5Dataset(f'{dataset}_valid.h5', grid, period, random_rotation=False, attach_point_data=True)
    model = PedPred()
    TrainingState(model_name, model=model)
    pred = model(dataset[0:10].unsqueeze(dim=1), horizon=ceil(ts[-1] / period)).squeeze(dim=1)
    pred = GridData(pred)
    self = pred

    import numpy as np
    from pedpred.tools.mpl import circles
    T, _, H, W = self.shape
    xx, yy = np.meshgrid(range(H + 1), range(W + 1), indexing='ij')
    x, y = np.meshgrid(range(H), range(W), indexing='ij')
    x = x + 0.5;
    y = y + 0.5

    for t in range(0, T, 10):
        zi = t * period
        d = self.density[t, 0].detach().cpu()  # .numpy()
        u, v = self.vel_mean[t, :].detach().cpu().numpy()
        s = self.vel_std[t, 0].detach().cpu().numpy()

        z = zi * np.ones_like(x)
        zz = zi * np.ones_like(xx)
        fc = plt.cm.Blues(d.clamp(0, 1).numpy())
        fc[..., -1] = (d * 1.5).clamp(0, 1)  # change the alpha value
        ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, facecolors=fc, shade=False)

    # x_val = xs[t]
    # y_val = ys[t]
    # z_val = ts[t]
    #
    # ax.quiver(y,x,z,v,u,0, color=vel_color, length=vel_scale, arrow_length_ratio=0.1)

    # col = circles(y + v*vel_scale, x + u*vel_scale, s*vel_scale, color=vel_color, fc='none')
    # zs = 0
    # pvv = ax.add_collection3d(col, zs=zs, zdir='z')
    ax.plot(xs, ys, ts, '.-r', linewidth=5, markevery=[0, -1])
    return fig


if __name__ == '__main__':
    # mpl.use('Agg')
    # draw_path3D()
    # # model = 'amazed-maggot'  # WMSE
    # model = 'live-jaguar'  # WNLLL
    model = 'bold-yeti'  # atc:corridor
    model += '_best'
    n = 1_000

    # plan2h5(model, n)
    # exit()

for heading in [
'SW'
	# 'SE','SW','NE','NW',
	# 'S','N','W','E','S2','N2','W2','E2','S3','N3','W3','E3'
]:
	filename = f'planning baseline+lightweight model/planning_all_{n}_{heading}_A32.h5'

	plot_h5(filename,18)
# 	plot_h5_multi_horizon(f'planning_{model}_{n}_{heading}.h5', max_horizon=10)
# # 	animate_h5(filename)
# 	# smoov(filename, model)
