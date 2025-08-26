# Standard Library
from functools import total_ordering, wraps
from math import ceil, e as euler, inf, log
from typing import Any, Callable

# PyPI
import torch
from pyplusplus.function_transformers import output
from torch.utils.data import DataLoader

# pedpred
from ..dataset import GridFromH5Dataset, SeqDataset
from ..grid import Grid, GridData
from ..models import PedPred
from ..saving_loading import TrainingState
from ..planning.dijk import dijkstra
from ..tools import tqdm


# def method_decorator(decorator: Callable):
# 	@wraps(decorator)
# 	def wrapped_decorator(func: Callable[[object,...],Any]):
# 		@wraps(func)
# 		def wrapped_func(self: object, *args, **kwargs):
# 			return func(self, *args, **kwargs)
# 		return wrapped_func
# 	return wrapped_decorator
# 
# Method = Callable[[object,Any],Any]
# MethodDecorator = Callable[[object,Method],Method]
# def method_decorator(decorator: MethodDecorator):
# 	...
# 	


def transform_back(x_transformed):
	x_hat_raw = x_transformed.clone()
	x_hat_raw[:, :, 0, :, :] = torch.exp(x_transformed[:, :, 0, :, :])  # back to density
	x_hat_raw[:, :, 3, :, :] = torch.exp(x_transformed[:, :, 3, :, :])  # back to velocity variance
	return x_hat_raw

class Predictor:
	def __init__(self, model: PedPred):
		self.linear = False
		self.model = model
		self.encoder_hidden = None
		self.forecaster_hidden = None
		self.buffer = None

		self.A = torch.load('logx=Ax+B/A_pedpred3_t16.pt')  # [1, 10, 4, 36, 12, 1, 10, 4, 36, 12]
		self.B = torch.load('logx=Ax+B/B_pedpred3_t16.pt').view(1, -1)
		# self.A = torch.load('average_jacobian_pedpred3_200.pt')
		# self.B = torch.load('estimated_bias.pt').view(1, -1)
		self.A = self.A.squeeze(0).squeeze(4).reshape(17280, 17280)  # reshape to [D, D]



		self.data: torch.Tensor = []
		self.min_cost_per_dist = []
		# self.bar = tqdm(desc=Predictor, total=)
	
	
	def clear(self):
		self.encoder_hidden = None
		self.forecaster_hidden = None
		self.data = []
		self.min_cost_per_dist = []
	
	
	@torch.no_grad()
	def inform(self, input: GridData):
		if input.ndim < 3: raise ValueError("Prediction.inform input needs to be at least 3 dimensional, [C,H,W].")
		if input.ndim < 4: input = input.unsqueeze(dim=0)  # add time dimension
		if input.ndim < 5: input = input.unsqueeze(dim=1)  # add batch dimension
		
		input = GridData(input).as_tensor('density','vel_mean','vel_var')  # todo: this should be part of the model
		self.encoder_hidden = self.model.encoder(input, self.encoder_hidden)
		self.forecaster_hidden = list(reversed(self.encoder_hidden))
		self.data = []
		self.min_cost_per_dist = []

	def inform_linear(self, input: GridData):
		if input.ndim < 3: raise ValueError("Prediction.inform input needs to be at least 3 dimensional, [C,H,W].")
		if input.ndim < 4: input = input.unsqueeze(dim=0)  # add time dimension
		if input.ndim < 5: input = input.unsqueeze(dim=1)  # add batch dimension

		input = GridData(input).as_tensor('density', 'vel_mean', 'vel_var')  # todo: this should be part of the model

		if not hasattr(self, 'buffer_size'):
			self.buffer_size = 10  # or make it an argument

		# Initialize buffer with first state(s)
		if not hasattr(self, 'buffer') or self.buffer is None:
			self.buffer = input
		else:
			self.buffer = torch.cat([self.buffer, input], dim=1)  # Concatenate along time: [1, T, 4, H, W]
			if self.buffer.shape[1] > self.buffer_size:
				self.buffer = self.buffer[:, -self.buffer_size:]  # Keep last `buffer_size` frames

		# print('buffer shape', self.buffer.shape)
		self.data = []
		self.min_cost_per_dist = []

	
	def inform_online(self, input: GridData, idx: int):
		""" Same as inform, but don't reset time to zero. Predictions extend into future. """
		if input.ndim < 3: raise ValueError("Prediction.inform input needs to be at least 3 dimensional, [C,H,W].")
		if input.ndim < 4: input = input.unsqueeze(dim=0)  # add time dimension
		
		past_data = self.data[:idx]
		# print('shape past data', past_data.shape)
		past_min_cost_per_dist = self.min_cost_per_dist[:idx+1]
		self.inform(input)
		# print('shape input in inform_online function', input.shape)
		self.data = torch.stack((*past_data, *input))
		# print('shape data', self.data.shape)
		self.min_cost_per_dist = past_min_cost_per_dist

	def inform_online_linear(self, input: GridData, idx: int):
		""" Same as inform, but don't reset time to zero. Predictions extend into future. """
		if input.ndim < 3: raise ValueError("Prediction.inform input needs to be at least 3 dimensional, [C,H,W].")
		if input.ndim < 4: input = input.unsqueeze(dim=0)  # add time dimension

		past_data = self.data[:idx]
		# print('shape past data', past_data.shape)
		past_min_cost_per_dist = self.min_cost_per_dist[:idx + 1]
		self.inform_linear(input)
		# print('shape input in inform_online function', input.shape)
		self.data = torch.stack((*past_data, *input))
		# print('shape data', self.data.shape)
		self.min_cost_per_dist = past_min_cost_per_dist


	@torch.no_grad()
	def predict(self, horizon: int):
		self.linear = False
		# print('horizen', horizon)
		# print('len(self.data)', len(self.data))
		new_horizon = horizon - len(self.data)
		if new_horizon > 0:
			# print(f"Predicting {new_horizon} to {horizon}.")  # debug
			# self.bar.update(new_horizon)
			# print('new_horizon', new_horizon)
			output, self.forecaster_hidden = self.model.forecaster(self.forecaster_hidden, horizon=new_horizon)
			# print('output.shape before squezee',output.shape)
			output = output.squeeze(dim=1)  # remove batch dimension
			# print('output.shape after squezee', output.shape)
			output = GridData.from_tensor(output,('logdensity','vel_mean','vel_logvar'))  # todo: this should be part of the model
			output = output.as_tensor('density','vel_mean','vel_var')

			self.data = torch.stack((*self.data, *output))
			
			density, vel_mean, vel_var = self.data.split((1,2,1), dim=-3)
			vel_mean2 = vel_mean.square().sum(dim=-3, keepdim=True)
			opt_spd = (vel_mean2 + 2*vel_var).sqrt()
			# min_cost_per_dist = density * 2*vel_var / opt_spd  # looser bound
			min_cost_per_dist = 2 * density * (opt_spd - (2*vel_mean2).sqrt())  # tighter bound
			self.min_cost_per_dist = min_cost_per_dist.min(dim=-2).values.min(dim=-1).values  # min across space



	@torch.no_grad()
	def predict_linear(self, horizon: int):
		self.linear = True
		# print('horizen', horizon)
		# print('len(self.data)', len(self.data))
		new_horizon = horizon - len(self.data)
		if new_horizon > 0:
			#predict next 10 state using linear model but take next "new_horizen" sttae as output
			# print('buffer shape in prediction', self.buffer.shape)
			input_flat = self.buffer.view(-1)
			# print('input_flat.shape', input_flat.shape)
			next_state_flat = self.A @ input_flat + self.B
			next_state_log = next_state_flat.view_as(self.buffer)
			output = transform_back(next_state_log)
			# print(output.shape)
			output = output[:,:new_horizon,:,:,:]
			# print(output.shape)
			# output, self.forecaster_hidden = self.model.forecaster(self.forecaster_hidden, horizon=new_horizon)
			output = output.squeeze(0)  # remove batch dimension
			# print(output.shape)

			self.data = torch.stack((*self.data, *output))

			density, vel_mean, vel_var = self.data.split((1, 2, 1), dim=-3)
			vel_mean2 = vel_mean.square().sum(dim=-3, keepdim=True)
			opt_spd = (vel_mean2 + 2 * vel_var).sqrt()
			# min_cost_per_dist = density * 2*vel_var / opt_spd  # looser bound
			min_cost_per_dist = 2 * density * (opt_spd - (2 * vel_mean2).sqrt())  # tighter bound
			self.min_cost_per_dist = min_cost_per_dist.min(dim=-2).values.min(dim=-1).values  # min across space
	
	def __getitem__(self, item):
		it,ih,iw = item
		# print('it,ih,iw', it,ih,iw)
		# print('it.max()+1', it.max()+1)
		predict_fn = self.predict_linear if self.linear else self.predict
		predict_fn(it.max() + 1)
		return self.data[it,:,ih,iw].split((1,2,1), dim=-1)




class InfiniteGraph:
	
	@total_ordering
	class _Node:
		def __init_subclass__(cls, graph=None, **kwargs):
			super().__init_subclass__(**kwargs)
			cls.graph: InfiniteGraph = graph
		
		__slots__ = (
			'idx',
			'parent',
			'_child_idx',
			'_cost',
			'_cost_to_go',
		)
		# idx: int
		# parent: Any=None
		# _child_idx: int=None
		# _cost: float=None
		
		def __init__(self, idx, parent=None, child_idx=None, *, cost=None):
			self.idx = idx
			self.parent = parent
			self._child_idx = child_idx
			self._cost = cost
			self._cost_to_go = None
		
		@property
		def cost(self):
			if self._cost is None:
				self._cost = self.parent.cost + self.graph.traversal_cost(self.parent.idx, self._child_idx).item()
			return self._cost
		
		@property
		def cost_to_go(self):
			if self._cost_to_go is None:
				self._cost_to_go = self.graph.heuristic(self.idx).item()
			return self._cost_to_go
		
		@property
		def cost_to_goal(self): return self.cost + self.cost_to_go
		
		_cmp_value = cost  # regular dijkstra
		# _cmp_value = cost_to_goal  # A*
		def __lt__(self, other): return self._cmp_value < other._cmp_value
		
		@property
		def time(self): return self.graph.time(self.idx)
		@property
		def position(self): return self.graph.position(self.idx)
		
		def children(self):
			idx = self.graph.children(self.idx)
			for child in range(len(idx)):
				child_node = self.__class__(idx[child].item(), self, child_idx=child)
				if child_node.time >= self.graph.horizon: continue
				yield child_node
	
	
	def __init__(self, n: int, shape, predictor: Predictor):
		""" Now in image coordinates! Because the (velocity) data is in image coordinates."""
		self.n = n
		self.shape = shape
		self.predictor = predictor
		self.horizon = 50
		
		t = torch.rand(n,1).sort(dim=0).values
		p = torch.rand(n,2) * shape
		s = torch.cat((t,p), dim=-1)  # [n,3]
		self._state = s
		
		# tf = t + 1  # time one period into the future
		# sf = torch.cat((tf,p), dim=-1)  # states one period into future
		# ss = torch.cat((s,sf), dim=0)  # states across 2 periods, [M*n,3]
		M = 5  # future steps
		t_vector = torch.as_tensor([1,0,0])
		ss = s + torch.arange(M).reshape(M,1,1) * t_vector  # [M,1,3]
		s1 = s .reshape(n,   1, 3)  # start nodes
		s2 = ss.reshape(1, M*n, 3)  # end nodes
		
		d = s2-s1  # [n,M*n,3]
		d_t = d[...,0]
		d_p = d[...,1:3]
		v2 = d_p.square().sum(dim=-1) / d_t.square()
		max_spd = 1  # todo beware units: per cell per period
		d2 = d.square().sum(dim=-1)  # [n,M*n]
		invalid = (d_t <= 0) | (d_t > M-1) | (v2 > max_spd**2)  # only forwards, maximum time-distance of period, max speed
		invalid = invalid | ( (d_p == 0).all(dim=-1) & (d_t > 1) )
		d2[invalid] = inf
		
		k_PRMstar = euler * (1+1/2)
		k = ceil(k_PRMstar * log(n))
		
		children = d2.topk(k=k, largest=False).indices
		self._children = children
		
		class Node(self._Node, graph=self): pass
		self.Node = Node
		
		import numpy as np
		self._edge_seg_grid_idx = np.empty((n,k), dtype=object)
		self._edge_seg_vel = np.empty((n,k), dtype=object)
		self._edge_seg_dt = np.empty((n,k), dtype=object)
		# Note: this needs to be done in local coordinates
		
		t_vector = torch.as_tensor([1,0,0])
		for node in tqdm(range(n), 'graph edge cache'):
			s1 = s[node]
			for child in range(k):
				child_node = children[node,child]
				s2 = s[child_node % n] + (child_node // n)*t_vector
				ds = s2-s1
				dt = ds[0]
				dx = ds[1:3]
				vel = dx/dt
				if vel.square().sum() > max_spd**2:
					vel[:] = inf  # to force invasiveness to infinity
				
				q_boundaries = [torch.as_tensor(x) for x in (0,1)]  # q, from 0 to 1, defines s(q) = ds*q + s1
				dir = ds.sign()
				s_start = (dir*s1 + 1).floor().int()
				s_stop = (dir*s2).ceil().int()  # not-inclusive, use with range
				
				for dim in range(3):
					s_range = range(s_start[dim], s_stop[dim])
					if s_range:
						s_range = dir[dim] * torch.as_tensor(s_range)
						q = (s_range - s1[dim])/ds[dim]
						q_boundaries += q
				
				q_boundaries = torch.stack(q_boundaries).sort().values
				dq = q_boundaries[1:] - q_boundaries[:-1]
				# mid is halfway between boundaries, should be more (numerically) robust than other methods
				q_mid = q_boundaries[:-1] + dq/2.
				s_mid = q_mid.unsqueeze(dim=-1) * ds + s1
				grid_idx = s_mid.floor().int()
				vel = vel.unsqueeze(dim=0)
				dt = dt * dq
				
				self._edge_seg_grid_idx[node,child] = grid_idx
				self._edge_seg_vel[node,child] = vel
				self._edge_seg_dt[node,child] = dt
		
		return
	
	
	# @method_decorator
	def divmod_node(f: Callable[[object,int,int,Any],Any]):
		@wraps(f)
		def wrap(self, node: int, *args):
			q = node // self.n
			r = node % self.n
			return f(self, q, r, *args)
		return wrap
	
	
	@divmod_node
	def time(self, q, r): return self._state[r, 0] + q
	@divmod_node
	def position(self, q, r):return self._state[r, 1:3]
	@divmod_node
	def children(self, q, r): return self._children[r] + q * self.n
	
	@divmod_node
	def traversal_cost(self, node_q, node_r, child):
		grid_idx = self._edge_seg_grid_idx[node_r, child]  +  node_q * torch.as_tensor([1,0,0])  # todo check this
		vel = self._edge_seg_vel[node_r, child]
		dt = self._edge_seg_dt[node_r, child]
		
		density, vel_mean, vel_var = self.predictor[grid_idx.unbind(dim=-1)]  # these are probably normalised to image coords
		
		# TODO: NOTE 2*vel_var. 'vel_var' is isotropic variance, not scalar variance as per formulas in paper.
		alpha= 1
		beta = 0.0001

		dist = (self.position(node_r) - self.position(child)).square().sum(dim=-1).sqrt()
		cost = (alpha * (dt.squeeze() * density.squeeze() * ( (vel - vel_mean).square().sum(dim=-1)  +  2 * vel_var.squeeze())) +
				beta *dist)

		cost = cost.sum()
		if cost.isnan():
			cost = torch.tensor(torch.inf)
		if cost.isinf():
			print(f"INFINITE COST between {node_r}->{child}")
			print(f"Grid values: density={density}, vel={vel_mean}")
		return cost
	
	@divmod_node
	def heuristic(self, node_q, node_r):
		dist = (self.position(node_r) - self.goal).square().sum(dim=-1).sqrt()
		self.predictor.predict(node_q+1)
		min_cost_per_dist = self.predictor.min_cost_per_dist[node_q]
		return dist * min_cost_per_dist
	
	def nearest(self, state):
		state = torch.as_tensor(state)
		ds = state - self._state[:,-len(state):]  # will treat time the same as space (for distance) if you give it
		d2 = ds.square().sum(dim=-1)
		idx = d2.argmin()
		return idx.item()



# if __name__ == '__main__':
# 	#%% get model
# 	model = PedPred()
# 	
# 	# name = 'amazed-maggot'  # r=1, period=0.5, loss=WMSE
# 	# name = 'live-jaguar'  # r=1, period=0.5, loss=weighted NLLL
# 	name = 'merry-macaw'  # r=2, period=0.5, loss=weighted NLLL
# 	state = TrainingState(name, model=model)
# 	model.eval()
# 	
# 	#%% get data
# 	r: int=2  # cells per meter
# 	grid = Grid((-6, -6), 0, (12*r, 12*r), 1/r)
# 	period = 0.5
# 	nin = 10
# 	nout = 20
# 	batch = 1
# 	dataset = 'data/alex1'
# 	valid_data = DataLoader(SeqDataset(GridFromH5Dataset(f'{dataset}_valid.h5', grid, period, random_rotation=False), nin, nout), batch, shuffle=False)
# 	data_iter = iter(valid_data)
# 	
# 	#%% set up planning (once at start)
# 	predictor = Predictor(model)
# 	graph = InfiniteGraph(1000, grid, period, predictor)
# 	goal_idx = graph.nearest(torch.as_tensor([-6,+6]))
# 	goal_p = graph.position(goal_idx)
# 	graph.goal = goal_p
# 	def goal_condition(node: graph.Node):
# 		return (node.idx % node.graph.n) == goal_idx
# 	
# 	#%% set up planning (each re-plan)
# 	data_in, data_out = next(data_iter)
# 	data_in.transpose_(0, 1)  # todo remove the need for this nonsense
# 	data_out.transpose_(0, 1)InfiniteGraph
# 	predictor.inform(data_in)
# 	# predictor.data = data_out.squeeze(dim=1)
# 	# predictor.data = torch.ones(1,4,*grid.shape).expand(1000,-1,-1,-1)
# 	
# 	start_idx = graph.nearest(torch.as_tensor([0,+6,-6]))
# 	start_node = graph.Node(start_idx, cost=0)
# 	
# 	goal_node = dijkstra(start_node, goal_condition, verbose=True)
# 	
# 	
# 	waypoints = [goal_node]
# 	while (prev_node:=waypoints[-1].parent) is not None:
# 		waypoints.append(prev_node)
# 	
# 	print(f"{len(waypoints)=}")
# 	
# 	import matplotlib as mpl
# 	mpl.use('TkAgg')
# 	from matplotlib import pyplot as plt
# 	from mpl_toolkits.mplot3d import Axes3D
# 	
# 	(xs, ys, zs) = zip(*((n.position[0], n.position[1], n.time) for n in waypoints))
# 	fig = plt.figure()
# 	ax = Axes3D(fig)
# 	h = ax.scatter3D(xs,ys,zs)
# 	plt.show()
# 	
# 	exit()
