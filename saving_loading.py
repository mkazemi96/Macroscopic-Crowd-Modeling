# Standard Library
from datetime import datetime
from functools import cached_property
from glob import glob
from math import inf
import os
from os import PathLike
from pathlib import Path
from sys import stderr
import time

# PyPI
import git
import petname
import torch
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.tensorboard import SummaryWriter

# Local
from .config import cfg
from .tools import Exporter, doing
from .hickle.loaders import load_torch
# load_torch.manual_register()

export = Exporter()


# # HDF5 things
# 
# def h5_insert_dict(h: h5py.Group, d: dict):
# 	for k,v in d.items():
# 		if isinstance(v,dict):
# 			h.create_group(k, True)  # python3.7+ dicts are ordered
# 			h5_insert_dict(h[k], v)
# 		else:
# 			h.create_dataset(k, data=v)
# 
# def h5_extract_dict(h: h5py.Group, d: dict=None) -> dict:
# 	if d is None: d = {}
# 	for k,v in h.items():
# 		if isinstance(v, h5py.Group):
# 			d[k] = h5_extract_dict(h[k])
# 		else:
# 			d[k] = h[k]
# 	return d


@export
class TrainingState:
	
	def __init__(self,
		file_glob: str=None,
		*,
		model: Module=None,
		optimizer: Optimizer=None,
		lr_scheduler: LRScheduler=None,
		directory=None,
	):
		self.model = model
		self.optimizer = optimizer
		self.lr_scheduler = lr_scheduler
		self.state = {}
		self.start_time = None
		self.dir = Path(directory) if directory else Path()
		
		try:
			self.repo = git.Repo(path=self.dir, search_parent_directories=True)
		except git.InvalidGitRepositoryError as e:
			print("No git repository found at", e, file=stderr)
			self.repo = None
		
		if file_glob is None:
			self.init_state()
		else:
			self.load(file_glob)
	
	
	_state_settable = {'steps', 'epochs'}
	_state_gettable = {*_state_settable, 'name', 'commit', 'config', 'born'}
	
	def __getattr__(self, key):
		if key in self._state_gettable:
			return self.state[key]
		else:
			raise AttributeError
	
	def __setattr__(self, key, value):

		if key in self._state_settable:
			# print(f"Setting self.state[{key}] to: {value} (type: {type(value)})")
			self.state[key] = value
		else:
			super().__setattr__(key, value)
	
	
	@cached_property
	def writer(self):
		return SummaryWriter(self.dir/'runs'/self.name, purge_step=self.steps+1)
	
	
	@property
	def loss(self): return self.state['loss']  # get as normal
	@loss.setter
	def loss(self, value):
		self.state['loss'] = value  # set as normal
		if value < self.best_loss:  # save if better
			self.save(best=True)
			self.best_loss = value
	
	@cached_property
	def best_loss(self):
		file: Path = self._file_name(best=True)
		if file.is_file():
			try:
				return TrainingState(file).loss
			except Exception:
				pass
		return +inf
	
	
	def init_state(self):
		"""Start a new model."""
		
		self.state = {}
		self.state['name'] = petname.generate()
		print(f"New model: {self.name}")
		
		if self.repo:
			tag = self.repo.create_tag(f'model/{self.name}')
			self.state['commit'] = tag.commit.hexsha
			if self.repo.is_dirty():
				self.state['commit'] += '-dirty'
		
		self.state['config'] = vars(cfg)
		self.state['born'] = str(datetime.now())
		self.start_time = time.time()
		self.state['age'] = 0
		self.state['steps'] = 0
		self.state['epochs'] = 0
		self.state['loss'] = +inf
	
	
	def load(self, file_glob=None):
		file = self._file_glob(file_glob)
		self._load_file(file)
		self._check_git()
		self._check_cfg()
		self._export_state()
	
	def save(self, *, best=False, err=False):
		self._suffix = self.__class__._suffix  # always save in preferable format
		self._import_state()
		file = self._file_name(best=best, err=err)
		self._save_file(file)
	_suffix = '.hkl'  # default load format, save format
	
	
	#%% state from/to objects
	def _import_state(self):
		"""Record state into self. Note: does not save to file."""
		# metadata
		self.state['age'] += (time.time() - self.start_time)
		self.state['date'] = str(datetime.now())
		# state
		self.state['model'] = self.model.state_dict() if self.model else None
		filename = f"{self.name}_train_model_3D.pth"
		torch.save({
			'model': self.model.state_dict(),
			'optimizer': self.optimizer.state_dict(),
			'lr_scheduler': self.lr_scheduler.state_dict(),
			'epoch': self.epochs,
			'loss': self.loss,
		}, filename)
		self.state['optimizer'] = self.optimizer.state_dict() if self.optimizer else None
		self.state['lr_scheduler'] = self.lr_scheduler.state_dict() if self.lr_scheduler else None
	
	def _export_state(self):
		"""Export state into objects that need them."""
		
		if self.model: self.model.load_state_dict(self.state['model'])
		if self.optimizer: self.optimizer.load_state_dict(self.state['optimizer'])
		if self.lr_scheduler: self.lr_scheduler.load_state_dict(self.state['lr_scheduler'])
		self.start_time = time.time()
	
	
	#%% state from/to file
	def _load_file(self, file: Path): return self._loadsave_file('load', file)
	def _save_file(self, file: Path): return self._loadsave_file('save', file)
	
	def _loadsave_file(self, saveload, file: Path):
		suff = ''.join(file.suffixes)
		name = f"_{saveload}{suff.replace('.','_')}_file"
		try:
			meth = getattr(self, name)
		except AttributeError as e:
			raise NotImplementedError(f"Unknown format, suffix {suff}") from e
		else:
			verbing = dict(load="Loading", save="Saving").get(saveload, "Doing")
			with doing(f"{verbing} {file}"):
				return meth(file)
	
	
	#%% hdf5 format
	def _load_h5_file(self, file: PathLike):
		raise NotImplementedError("HDF5 loading not working yet.")
		# import h5py
		# with h5py.File(file, 'r') as h:
		# 	self.state = h5_extract_dict(h)
		import h5py_wrapper as h5w
		self.state = h5w.load(file)
	
	def _save_h5_file(self, file: PathLike):
		raise NotImplementedError("HDF5 saving not working yet.")
		# make sure dates are strings
		for k in ('born','date'):
			if isinstance(self.state[k], datetime):
				self.state[k] = self.state[k].isoformat()
		# import h5py
		# with h5py.File(file, 'w') as h:
		# 	h5_insert_dict(h, self.state)
		import h5py_wrapper as h5w
		h5w.save(file, self.state, 'w')
	
	
	#%% hickle format
	def _load_hkl_file(self, file: PathLike):
		import hickle
		self.state = hickle.load(file)

	def _save_hkl_file(self, file: PathLike):
		import hickle
		file_path = Path(file)  # Convert to Path object if not already
		file_path.parent.mkdir(parents=True, exist_ok=True)
		# Debugging: Log self.state
		print(f"self.state before saving: (type: {type(self.state)})")

		# Ensure self.state is a dictionary
		if not isinstance(self.state, dict):
			raise TypeError(f"self.state must be a dictionary, but got {type(self.state)}")

		hickle.dump(self.state, file)
	
	
	#%% pytorch pkl format
	def _load_pt_pkl_file(self, file: PathLike):
		import torch
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # todo global variable?
		self.state = torch.load(file, map_location=device)
	
	def _save_pt_pkl_file(self, file: PathLike):
		import torch
		torch.save(self.state, file)
	
	
	#%% utilities
	def _file_name(self, *, best: bool=False, err: bool=False) -> Path:
		stamp = 'best' if best else f'{self.steps:07}'
		err_tag = '_error' if err else ''
		return self.dir/'checkpoints'/f'{self.name}_{stamp}{err_tag}{self._suffix}'
	
	
	def _file_glob(self, file_glob: str=None) -> Path:
		if file_glob is None:
			file_glob = ''
		else:
			file_glob = str(file_glob)
		if '/' not in file_glob:
			file_glob = str(self.dir/'checkpoints'/(file_glob+'*'+self._suffix))
		files = glob(file_glob)
		if '_error' not in file_glob:
			files = [f for f in files if '_error' not in f]
		if files:
			file = max(files, key=os.path.getctime)  # newest
		else:
			raise FileNotFoundError(f"No files match {file_glob}")
		return Path(file)
	
	
	def _check_git(self):
		if self.repo and self.commit:
			file_hexsha, _, dirty = self.commit.partition('-')
			curr_hexsha = self.repo.head.commit.hexsha
			if file_hexsha != curr_hexsha:
				print(
					f"Checkpoint commit hash {file_hexsha}",
					f"   does not match HEAD {curr_hexsha}",
					sep='\n',
					file=stderr,
				)
			if dirty:
				print("Checkpoint commit is dirty.", file=stderr)
			if self.repo.is_dirty():
				print("HEAD is dirty.", file=stderr)
	
	
	def _check_cfg(self):
		if 'config' in self.state:
			cfg_dict = vars(cfg)
			do_update = False
			for key in cfg_dict:
				if key in self.config and self.config[key] != cfg_dict[key]:
					print(
						f"Checkpoint config {key} {self.config[key]}",
						f"   does not match {key} {cfg_dict[key]}",
						sep='\n',
						file=stderr,
					)
					do_update = True
			# if do_update:
			# 	print("Updating config to match model")
			# 	cfg_dict.update(self.config)


# Note: this is documentation (with syntax highlighting)
# saved files and their (dict) contents
saved_files = {
	# regular training save
	'pet-name_0009000.pt.pkl': {
		# metadata (model-specific)
		'name',  # pet name, unique per model initialisation
		'born',  # date of birth, celebrate yearly
		'commit',  # git commit (hexsha) at birth, possibly tagged '-dirty'
		'config',  # config (see config.py) at birth
		# metadata (state-specific)
		'age',  # training time since birth (seconds, wall-clock)
		'date',  # save date, last train
		'steps',  # global_step, preferably 1 per forward pass
		'epochs',  # number of iterations through the dataset
		'loss',  # latest validation loss
		# state
		'model',  # model state_dict
		'optimizer',  # optimizer state_dict
		'lr_scheduler',  # learning rate scheduler state_dict
	},
	# the best state (so far) of this model, according to validation loss
	'pet-name_best.pt.pkl': {...},  # as above
	# panic save, stop on exception, may have errors
	'pet-name_0009999_error.pt.pkl': {...},  # as above
}


if __name__ == '__main__':
	state = TrainingState.__new__(TrainingState)
	state.repo = None
	state.dir = Path()
	# file = './cluster/checkpoints/strong-newt_best.pt.pkl'
	file = './cluster/checkpoints/valid-llama_best.pt.pkl'
	state._load_file(Path(file))
	state._save_file(Path(file.replace('.pt.pkl', '.hkl')))
