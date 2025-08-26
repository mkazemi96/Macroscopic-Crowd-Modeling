import torch
import numpy as np
# from hickle.helpers import get_type_and_data, pickle
import pickle
import hickle.lookup
from hickle.lookup import LoaderManager

# data
# dtype: torch.dtype = None
# device: torch.device = None
# requires_grad: bool = False

def create_torch_tensor(py_obj, h_group, name, **kwargs):
	assert py_obj.layout == torch.strided

	# tensor data
	d = h_group.create_dataset(name, data=py_obj.detach().cpu().numpy(), **kwargs)

	# attributes
	d.attrs['torch_dtype'] = bytes(repr(py_obj.dtype), 'ascii')
	d.attrs['requires_grad'] = py_obj.requires_grad

	# grad
	if py_obj.grad is not None:
		grad_name = '%s_grad' % name
		g = create_torch_tensor(py_obj.grad, h_group, grad_name, **kwargs)
		g.attrs['base_type'] = b'torch_tensor_grad'
		g.attrs['type'] = np.array(pickle.dumps(py_obj.grad.__class__))
		d.attrs['grad'] = g.ref

	return d


def load_torch_tensor(h_node):
	# _, _, data = get_type_and_data(h_node)
	data = h_node[()]

	# dtype
	torch_str, dtype_str = h_node.attrs['torch_dtype'].decode('ascii').split('.')
	assert torch_str == 'torch'
	dtype = getattr(torch, dtype_str)

	# requires_grad
	requires_grad = bool(h_node.attrs['requires_grad'])

	# tensor
	py_obj = torch.as_tensor(data, dtype=dtype)
	py_obj.requires_grad = requires_grad

	# gradient
	if (k:='grad') in (a:=h_node.attrs) and (ref:=a[k]):
		py_obj.grad = load_torch_tensor(h_node[ref])

	return py_obj


def check_torch_tensor(py_obj):
	return isinstance(py_obj, torch.Tensor)


class_register = [
	# (myclass_type, hkl_str, dump_function, load_function, ndarray_check_fn=None, to_sort=True),
	(torch.Tensor, b'torch_tensor', create_torch_tensor, load_torch_tensor, check_torch_tensor),
]
exclude_register = [
	# hkl_str,
	b'torch_tensor_grad',
]
# def manual_register():
# 	# from hickle.lookup.LoaderManager import loaded_loaders, register_class, register_class_exclude
# 	for args in class_register: LoaderManager.register_class(*args)
# 	for arg in exclude_register: LoaderManager.register_class_exclude(arg)
# 	(LoaderManager.__loaded_loaders__.add('hickle.loaders.load_torch'))
	# loaded_loaders.append()


# def manual_register():
# 	# from hickle.lookup import loaded_loaders, register_class, register_class_exclude
# 	# for args in class_register: register_class(*args)
# 	# for arg in exclude_register: register_class_exclude(arg)
# 	# loaded_loaders.append('hickle.loaders.load_torch')
# 	for args in class_register: LoaderManager.register_class(*args)
# 	for arg in exclude_register: LoaderManager.register_class_exclude(arg)
# 	# LoaderManager.load_loader('hickle.loaders.load_torch')
# 	# # # Accessing the loaded_loaders from the LoaderManager class
# 	# if 'hickle.loaders.load_torch' not in LoaderManager.__loaded_loaders__:
# 	# 	# Add the loader to the loaded_loaders set
# 	# 	LoaderManager.__loaded_loaders__.add('hickle.loaders.load_torch')
