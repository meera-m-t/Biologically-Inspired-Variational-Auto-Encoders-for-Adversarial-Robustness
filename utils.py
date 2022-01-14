import torch
import torch.nn.functional as fn
import numpy as np
import math
from torchvision import transforms
from torchvision import datasets
import os

def to_pair(data):
	r"""Converts a single or a tuple of data into a pair. If the data is a tuple with more than two elements, it selects
	the first two of them. In case of single data, it duplicates that data into a pair.
	Args:
		data (object or tuple): The input data.
	Returns:
		Tuple: A pair of data.
	"""
	if isinstance(data, tuple):
		return data[0:2]
	return (data, data)

def generate_inhibition_kernel(inhibition_percents):

	inhibition_kernel = torch.zeros(2*len(inhibition_percents)+1, 2*len(inhibition_percents)+1).float()
	center = len(inhibition_percents)
	for i in range(2*len(inhibition_percents)+1):
		for j in range(2*len(inhibition_percents)+1):
			dist = int(max(math.fabs(i - center), math.fabs(j - center)))
			if dist != 0:
				inhibition_kernel[i,j] = inhibition_percents[dist - 1]
	return inhibition_kernel

def tensor_to_text(data, address):

	f = open(address, "w")
	data_cpu = data.cpu()
	shape = data.shape
	print(",".join(map(str, shape)), file=f)
	data_flat = data_cpu.view(-1).numpy()
	print(",".join(data_flat.astype(np.str)), file=f)
	f.close()

def text_to_tensor(address, type='float'):

	f = open(address, "r")
	shape = tuple(map(int, f.readline().split(",")))
	data = np.array(f.readline().split(","))
	if type == 'float':
		data = data.astype(np.float32)
	elif type == 'int':
		data = data.astype(np.int32)
	else:
		raise ValueError("type must be 'int' or 'float'")
	data = torch.from_numpy(data)
	data = data.reshape(shape)
	f.close()
	return data

class LateralIntencityInhibition:

	def __init__(self, inhibition_percents):
		self.inhibition_kernel = generate_inhibition_kernel(inhibition_percents)
		self.inhibition_kernel.unsqueeze_(0).unsqueeze_(0)


	def intensity_lateral_inhibition(self, intencities):
		intencities.squeeze_(0)
		intencities.unsqueeze_(1)

		inh_win_size = self.inhibition_kernel.size(-1)
		rad = inh_win_size//2

		values = intencities.reshape(intencities.size(0),intencities.size(1),-1,1)
		values = values.repeat(1,1,1,inh_win_size)
		values = values.reshape(intencities.size(0),intencities.size(1),-1,intencities.size(-1)*inh_win_size)
		values = values.repeat(1,1,1,inh_win_size)
		values = values.reshape(intencities.size(0),intencities.size(1),-1,intencities.size(-1)*inh_win_size)

		padded = fn.pad(intencities,(rad,rad,rad,rad))

		patches = padded.unfold(-1,inh_win_size,1)
		patches = patches.reshape(patches.size(0),patches.size(1),patches.size(2),-1,patches.size(3)*patches.size(4))
		patches.squeeze_(-2)

		patches = patches.unfold(-2,inh_win_size,1).transpose(-1,-2)
		patches = patches.reshape(patches.size(0),patches.size(1),1,-1,patches.size(-1))
		patches.squeeze_(-3)

		coef = values - patches
		coef.clamp_(min=0).sign_() 

		factors = fn.conv2d(coef, self.inhibition_kernel, stride=inh_win_size)
		result = intencities + intencities * factors

		intencities.squeeze_(1)
		intencities.unsqueeze_(0)
		result.squeeze_(1)
		result.unsqueeze_(0)
		return result

	def __call__(self,input):
		return self.intensity_lateral_inhibition(input)



class Intensity2Latency:

	def __init__(self, number_of_spike_bins, to_spike=False):
		self.time_steps = number_of_spike_bins
		self.to_spike = to_spike

	def intensity_to_latency(self, intencities):
		#bins = []
		bins_intencities = []
		nonzero_cnt = torch.nonzero(intencities).size()[0]

		#check for empty bins
		bin_size = nonzero_cnt//self.time_steps

		#sort
		intencities_flattened = torch.reshape(intencities, (-1,))
		intencities_flattened_sorted = torch.sort(intencities_flattened, descending=True)

		#bin packing
		sorted_bins_value, sorted_bins_idx = torch.split(intencities_flattened_sorted[0], bin_size), torch.split(intencities_flattened_sorted[1], bin_size)

		#add to the list of timesteps
		spike_map = torch.zeros_like(intencities_flattened_sorted[0])
	
		for i in range(self.time_steps):
			spike_map.scatter_(0, sorted_bins_idx[i], sorted_bins_value[i])
			spike_map_copy = spike_map.clone().detach()
			spike_map_copy = spike_map_copy.reshape(tuple(intencities.shape))
			bins_intencities.append(spike_map_copy.squeeze(0).float())
			#bins.append(spike_map_copy.sign().squeeze_(0).float())
	
		return torch.stack(bins_intencities)#, torch.stack(bins)
		#return torch.stack(bins)

	def __call__(self, image):
		if self.to_spike:
			return self.intensity_to_latency(image).sign()
		return self.intensity_to_latency(image)



class CacheDataset(torch.utils.data.Dataset):

	def __init__(self, dataset, cache_address=None):
		self.dataset = dataset
		self.cache_address = cache_address
		self.cache = [None] * len(self.dataset)

	def __getitem__(self, index):
		if self.cache[index] is None:
			#cache it
			sample, target = self.dataset[index]
			if self.cache_address is None:
				self.cache[index] = sample, target
			else:
				save_path = os.path.join(self.cache_address, str(index))
				torch.save(sample, save_path + ".cd")
				torch.save(target, save_path + ".cl")
				self.cache[index] = save_path
		else:
			if self.cache_address is None:
				sample, target = self.cache[index]
			else:
				sample = torch.load(self.cache[index] + ".cd")
				target = torch.load(self.cache[index] + ".cl")
		return sample, target

	def reset_cache(self):
		r"""Clears the cached data. It is useful when you want to change a pre-processing parameter during
		the training process.
		"""
		if self.cache_address is not None:
			for add in self.cache:
				os.remove(add + ".cd")
				os.remove(add + ".cl")
		self.cache = [None] * len(self)

	def __len__(self):
		return len(self.dataset)