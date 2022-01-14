
import torch
import torch.nn as nn
import torch.nn.functional as fn
from . import functional as sf
from torch.nn.parameter import Parameter
from .utils import to_pair

class Convolution(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, weight_mean=0.8, weight_std=0.02):
		super(Convolution, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = to_pair(kernel_size)
		#self.weight_mean = weight_mean
		#self.weight_std = weight_std

		# For future use
		self.stride = stride
		self.bias = bias
		self.dilation = 1
		self.groups = 1
		self.padding = padding

		# Parameters
		self.weight = Parameter(torch.Tensor(self.out_channels, self.in_channels, *self.kernel_size))
		self.weight.requires_grad_(False) # We do not use gradients
		self.reset_weight(weight_mean, weight_std)

	def reset_weight(self, weight_mean=0.8, weight_std=0.02):
		"""Resets weights to random values based on a normal distribution.
		Args:
			weight_mean (float, optional): Mean of the random weights. Default: 0.8
			weight_std (float, optional): Standard deviation of the random weights. Default: 0.02
		"""
		self.weight.normal_(weight_mean, weight_std)

	def load_weight(self, target):
		"""Loads weights with the target tensor.
		Args:
			target (Tensor=): The target tensor.
		"""
		self.weight.copy_(target)	

	def forward(self, input):
		return fn.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
##################################################################
class BatchNormalization2D(nn.Module):
	def __init__(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
		super(BatchNormalization2D, self).__init__()
		self.num_features= num_features	
	def forward(self, input):
		return nn.BatchNorm2d(input, self.num_features)

class BatchNormalization1D(nn.Module):
	def __init__(num_features,, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
		super(BatchNormalization1D, self).__init__()
		self.num_features= num_features	
	def forward(self, input):
		return nn.BatchNorm1d( input,self.num_features)
##########################################################
class linear(nn.Module):
	def __init__(self,in_features, out_features, bias= True, weight_mean=0.8, weight_std=0.02):
		super(linear, self).__init__()
		self.in_features= in_features
		self.out_features = out_features
		self.bias=bias
		self.reset_weight(weight_mean, weight_std)

	def reset_weight(self, weight_mean=0.8, weight_std=0.02):
		"""Resets weights to random values based on a normal distribution.
		Args:
			weight_mean (float, optional): Mean of the random weights. Default: 0.8
			weight_std (float, optional): Standard deviation of the random weights. Default: 0.02
		"""
		self.weight.normal_(weight_mean, weight_std)

	def load_weight(self, target):
		"""Loads weights with the target tensor.
		Args:
			target (Tensor=): The target tensor.
		"""
		self.weight.copy_(target)	

	def forward(self, input):
		return nn.Linear( input,self.out_features, self.in_features, self.bias)

#################################################################################################################


class ConvolutionTranspose2D(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, weight_mean=0.8, weight_std=0.02):
		super(ConvolutionTranspose2D, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = to_pair(kernel_size)
		#self.weight_mean = weight_mean
		#self.weight_std = weight_std

		# For future use
		self.stride = stride
		self.bias = bias
		self.dilation = 1
		self.groups = 1
		self.padding = padding

		# Parameters
		self.weight = Parameter(torch.Tensor(self.out_channels, self.in_channels, *self.kernel_size))
		self.weight.requires_grad_(False) # We do not use gradients
		self.reset_weight(weight_mean, weight_std)

	def reset_weight(self, weight_mean=0.8, weight_std=0.02):
		"""Resets weights to random values based on a normal distribution.
		Args:
			weight_mean (float, optional): Mean of the random weights. Default: 0.8
			weight_std (float, optional): Standard deviation of the random weights. Default: 0.02
		"""
		self.weight.normal_(weight_mean, weight_std)

	def load_weight(self, target):
		"""Loads weights with the target tensor.
		Args:
			target (Tensor=): The target tensor.
		"""
		self.weight.copy_(target)	

	def forward(self, input):
		return fn.ConvTranspose2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)





################################################################################################################
class mSTDP(nn.Module):
	def __init__(self, conv_layer, learning_rate, use_stabilizer = True, lower_bound = 0, upper_bound = 1):
		super(mSTDP, self).__init__()
		self.conv_layer = conv_layer
		if isinstance(learning_rate, list):
			self.learning_rate = learning_rate
		else:
			self.learning_rate = [learning_rate] * conv_layer.out_channels
		for i in range(conv_layer.out_channels):
			self.learning_rate[i] = (Parameter(torch.tensor([self.learning_rate[i][0]])),
							Parameter(torch.tensor([self.learning_rate[i][1]])))
			self.register_parameter('ltp_' + str(i), self.learning_rate[i][0])
			self.register_parameter('ltd_' + str(i), self.learning_rate[i][1])
			self.learning_rate[i][0].requires_grad_(False)
			self.learning_rate[i][1].requires_grad_(False)
		self.use_stabilizer = use_stabilizer
		self.lower_bound = lower_bound
		self.upper_bound = upper_bound

	def get_visable_hidden_ordering(self, input_spikes, output_spikes, winners):
		# accumulating input and output spikes to get latencies
		input_latencies = torch.sum(input_spikes, dim=0)
		output_latencies = torch.sum(output_spikes, dim=0)
		result = []
		for winner in winners:
			out_tensor = torch.ones(*self.conv_layer.kernel_size, device=output_latencies.device) * output_latencies[winner]
			in_tensor = input_latencies[:,winner[-2]:winner[-2]+self.conv_layer.kernel_size[-2],winner[-1]:winner[-1]+self.conv_layer.kernel_size[-1]]
			result.append(torch.ge(in_tensor,out_tensor))
		return result

	def forward(self, input_spikes, potentials, output_spikes, winners=None, kwta = 1, inhibition_radius = 0):
		if winners is None:
			winners = sf.get_k_winners(potentials, kwta, inhibition_radius, output_spikes)
		pairings = self.get_visable_hidden_ordering(input_spikes, output_spikes, winners)
		
		lr = torch.zeros_like(self.conv_layer.weight)
		for i in range(len(winners)):
			f = winners[i][0]
			lr[f] = torch.where(pairings[i], *(self.learning_rate[f]))

		self.conv_layer.weight += lr * ((self.conv_layer.weight-self.lower_bound) * (self.upper_bound-self.conv_layer.weight) if self.use_stabilizer else 1)
		self.conv_layer.weight.clamp_(self.lower_bound, self.upper_bound)

	def update_learning_rate(self, feature, ap, an):
		self.learning_rate[feature][0][0] = ap
		self.learning_rate[feature][1][0] = an

	def update_all_learning_rate(self, ap, an):
		for feature in range(self.conv_layer.out_channels):
			self.learning_rate[feature][0][0] = ap
			self.learning_rate[feature][1][0] = an