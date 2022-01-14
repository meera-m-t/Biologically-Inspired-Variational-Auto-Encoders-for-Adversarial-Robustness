import os
import numpy as np
import pandas as pd
from skimage import io
from torch.utils.data import Dataset
import torch
from torchvision import transforms
import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from torchvision import transforms
from tqdm import tqdm

from time import time as t
import torch
import os
import numpy as np

def single(
    datum: torch.Tensor,
    time: int,
    **kwargs
) -> torch.Tensor:
    # language=rst
    sparsity=0.5
    dt=1.0
    """
    Generates timing based single-spike encoding. Spike occurs earlier if the
    intensity of the input feature is higher. Features whose value is lower than
    threshold is remain silent.
    :param datum: Tensor of shape ``[n_1, ..., n_k]``.
    :param time: Length of the input and output.
    :param dt: Simulation time step.
    :param sparsity: Sparsity of the input representation. 0 for no spikes and 1 for all
        spikes.
    :return: Tensor of shape ``[time, n_1, ..., n_k]``.
    """
    time = int(time / dt)
    shape = list(datum.shape)
    datum = np.copy(datum)
    quantile = np.quantile(datum, 1 - sparsity)
    s = np.zeros([time, *shape])
    s[0] = np.where(datum > quantile, np.ones(shape), np.zeros(shape))
    return torch.Tensor(s).byte()

def repeat(datum: torch.Tensor, time: int, **kwargs) -> torch.Tensor:
    # language=rst
    """
    :param datum: Repeats a tensor along a new dimension in the 0th position for
        ``int(time / dt)`` timesteps.
    :param time: Tensor of shape ``[n_1, ..., n_k]``.
    :param dt: Simulation time step.
    :return: Tensor of shape ``[time, n_1, ..., n_k]`` of repeated data along the 0-th
        dimension.
    """
    time = int(time / dt)
    return datum.repeat([time, *([1] * len(datum.shape))])


def bernoulli(
    datum: torch.Tensor,
    time: Optional[int] = None,


    **kwargs
) -> torch.Tensor:
    # language=rst
    """
    Generates Bernoulli-distributed spike trains based on input intensity. Inputs must
    be non-negative. Spikes correspond to successful Bernoulli trials, with success
    probability equal to (normalized in [0, 1]) input value.
    :param datum: Tensor of shape ``[n_1, ..., n_k]``.
    :param time: Length of Bernoulli spike train per input variable.
    :param dt: Simulation time step.
    :return: Tensor of shape ``[time, n_1, ..., n_k]`` of Bernoulli-distributed spikes.
    Keyword arguments:
    :param float max_prob: Maximum probability of spike per Bernoulli trial.
    """
    # Setting kwargs.
    max_prob = kwargs.get("max_prob", 1.0)

    assert 0 <= max_prob <= 1, "Maximum firing probability must be in range [0, 1]"
    assert (datum >= 0).all(), "Inputs must be non-negative"

    shape, size = datum.shape, datum.numel()
    datum = datum.flatten()

    if time is not None:
        time = int(time /1.0)

    # Normalize inputs and rescale (spike probability proportional to input intensity).
    if datum.max() > 1.0:
        datum /= datum.max()

    # Make spike data from Bernoulli sampling.
    if time is None:
        spikes = torch.bernoulli(max_prob * datum).to(device)
        spikes = spikes.view(*shape)
    else:
        spikes = torch.bernoulli(max_prob * datum.repeat([time, 1]))
        spikes = spikes.view(time, *shape)

    return spikes.byte()
def poisson( datum: torch.Tensor, time: int, **kwargs
) -> torch.Tensor:
    dt=1.0
    """
    Generates Poisson-distributed spike trains based on input intensity. Inputs must be
    non-negative, and give the firing rate in Hz. Inter-spike intervals (ISIs) for
    non-negative data incremented by one to avoid zero intervals while maintaining ISI
    distributions.
    :param datum: Tensor of shape ``[n_1, ..., n_k]``.
    :param time: Length of Poisson spike train per input variable.
    :param dt: Simulation time step.
    :return: Tensor of shape ``[time, n_1, ..., n_k]`` of Poisson-distributed spikes.
    """
    assert (datum >= 0).all(), "Inputs must be non-negative"

    # Get shape and size of data.
    shape, size = datum.shape, datum.numel()
    datum = datum.flatten()
    time = int(time / dt)

    # Compute firing rates in seconds as function of data intensity,
    # accounting for simulation time step.
    rate = torch.zeros(size)
    rate[datum != 0] = 1 / datum[datum != 0] * (1000 / dt)

    # Create Poisson distribution and sample inter-spike intervals
    # (incrementing by 1 to avoid zero intervals).
    dist = torch.distributions.Poisson(rate=rate)
    intervals = dist.sample(sample_shape=torch.Size([time + 1]))
    intervals[:, datum != 0] += (intervals[:, datum != 0] == 0).float()

    # Calculate spike times by cumulatively summing over time dimension.
    times = torch.cumsum(intervals, dim=0).long()
    times[times >= time + 1] = 0

    # Create tensor of spikes.
    spikes = torch.zeros(time + 1, size).byte()
    spikes[times, torch.arange(size)] = 1
    spikes = spikes[1:]

    return spikes.view(time, *shape)


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  
from torchvision import transforms
import PIL 
for i in range(32):

    img=Image.open("./input/imginp%s.png" % str(i) )
    #img = img.resize((64, 64)) 
    trans1 = transforms.ToTensor()
    trans=transforms.ToPILImage()
    img=trans1((img))
    img= img
    img=single(img, time=100)

    img = img[0,:, :, :]

    img=trans(img[0])
    plt.imshow(img)

    plt.imsave("./spiking/spike%s.png" % str(i), np.array(img),format='png', cmap='gray' )