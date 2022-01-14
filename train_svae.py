import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from typing import Optional
from time import time as t
import torch
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.nn.parameter import Parameter
import torchvision
import numpy as np
import snn
import functional as sf
import visualization as vis
import utils
from torchvision import transforms
import struct
import glob
import svae_model
use_cuda = True
base_savedir = 'mnist/'

####################################################################################################
def bernoulli(
    datum: torch.Tensor,
    time: Optional[int] = None,
    dt: float = 1.0,
    device="cpu",
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
    time=50
    max_prob = kwargs.get("max_prob", 1.0)

    assert 0 <= max_prob <= 1, "Maximum firing probability must be in range [0, 1]"
    assert (datum >= 0).all(), "Inputs must be non-negative"

    shape, size = datum.shape, datum.numel()
    datum = datum.flatten()

    if time is not None:
        time = int(time / dt)

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

intensity=128
batch_size=265
s1 = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
####################################################################################################
### read adv:adv_x.npy and true:xt.npy label:yt.npy
savedir = base_savedir + 'AdvSave/multiattack/model_{}/cnnEpochs_{}/multi_{}/'.format(args.cnn_model, args.cnn_epochs, args.multi)
adv_x = np.load(savedir + 'adv_x.npy')
adv_x = adv_x.transpose(0, 3, 1, 2)
adv_x = torch.tensor(adv_x)

xt = np.load(savedir + 'xt.npy')
xt = xt.transpose(0, 3, 1, 2)
xt = torch.tensor(xt).type(torch.FloatTensor)
yt = np.load(savedir + 'yt.npy')
yt = torch.tensor(yt)
yt = yt.to(device

train_data = torch.utils.data.CustomTensorDataset(tensors=(adv_x,xt),transform = bernoulli(s1))
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
####################################################################################################

#TRAIN SVAE

model = SVAE18()

def train(epoch, savedir):
    model.train()
    train_loss = 0

    loss_ll = []
    loss_B = []
    loss_K = []
    for batch_idx, (adv_batch, clean_batch) in enumerate(train_data_loader):
        adv_batch = adv_batch.to(device)
        clean_batch = clean_batch.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(adv_batch)

        r = torch.tensor(r_list[epoch - 1]).to(device)
        loss, BCE, KLD = loss_lambda(recon_batch, clean_batch, mu, logvar, r)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(adv_batch), len(train_data_loader.dataset),
                       100. * batch_idx / len(train_data_loader), loss.item() / len(adv_batch)))
        loss_a = loss.item() / len(adv_batch)
        loss_ll.append(loss_a)

        loss_b = BCE.item() / len(adv_batch)
        loss_k = KLD.item() / len(adv_batch)
        loss_B.append(loss_b)
        loss_K.append(loss_k)

    loss_norm = train_loss / len(train_data_loader.dataset)

    torch.save(model.state_dict(), savedir + 'model{}.pth'.format(epoch))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, loss_norm))
    return loss_ll, loss_B, loss_K

####################################################################################################




