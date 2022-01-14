from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from torch.nn import functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.nn.parameter import Parameter
import torchvision
import numpy as np
import snn
import functional as sf
import utils
from torchvision import transforms
import struct
import glob
import pytorch_spiking
import torch.optim as optim
import vae_model
####################################################################################
#load weight from varitional auto encoder (VAE)

PATH = "VAE_model.pth"

# Save
torch.save(net.state_dict(), PATH)

# Load
model = VAE18()
model=model.load_state_dict(torch.load(PATH))

#encoder
vae_conv0=(model.encoder.layers[0].weight)
Gauss = lambda x, mu, sigma: mu + np.sqrt(2)*sigma*erfinv(2*vae_conv0-1)
vae_conv0=Gauss(vae_conv0, 0.8, 0.05)


vae_conv1=(model.encoder.layers[1].weight)
Gauss = lambda x, mu, sigma: mu + np.sqrt(2)*sigma*erfinv(2*vae_conv1-1)
vae_conv1=Gauss(vae_conv1, 0.8, 0.05)



vae_conv2=(model.encoder.layers[2].weight)
Gauss = lambda x, mu, sigma: mu + np.sqrt(2)*sigma*erfinv(2*vae_conv2-1)
vae_conv2=Gauss(vae_conv2, 0.8, 0.05)


vae_conv3=(model.encoder.layers[3].weight)
Gauss = lambda x, mu, sigma: mu + np.sqrt(2)*sigma*erfinv(2*vae_conv3-1)
vae_conv3=Gauss(vae_conv3, 0.8, 0.05)



vae_fc21=(model.encoder.layers[4].weight)
Gauss = lambda x, mu, sigma: mu + np.sqrt(2)*sigma*erfinv(2*vae_fc21-1)
vae_fc21=Gauss(vae_fc21, 0.8, 0.05)

vae_fc22=(model.encoder.layers[5].weight)
Gauss = lambda x, mu, sigma: mu + np.sqrt(2)*sigma*erfinv(2*vae_fc22-1)
vae_fc22=Gauss(vae_fc22, 0.8, 0.05)

############
#decoder
vae_fc3=(model.decoder.layers[0].weight)
Gauss = lambda x, mu, sigma: mu + np.sqrt(2)*sigma*erfinv(2*vae_fc3-1)
vae_fc3=Gauss(vae_fc3, 0.8, 0.05)

vae_deconv1=(model.decoder.layers[1].weight)
Gauss = lambda x, mu, sigma: mu + np.sqrt(2)*sigma*erfinv(2*vae_deconv1-1)
vae_deconv1=Gauss(vae_deconv1, 0.8, 0.05)


vae_deconv2=(model.decoder.layers[2].weight)
Gauss = lambda x, mu, sigma: mu + np.sqrt(2)*sigma*erfinv(2*vae_deconv2-1)
vae_deconv2=Gauss(vae_deconv2, 0.8, 0.05)



vae_deconv3=(model.decoder.layers[3].weight)
Gauss = lambda x, mu, sigma: mu + np.sqrt(2)*sigma*erfinv(2*vae_deconv3-1)
vae_deconv3=Gauss(vae_deconv3, 0.8, 0.05)


vae_deconv4=(model.decoder.layers[4].weight)
Gauss = lambda x, mu, sigma: mu + np.sqrt(2)*sigma*erfinv(2*vae_deconv4-1)
vae_deconv4=Gauss(vae_deconv4, 0.8, 0.05)



#################################################################################3
class SVAE18(nn.Module):
    def __init__(self):
        super(SVAE18, self).__init__()
        # SVAE Encoder
		self.sconv0 = snn.Convolution(1, 64, kernel_size=5, stride=1, padding=2, bias= False, weight_mean=0.8, weight_std=0.05)
		# each spiking convolution layer  take the weight from the convolution layer that has the same shape as the initial 
        self.sconv0.load_weight(vae_conv0) 
		self.sconv0_t = 36
		self.k0 = 5
		self.r0 = 3
        self.sconv0_bn = snn.BatchNormalization2D(64)
        self.sconv1 = snn.Convolution(64, 64, kernel_size=4, stride=2, padding=3, bias= False, weight_mean=0.8, weight_std=0.05)
        self.sconv1.load_weight(vae_conv1)       
		self.sconv1_t = 23
		self.k1 = 5
		self.r1 = 2       
        self.sconv1_bn = snn.BatchNormalization2D(64)
        self.sconv2 = snn.Convolution(64, 128, kernel_size=4, stride=2, padding=1, bias= False , weight_mean=0.8, weight_std=0.05)
        self.sconv2.load_weight(vae_conv2)         
		self.sconv2_t = 23
		self.k2 = 8
		self.r2 = 1            
        self.sconv2_bn = snn.BatchNormalization2D(128)
        self.sconv3 = snn.Convolution(128, 256, kernel_size=4, stride=2, padding=1, bias= False , weight_mean=0.8, weight_std=0.05)
        self.sconv3.load_weight(vae_conv3)       
		self.sconv3_t = 36       
		self.k3 = 1
		self.r3 = 0   

        self.sconv3_bn = snn.BatchNormalization2D(256)

        # Spiking Latent space
        self.fc21 = snn.linear(4096, 128, weight_mean=0.8, weight_std=0.05)
        self.fc21.load_weight(vae_fc21) 
        self.fc22 = snn.linear(4096, 128, weight_mean=0.8, weight_std=0.05)
        self.fc22.load_weight(vae_fc22)   

 ##############################################            
		self.mstdp0 = snn.mSTDP(self.sconv0, (0.004, -0.003))
		self.mstdp1 = snn.mSTDP(self.sconv1, (0.004, -0.003))
		self.mstdp2 = snn.mSTDP(self.sconv2, (0.004, -0.003))
		self.mstdp3 = snn.mSTDP(self.sconv3, (0.004, -0.003))        
		self.max_ap = Parameter(torch.Tensor([0.15]))
		self.dec_mstdp1 = snn.mSTDP(self.desconv1, (0.004, -0.003))
		self.dec_mstdp2 = snn.mSTDP(self.desconv2, (0.004, -0.003))
		self.dec_mstdp3 = snn.mSTDP(self.desconv3, (0.004, -0.003))   
		self.dec_mstdp4 = snn.mSTDP(self.desconv4, (0.004, -0.003))   		
		self.ctx = {"input_spikes":None, "potentials":None, "output_spikes":None, "winners":None}
		self.spk_cnt0 = 0
		self.spk_cnt1 = 0
		self.spk_cnt2 = 0	
		self.spk_cnt3 = 0	       
 ##############################################   

	def save_data(self, input_spike, potentials, output_spikes, winners):
		self.ctx["input_spikes"] = input_spike
		self.ctx["potentials"] = potentials
		self.ctx["output_spikes"] = output_spikes
		self.ctx["winners"] = winners
 ##############################################   
	def forward_encoder(self, input, max_layer):
		if self.training:
			pot = self.sconv0_bn(self.sconv0(input))
			spk, pot = sf.fire(pot, self.sconv0_t, True)
			self.spk_cnt0 += 1
			if self.spk_cnt0 >= 500:
				self.spk_cnt0 = 0
				ap = torch.tensor(self.mstdp0.learning_rate[0][0].item(), device=self.mstdp0.learning_rate[0][0].device) * 2
				ap = torch.min(ap, self.max_ap)
				an = ap * -0.75
				self.mstdp0.update_all_learning_rate(ap.item(), an.item())
			pot = sf.pointwise_inhibition(pot)
			spk = pot.sign()
			winners = sf.get_k_winners(pot, self.k0, self.r0, spk)
			self.save_data(input, pot, spk, winners)
				
        ##############################################   
			spk_in = sf.pointwise_inhibition(spk)
			pot = self.sconv1_bn(self.sconv1(spk_in))
			spk, pot = sf.fire(pot, self.sconv1_t, True)

			self.spk_cnt1 += 1
			if self.spk_cnt1 >= 500:
				self.spk_cnt1 = 0
				ap = torch.tensor(self.mstdp1.learning_rate[0][0].item(), device=self.mstdp1.learning_rate[0][0].device) * 2
				ap = torch.min(ap, self.max_ap)
				an = ap * -0.75
				self.mstdp1.update_all_learning_rate(ap.item(), an.item())                
			pot = sf.pointwise_inhibition(pot)
			spk = pot.sign()
			winners = sf.get_k_winners(pot, self.k1, self.r1, spk)
			self.save_data(spk_in, pot, spk, winners)
			
        ##############################################      
			spk_in = sf.pointwise_inhibition(spk)
			pot = self.sconv2_bn(self.sconv2(spk_in))
			spk, pot = sf.fire(pot, self.sconv2_t, True)

			self.spk_cnt2 += 1
			if self.spk_cnt2 >= 500:
				self.spk_cnt2 = 0
				ap = torch.tensor(self.mstdp2.learning_rate[0][0].item(), device=self.mstdp2.learning_rate[0][0].device) * 2
				ap = torch.min(ap, self.max_ap)
				an = ap * -0.75
				self.mstdp2.update_all_learning_rate(ap.item(), an.item())                   
			pot = sf.pointwise_inhibition(pot)
			spk = pot.sign()
			winners = sf.get_k_winners(pot, self.k2, self.r2, spk)
			self.save_data(spk_in, pot, spk, winners)
				
         ##############################################   

			spk_in = sf.pointwise_inhibition(spk)
			pot = self.sconv3_bn(self.sconv3(spk_in))
			spk,pot= sf.fire(pot, self.sconv3_t, True)
	
			pot = sf.pointwise_inhibition(pot)
			spk = pot.sign()
			winners = sf.get_k_winners(pot, self.k3, self.r3, spk)
			self.save_data(spk_in, pot, spk, winners)				
            spk = spk.view(spk.size(0), -1)
			return self.fc21(spk), self.fc22(spk)
		else:
			pot = sconv0_bn(self.sconv0(input)
			spk, pot = sf.fire(pot, self.sconv0_t, True)
			pot = sconv1_bn(self.sconv1(spk))
			spk, pot = sf.fire(pot, self.sconv1_t, True)
			pot = sconv2_bn(self.sconv2(spk))
			spk, pot = sf.fire(pot, self.sconv2_t, True)
			pot = sconv3_bn(self.sconv3(spk))
			spk, pot = sf.fire(pot, self.sconv3_t, True)
            spk = spk.view(spk.size(0), -1)            
			return self.fc21(spk), self.fc22(spk)
 ##############################################               
	
	def mstdp(self):

		self.mstdp0(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

		self.mstdp1(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

		self.mstdp2(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

		self.mstdp3(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])   

 ##############################################                

    def encoder(network, data):
        network.train()
		layer_idx=0
        for i in range(len(data)):
            data_in = data[i]
            if use_cuda:
                data_in = data_in.cuda()
            network(data_in)
            network.mstdp()
 ##############################################   

         # Decoder
        self.fc3 = snn.linear(128, 4096)
		self.fc3.load_weight(vae_fc3)      
        self.fc3_bn = snn.BatchNormalization1D(4096)
        self.desconv1 = snn.ConvolutionTranspose2D(256, 128, kernel_size=4, stride=2, padding=1, bias= False, weight_mean=0.8, weight_std=0.05)
		self.desconv1.load_weight(vae_deconv1)    	
        self.desconv1_bn = snn.BatchNormalization2D(128)
        self.desconv2 = snn.ConvolutionTranspose2D(128, 64, kernel_size=4, stride=2, padding=1, bias= False, weight_mean=0.8, weight_std=0.05)
		self.desconv2.load_weight(vae_deconv2)    		
        self.desconv2_bn = snn.BatchNormalization2D(64)
        self.desconv3 = snn.ConvolutionTranspose2D(64, 64, kernel_size=4, stride=2, padding=3, bias= False, weight_mean=0.8, weight_std=0.05)
		self.desconv3.load_weight(vae_deconv3)    	
        self.desconv3_bn = snn.BatchNormalization2D(64)
        self.desconv4 = snn.ConvolutionTranspose2D(64, 1, kernel_size=5, stride=1, padding=2, bias=False, weight_mean=0.8, weight_std=0.05)
		self.desconv4.load_weight(vae_deconv4)  
        self.sigmoid = pytorch_spiking.SpikingActivation(torch.nn.Sigmoid()) #Module for converting an arbitrary activation function to a spiking equivalent.

   ##############################################  
   	def forward_decoder(self, z, max_layer):
		if self.training:
			h3 = self.fc3(z)
            out = h3.view(h3.size(0), 256, 4, 4)
			pot = self.desconv1_bn(self.desconv1(out))
			spk, pot = sf.fire(pot, self.sconv0_t, True)
			
			self.spk_cnt0 += 1
			if self.spk_cnt0 >= 500:
				self.spk_cnt0 = 0
				ap = torch.tensor(self.dec_mstdp1.learning_rate[0][0].item(), device=self.dec_mstdp1.learning_rate[0][0].device) * 2
				ap = torch.min(ap, self.max_ap)
				an = ap * -0.75
				self.dec_mstdp1.update_all_learning_rate(ap.item(), an.item())
			pot = sf.pointwise_inhibition(pot)
			spk = pot.sign()
			winners = sf.get_k_winners(pot, self.k0, self.r0, spk)
			self.save_data(out, pot, spk, winners)
				
    ##############################################   
			spk_in = sf.pointwise_inhibition(spk)
			pot = self.desconv2_bn(self.desconv2(spk_in))
			spk, pot = sf.fire(pot, self.sconv1_t, True)
		
			self.spk_cnt2 += 1
			if self.spk_cnt1 >= 500:
				self.spk_cnt1 = 0
				ap = torch.tensor(self.dec_mstdp2.learning_rate[0][0].item(), device=self.dec_mstdp2.learning_rate[0][0].device) * 2
				ap = torch.min(ap, self.max_ap)
				an = ap * -0.75
				self.dec_mstdp2.update_all_learning_rate(ap.item(), an.item())                
			pot = sf.pointwise_inhibition(pot)
			spk = pot.sign()
			winners = sf.get_k_winners(pot, self.k1, self.r1, spk)
			self.save_data(spk_in, pot, spk, winners)
				
              ##############################################      
			spk_in = sf.pointwise_inhibition(spk)
			pot = self.desconv3_bn(self.desconv3(spk_in))
			spk, pot = sf.fire(pot, self.sconv2_t, True)
			
			self.spk_cnt2 += 1
			if self.spk_cnt2 >= 500:
				self.spk_cnt2 = 0
				ap = torch.tensor(self.dec_mstdp3.learning_rate[0][0].item(), device=self.dec_mstdp3.learning_rate[0][0].device) * 2
				ap = torch.min(ap, self.max_ap)
				an = ap * -0.75
				self.dec_mstdp3.update_all_learning_rate(ap.item(), an.item())                   
			pot = sf.pointwise_inhibition(pot)
			spk = pot.sign()
			winners = sf.get_k_winners(pot, self.k2, self.r2, spk)
			self.save_data(spk_in, pot, spk, winners)
				
             ##############################################   

			spk_in = sf.pointwise_inhibition(spk)
			pot = self.desconv4(spk_in)
			spk,pot= sf.fire(pot, self.sconv3_t, True)
			
			pot = sf.pointwise_inhibition(pot)
			spk = pot.sign()
			winners = sf.get_k_winners(pot, self.k3, self.r3, spk)
			self.save_data(spk_in, pot, spk, winners)	
            
			return self.sigmoid(spk)
		else:
			h3 = self.fc3(z)
            out = h3.view(h3.size(0), 256, 4, 4)
			pot = self.desconv1_bn(self.desconv1(out))           
			spk, pot = sf.fire(pot, self.sconv0_t, True)
			pot = desconv2_bn(self.desconv2(spk))
			spk, pot = sf.fire(pot, self.sconv1_t, True)
			pot = desconv3_bn(self.desconv3(spk))
			spk, pot = sf.fire(pot, self.sconv2_t, True)
			pot = self.desconv4(spk)
			spk, pot = sf.fire(pot, self.sconv3_t, True)         
			return self.sigmoid(spk)
             ##############################################  
	
	def dec_mstdp(self):

		self.dec_mstdp1(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
		
		self.dec_mstdp2(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
		
		self.dec_mstdp3(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
		
		self.dec_mstdp4(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])   

 ##############################################                

    def decoder(network, data):
        network.train()
        for i in range(len(data)):
            data_in = data[i]
            if use_cuda:
                data_in = data_in.cuda()
            network(data_in, layer_idx)
            network.dec_mstdp()
 ##############################################    

     def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def apply(self, input):
		
        mu, logvar = self.encoder(input)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

 ##############################################    

    def loss(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD, BCE, KLD

 ##############################################    
	