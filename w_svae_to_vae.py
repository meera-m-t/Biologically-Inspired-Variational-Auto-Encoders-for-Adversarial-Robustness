from __future__ import print_function
import torch
import torch.utils.data
import torch.nn as nn
from torch.nn import functional as F
####################################################################################
#load weight from varitional auto encoder (VAE)

PATH = "SVAE_model.pth"

# Save
torch.save(net.state_dict(), PATH)

# encoder
model = SVAE18()
model=model.load_state_dict(torch.load(PATH))

svae_conv0=(model.encoder.layers[0].weight)
Gauss = lambda x, mu, sigma: mu + np.sqrt(2)*sigma*erfinv(2*svae_sconv0-1)
svae_sconv0=Gauss(svae_sconv0)


svae_sconv1=(model.encoder.layers[1].weight)
Gauss = lambda x, mu, sigma: mu + np.sqrt(2)*sigma*erfinv(2*svae_sconv1-1)
svae_sconv1=Gauss(svae_sconv1)



svae_sconv2=(model.encoder.layers[2].weight)
Gauss = lambda x, mu, sigma: mu + np.sqrt(2)*sigma*erfinv(2*svae_sconv2-1)
svae_sconv2=Gauss(svae_sconv2)


svae_sconv3=(model.encoder.layers[3].weight)
Gauss = lambda x, mu, sigma: mu + np.sqrt(2)*sigma*erfinv(2*svae_sconv3-1)
svae_sconv3=Gauss(svae_sconv3)



svae_fc21=(model.encoder.layers[4].weight)
Gauss = lambda x, mu, sigma: mu + np.sqrt(2)*sigma*erfinv(2*svae_fc21-1)
svae_fc21=Gauss(svae_fc21)

svae_fc22=(model.encoder.layers[5].weight)
Gauss = lambda x, mu, sigma: mu + np.sqrt(2)*sigma*erfinv(2*svae_fc22-1)
svae_fc22=Gauss(svae_fc22)


########################################################

##decoder

svae_fc3=(model.decoder.layers[0].weight)
Gauss = lambda x, mu, sigma: mu + np.sqrt(2)*sigma*erfinv(2*svae_fc3-1)
svae_fc3=Gauss(svae_fc3)

svae_desconv1=(model.decoder.layers[1].weight)
Gauss = lambda x, mu, sigma: mu + np.sqrt(2)*sigma*erfinv(2*svae_desconv1-1)
svae_desconv1=Gauss(svae_desconv1)


svae_desconv2=(model.decoder.layers[2].weight)
Gauss = lambda x, mu, sigma: mu + np.sqrt(2)*sigma*erfinv(2*svae_desconv2-1)
svae_desconv2=Gauss(svae_desconv2)



svae_desconv3=(model.decoder.layers[3].weight)
Gauss = lambda x, mu, sigma: mu + np.sqrt(2)*sigma*erfinv(2*svae_deconv3-1)
svae_desconv3=Gauss(svae_desconv3)


svae_desconv4=(model.decoder.layers[4].weight)
Gauss = lambda x, mu, sigma: mu + np.sqrt(2)*sigma*erfinv(2*svae_desconv4-1)
svae_desconv4=Gauss(svae_desconv4)


#################################################################################
### 4 convolutional layer
class VAE18(nn.Module):
    def __init__(self):
        super(VAE18, self).__init__()

        # Encoder
        self.conv0 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2, bias= False)
        self.conv0.weight(svae_sconv0)
        self.conv0_bn = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=3, bias= False)
        self.conv1.weight(svae_sconv1)        
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias= False)
        self.conv2.weight(svae_sconv2)        
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias= False)
        self.conv3.weight(svae_sconv3)        
        self.conv3_bn = nn.BatchNorm2d(256)
        # Latent space
        self.fc21 = nn.Linear(4096, 128)
        self.fc21.weight(svae_fc21)        
        self.fc22 = nn.Linear(4096, 128)
        self.fc22.weight(svae_fc22)   
        # Decoder
        self.fc3 = nn.Linear(128, 4096)
        self.fc3.weight(svae_fc3)           
        self.fc3_bn = nn.BatchNorm1d(4096)     
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias= False)
        self.deconv1.weight(svae_desconv1)
        self.deconv1_bn = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias= False)
        self.deconv2.weight(svae_desconv2)        
        self.conv0.weight(svae_sconv0)        
        self.deconv2_bn = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=3, bias= False)
        self.deconv3.weight(svae_desconv3)      
        self.deconv3_bn = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 1, kernel_size=5, stride=1, padding=2, bias=False)
        self.deconv4.weight(svae_desconv4)    
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


#################################################################################