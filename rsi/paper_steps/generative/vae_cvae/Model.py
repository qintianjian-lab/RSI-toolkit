
# Torch imports.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.autograd import Variable
from torch.nn import functional as F
from torch import nn
import numpy as np
import time
SEED = 123

# Device selection.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class Image_CVAE(nn.Module):
    def __init__(self, latent_dim=100, n_classes=2 ):
        super(Image_CVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), # 64x64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            
            nn.Flatten(), # Flatten the tensor
            nn.Linear(256*8*8, 1024), # Fully connected layer
            nn.LeakyReLU(inplace=True),
        )
        
        self.mean = torch.nn.Sequential(

            nn.Linear( 1024 + self.n_classes, self.latent_dim ),

        )

        self.log_var = torch.nn.Sequential(

            nn.Linear( 1024 + self.n_classes, self.latent_dim ),

        )
    
        self.decoder = nn.Sequential(
            
            nn.Linear(self.latent_dim+self.n_classes, 1024), # Fully connected layer
            nn.LeakyReLU(inplace=True),
            
            nn.Linear(1024, 256*8*8), # Fully connected layer
            nn.LeakyReLU(inplace=True),
            nn.Unflatten(1, (256, 8, 8)), # Unflatten the tensor
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # 64x64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1), # 128x128
            nn.Tanh() # Tanh activation function to output values from -1 to 1
        )

    def reparameterize(self, z_mean, z_log_var):
        epsilon = torch.randn(z_mean.shape).to(device)
        z = z_mean + torch.exp(z_log_var / 2) * epsilon
        return z

    def forward(self, x, label):

        x = self.encoder(x)
        x_new = torch.cat([x,label],dim = 1).to(device)

        # mean mu
        mean = self.mean(x_new).to(device)

        # log σ^2
        log_var = self.log_var(x_new).to(device)

        z = self.reparameterize(mean,log_var).to(device)

        z_new = torch.cat([z,label],dim = 1).to(device)

        x = self.decoder(z_new).to(device)

        return x, mean, log_var, z


class Image_VAE(nn.Module):
    def __init__(self, latent_dim=100):
        super(Image_VAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), # 64x64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            
            nn.Flatten(), # Flatten the tensor
            nn.Linear(256*8*8, 1024), # Fully connected layer
            nn.LeakyReLU(inplace=True),
        )
        
        self.mean = torch.nn.Sequential(

            nn.Linear(1024, self.latent_dim),

        )

        self.log_var = torch.nn.Sequential(

            nn.Linear(1024, self.latent_dim),

        )
    
        self.decoder = nn.Sequential(
            
            nn.Linear(self.latent_dim, 1024), # Fully connected layer
            nn.LeakyReLU(inplace=True),
            
            nn.Linear(1024, 256*8*8), # Fully connected layer
            nn.LeakyReLU(inplace=True),
            nn.Unflatten(1, (256, 8, 8)), # Unflatten the tensor
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # 64x64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1), # 128x128
            nn.Tanh() # Tanh activation function to output values from -1 to 1
        )

    def reparameterize(self, z_mean, z_log_var):
        epsilon = torch.randn(z_mean.shape).to(device)
        z = z_mean + torch.exp(z_log_var / 2) * epsilon
        return z

    def forward(self, x):

        x = self.encoder(x).to(device)

        # mean mu
        mean = self.mean(x).to(device)
        # log σ^2
        log_var = self.log_var(x).to(device)

        z = self.reparameterize(mean,log_var).to(device)

        x = self.decoder(z).to(device)

        return x, mean, log_var, z

class Image_CVAE2(nn.Module):
    def __init__(self, latent_dim=100, n_classes=2,image_size=128 ):
        super(Image_CVAE2, self).__init__()
        
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.image_size = image_size
        
        self.encoder_labels = nn.Sequential(
            nn.Linear( self.n_classes, 1024 ),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear( 1024, self.image_size*self.image_size ),
            nn.LeakyReLU(inplace=True),
            nn.Unflatten(1, (1, self.image_size, self.image_size)), # Unflatten the tensor
        )
        
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=4, stride=2, padding=1), # 64x64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            
            nn.Flatten(), # Flatten the tensor
            nn.Linear(256*8*8, 1024), # Fully connected layer
            nn.LeakyReLU(inplace=True),
        )
        
        self.mean = torch.nn.Sequential(

            nn.Linear( 1024, self.latent_dim ),

        )

        self.log_var = torch.nn.Sequential(

            nn.Linear( 1024, self.latent_dim ),

        )
    
        self.decoder = nn.Sequential(
            
            nn.Linear(self.latent_dim+self.n_classes, 1024), # Fully connected layer
            nn.LeakyReLU(inplace=True),
            
            nn.Linear(1024, 256*8*8), # Fully connected layer
            nn.LeakyReLU(inplace=True),
            nn.Unflatten(1, (256, 8, 8)), # Unflatten the tensor
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # 64x64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1), # 128x128
            nn.Tanh() # Tanh activation function to output values from -1 to 1
        )

    def reparameterize(self, z_mean, z_log_var):
        epsilon = torch.randn(z_mean.shape).to(device)
        z = z_mean + torch.exp(z_log_var / 2) * epsilon
        return z

    def forward(self, x, label):

        Unflatten_label = self.encoder_labels(label).to(device)
        x = torch.cat([x,Unflatten_label],dim = 1).to(device)
        x = self.encoder(x)

        # mean mu
        mean = self.mean(x).to(device)

        # log σ^2
        log_var = self.log_var(x).to(device)

        z = self.reparameterize(mean,log_var).to(device)

        z_new = torch.cat([z,label],dim = 1).to(device)

        x = self.decoder(z_new).to(device)

        return x, mean, log_var, z

class VAEWithDropout(nn.Module):
    def __init__(self, latent_dim=100, n_classes=2,dropout_decoder_p=0,dropout_encoder_p=0):
        super(VAEWithDropout, self).__init__()
        
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), # 64x64 
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout_encoder_p),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout_encoder_p),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout_encoder_p),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout_encoder_p),
            
            nn.Flatten(), # Flatten the tensor
            nn.Linear(256*8*8, 1024), # Fully connected layer
            nn.LeakyReLU(inplace=True),
        )
        
        self.mean = torch.nn.Sequential(

            nn.Linear( 1024 + self.n_classes, self.latent_dim ),

        )

        self.log_var = torch.nn.Sequential(

            nn.Linear( 1024 + self.n_classes, self.latent_dim ),

        )
    
        self.decoder = nn.Sequential(
            
            nn.Linear(self.latent_dim+self.n_classes, 1024), # Fully connected layer
            nn.LeakyReLU(inplace=True),
            
            nn.Linear(1024, 256*8*8), # Fully connected layer
            nn.LeakyReLU(inplace=True),
            nn.Unflatten(1, (256, 8, 8)), # Unflatten the tensor
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout_decoder_p),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout_decoder_p),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # 64x64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout_decoder_p),
            
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1), # 128x128
            nn.Tanh() # Tanh activation function to output values from -1 to 1
        )

    def reparameterize(self, z_mean, z_log_var):
        epsilon = torch.randn(z_mean.shape).to(device)
        z = z_mean + torch.exp(z_log_var / 2) * epsilon
        return z

    def forward(self, x, label):

        x = self.encoder(x)
        x_new = torch.cat([x,label],dim = 1).to(device)

        # mean mu
        mean = self.mean(x_new).to(device)

        # log σ^2
        log_var = self.log_var(x_new).to(device)

        z = self.reparameterize(mean,log_var).to(device)

        z_new = torch.cat([z,label],dim = 1).to(device)

        x = self.decoder(z_new).to(device)

        return x, mean, log_var, z

class Image_VAE_Encoder(nn.Module):
    def __init__(self, latent_dim=100):
        super(Image_VAE_Encoder, self).__init__()
        
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), # 64x64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            
            nn.Flatten(), # Flatten the tensor
            nn.Linear(256*8*8, 1024), # Fully connected layer
            nn.LeakyReLU(inplace=True),
        )
        
        self.mean = torch.nn.Sequential(

            nn.Linear(1024, self.latent_dim),

        )

    def reparameterize(self, z_mean, z_log_var):
        epsilon = torch.randn(z_mean.shape).to(device)
        z = z_mean + torch.exp(z_log_var / 2) * epsilon
        return z

    def forward(self, x):

        x = self.encoder(x).to(device)

        # mean mu
        mean = self.mean(x).to(device)

        return mean

class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view((-1,1,4000))
class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, int(x.shape[1]*x.shape[2]))
class Unflatten(torch.nn.Module):
    def forward(self, x):
        return x.view((-1,64,250))
class Reshape2(torch.nn.Module):
    def forward(self, x):
        return x.view((-1,4000))
    
class Spectra_VAE(nn.Module):
    def __init__(self, latent_dim=100):
        super(Spectra_VAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        self.encoder = torch.nn.Sequential(

            Reshape(),

            nn.Conv1d(1,8,5,padding=1,stride=1),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0),

            nn.MaxPool1d(2),

            nn.Conv1d(8,16,5,padding=1,stride=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0),

            nn.MaxPool1d(2),

            nn.Conv1d(16,32,5,padding=1,stride=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0),

            nn.MaxPool1d(2),

            nn.Conv1d(32,64,5,padding=1,stride=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0),

            nn.MaxPool1d(2),

            Flatten(),

            nn.Linear(64*248, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0),

        )

        self.mean = torch.nn.Sequential(

            nn.Linear(1024, self.latent_dim),

        )

        self.log_var = torch.nn.Sequential(

            nn.Linear(1024, self.latent_dim),

        )

        self.decoder = torch.nn.Sequential(

            nn.Linear(self.latent_dim, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0),

            nn.Linear(1024, 64*250),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0),

            Unflatten(),

            nn.ConvTranspose1d(64,32,5,stride=2,padding=1,output_padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0),

            nn.ConvTranspose1d(32,16,5,stride=2,padding=1,output_padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0),

            nn.ConvTranspose1d(16,8,5,stride=2,padding=1,output_padding=1),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0),

            nn.ConvTranspose1d(8,8,3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0),

            nn.Conv1d(8,1,3,padding=1,stride=1),
            # nn.ReLU(inplace=True),
            Flatten(),

            nn.Linear(4028,4000),
            # nn.ReLU(inplace=True),
            # Linear output preserves continuous flux values, including negative z-scored samples.
        )

    def reparameterize(self, z_mean, z_log_var):
        epsilon = torch.randn(z_mean.shape).to(device)
        z = z_mean + torch.exp(z_log_var / 2) * epsilon
        return z

    def forward(self, x):

        x = self.encoder(x).to(device)

        # mean mu
        mean = self.mean(x).to(device)
        # log σ^2
        log_var = self.log_var(x).to(device)

        z = self.reparameterize(mean,log_var).to(device)

        x = self.decoder(z).to(device)

        return x, mean, log_var, z

class Spectra_CVAE(nn.Module):
    def __init__(self, latent_dim=100,n_class=0):
        super(Spectra_CVAE, self).__init__()

        self.latent_dim = latent_dim
        self.n_class = n_class
        
        self.encoder = torch.nn.Sequential( 
    
            Reshape(),
    
            nn.Conv1d(1,8,5,padding=1,stride=1),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0),
    
            nn.MaxPool1d(2),
    
            nn.Conv1d(8,16,5,padding=1,stride=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0),
    
            nn.MaxPool1d(2),
    
            nn.Conv1d(16,32,5,padding=1,stride=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0),
    
            nn.MaxPool1d(2),
    
            nn.Conv1d(32,64,5,padding=1,stride=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0),
    
            nn.MaxPool1d(2),
    
            Flatten(),
    
            nn.Linear(64*248, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0),
    
        )
    
    
        self.mean = torch.nn.Sequential(
    
            nn.Linear(1024+self.n_class , self.latent_dim),
    
        )
    
        self.log_var = torch.nn.Sequential(
    
            nn.Linear(1024+self.n_class , self.latent_dim),
    
        )
    
        self.decoder = torch.nn.Sequential(
    
            nn.Linear(self.latent_dim + self.n_class, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0),
    
            nn.Linear(1024, 64*250),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0),
    
            Unflatten(),
    
            nn.ConvTranspose1d(64,32,5,stride=2,padding=1,output_padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0),
    
            nn.ConvTranspose1d(32,16,5,stride=2,padding=1,output_padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0),
    
            nn.ConvTranspose1d(16,8,5,stride=2,padding=1,output_padding=1),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0),
    
            nn.ConvTranspose1d(8,8,3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0),
    
            nn.Conv1d(8,1,3,padding=1,stride=1),
            nn.ReLU(inplace=True),
            Flatten(),
    
            nn.Linear(4028,4000) 
        )
    def reparameterize(self, z_mean, z_log_var):
        epsilon = torch.randn(z_mean.shape).to(device)
        z = z_mean + torch.exp(z_log_var / 2) * epsilon
        return z

    def forward(self, x, label):

        x = self.encoder(x)
        x_new = torch.cat([x,label],dim = 1).to(device)

        #mean μ
        mean = self.mean(x_new).to(device)

        # log σ^2
        log_var = self.log_var(x_new).to(device)

        z = self.reparameterize(mean,log_var).to(device)

        z_new = torch.cat([z,label],dim = 1).to(device)

        x = self.decoder(z_new).to(device)

        return x, mean, log_var, z
