
# Torch imports.
import torch
import torch.nn as nn
import torch.optim as optim
#from torchinfo import summary
#from torchviz import make_dot
#from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.autograd import Variable
from torch.nn import functional as F
from torch import nn
import numpy as np
import time
SEED = 123
import os

# Device selection.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# PyTorch seed helper.
def torch_seed(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

def gaussian_kernel(a, b):
    dim1_1, dim1_2 = a.shape[0], b.shape[0]
    depth = a.shape[1]
    a = a.view(dim1_1, 1, depth)
    b = b.view(1, dim1_2, depth)
    a_core = a.expand(dim1_1, dim1_2, depth)
    b_core = b.expand(dim1_1, dim1_2, depth)
    numerator = (a_core - b_core).pow(2).mean(2)/depth
    return torch.exp(-numerator)

def compute_kernel(self, x, y):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = x.unsqueeze(1) # (x_size, 1, dim)
        y = y.unsqueeze(0) # (1, y_size, dim)
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        # The example code divides by (dim) here, making <kernel_input> ~ 1/dim
        # excluding (dim) makes <kernel_input> ~ 1
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2)#/float(dim)
        return torch.exp(-kernel_input) # (x_size, y_size)

def compute_mmd(self, x, y):
        xx_kernel = compute_kernel(x,x)
        yy_kernel = compute_kernel(y,y)
        xy_kernel = compute_kernel(x,y)
        return torch.mean(xx_kernel) + torch.mean(yy_kernel) - 2*torch.mean(xy_kernel)

def MMD(a, b):
    return gaussian_kernel(a, a).mean() + gaussian_kernel(b, b).mean() - 2*gaussian_kernel(a, b).mean()

def KL_Loss(mean, log_var):
    # KL divergence.
    kl_loss = - 0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
    return kl_loss

MSE = torch.nn.MSELoss()
#lossf = torch.nn.GaussianNLLLoss()
def Reconstruction_Loss(inputs, outputs,w):
    MSE = 0.5*torch.mean(w*(inputs-outputs).pow(2))
    return MSE

def train_VAE(
    model,
    modelfile,
    testloader,
    trainloader,
    n_latent,
    recon_loss_func=MSE,
    kl_loss_func=KL_Loss,
    patience=20,
    lr = 5e-4,
    alpha = 1e3,
    beta = 1e-1,
    SEED = SEED,
):
    history = np.zeros((0,7))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    no_improvement = 0
    nan_count = 0
    First_rest = 0
    val_loss_history = torch.zeros(3)
    with torch.no_grad():
        model.eval()
        for i, (inputs, label) in enumerate(testloader):

            inputs = inputs.to(device).float()
            optimizer.zero_grad()

            outputs, mean, log_var, z = model(inputs)

            recon_loss = recon_loss_func(inputs,outputs.float())
            kl_loss = kl_loss_func(mean, log_var)

            loss = alpha * recon_loss + beta * kl_loss

            val_loss_history[0] += loss.detach().item()
            val_loss_history[1] += recon_loss.detach().item()
            val_loss_history[2] += kl_loss.detach().item()

        val_loss_history /= (i+1)
        best_loss = val_loss_history[0]

    for epoch in range(0, 2000):

        model.train()
        torch_seed(seed=SEED)
        train_loss_history = torch.zeros(3)
        st = time.time()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
        for i, (inputs, label) in enumerate(trainloader):

            inputs = inputs.to(device).float()
            optimizer.zero_grad()

            outputs, mean, log_var, z = model(inputs)
            recon_loss = recon_loss_func(inputs,outputs.float())
            kl_loss = kl_loss_func(mean, log_var)

            loss = alpha * recon_loss + beta * kl_loss

            if torch.any( torch.isnan( loss ) ) or torch.any( torch.isinf( loss ) ):
                lr = lr * 0.999
                nan_count += 1
                print(f'train loss {lr} /n')
                print(f'train outputs{torch.any( torch.isnan( outputs ) ) } /n')
                if torch.any( torch.isnan( inputs ) ):
                    print(inputs)
                    print(f'train inputs nan /n')
                if os.path.isfile(modelfile):
                    model.load_state_dict(torch.load(modelfile, map_location=device))
                break

            loss = loss.to(device)
            loss.backward()
            optimizer.step()

            train_loss_history[0] += loss.detach().item()
            train_loss_history[1] += recon_loss.detach().item()
            train_loss_history[2] += kl_loss.detach().item()

        train_loss_history /= (i+1)

        model.eval()
        torch_seed(seed=SEED)
        val_loss_history = torch.zeros(3)
        with torch.no_grad():
            for i, (inputs, label) in enumerate(testloader):

                inputs = inputs.to(device).float()
                optimizer.zero_grad()

                outputs, mean, log_var, z = model(inputs.float())
                recon_loss = recon_loss_func(inputs,outputs.float())
                kl_loss = kl_loss_func(mean, log_var)

                loss = alpha * recon_loss + beta * kl_loss
                loss = loss.to(device)

                if torch.any( torch.isnan( loss ) ) or torch.any( torch.isinf( loss ) ):
                    lr = lr * 0.999
                    print(f'test loss nan {i} {n_latent} {lr} /n')
                    print(f'test inputs {torch.any( torch.isnan( inputs ) ) } /n')
                    print(f'test outputs {torch.any( torch.isnan( outputs ) ) } /n')
                    print(inputs[0])
                    if os.path.isfile(modelfile):
                        model.load_state_dict(torch.load(modelfile, map_location=device))
                    no_improvement -= 1
                    if no_improvement < 0:
                        no_improvement = 0
                    break
                val_loss_history[0] += loss.detach().item()
                val_loss_history[1] += recon_loss.detach().item()
                val_loss_history[2] += kl_loss.detach().item()

            val_loss_history /= (i+1)

        item = np.hstack([epoch, train_loss_history.numpy(), val_loss_history.numpy()])
        history = np.vstack([history,item])

        if nan_count == patience/2:
            model.mean[0] = nn.Linear(1024, n_latent)
            model.log_var[0] = nn.Linear(1024, n_latent)
            model = model.to(device)
        if best_loss > val_loss_history[0]:
            print('renew')
            best_loss =  val_loss_history[0]
            torch.save(model.state_dict(), modelfile)
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement > patience/2 and First_rest == 0:
            print('Reset Mean and log_var layer')
            if os.path.isfile(modelfile):
                model.load_state_dict(torch.load(modelfile, map_location=device))
            model.mean[0] = nn.Linear(1024, n_latent)
            model.log_var[0] = nn.Linear(1024, n_latent)
            model = model.to(device)
            First_rest += 1
        if no_improvement == patience:
            print('Validation loss has not improved for {} epochs, stopping early.'.format(patience))
            return history
            break
        if val_loss_history[0] > 1e5:
            lr = lr * 0.999

        et = time.time()

        print(
            modelfile,'\n',
            'no_improvement', no_improvement ,'\n',

            'epoch', history[-1][0] ,'\n',

            'loss:', history[-1][1],'\n',
            'val loss:', history[-1][4],'\n',

            'reconstruct loss:', history[-1][2],'\n',
            'reconstruct  val loss:', history[-1][5],'\n',

            'KL loss:', history[-1][3],'\n',
            'KL val loss:', history[-1][6],'\n',

            '{}'.format(np.round( (et-st)/60, 5 )),
        )


def train_CVAE(
    model,
    modelfile,
    testloader,
    trainloader,
    n_latent,
    n_class,
    recon_loss_func=MSE,
    kl_loss_func=KL_Loss,
    patience=20,
    lr = 5e-4,
    beta = 1e-1,
    a = 1e3,
    SEED = SEED,
):
    history = np.zeros((0,7))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    no_improvement = 0
    nan_count = 0
    First_rest = 0
    val_loss_history = torch.zeros(3)
    with torch.no_grad():
        model.eval()
        for i, (inputs, label) in enumerate(testloader):

            optimizer.zero_grad()
            inputs = inputs.to(device).float()
            label = label.to(device).float()
            outputs, mean, log_var, z = model(inputs,label)

            recon_loss = recon_loss_func(inputs,outputs.float())
            kl_loss = kl_loss_func(mean, log_var)

            loss = a * recon_loss+beta*kl_loss

            val_loss_history[0] += loss.detach().item()
            val_loss_history[1] += recon_loss.detach().item()
            val_loss_history[2] += kl_loss.detach().item()

        val_loss_history /= (i+1)
        best_loss = val_loss_history[0]

    for epoch in range(0, 2000):

        model.train()
        torch_seed(seed=SEED)
        train_loss_history = torch.zeros(3)
        st = time.time()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
        for i, (inputs, label) in enumerate(trainloader):

            optimizer.zero_grad()
            inputs = inputs.to(device).float()
            label = label.to(device).float()
            outputs, mean, log_var, z = model(inputs,label)
            recon_loss = recon_loss_func(inputs,outputs.float())
            kl_loss = kl_loss_func(mean, log_var)

            loss = a * recon_loss + beta * kl_loss

            if torch.any( torch.isnan( loss ) ) or torch.any( torch.isinf( loss ) ):
                lr = lr * 0.999
                nan_count += 1
                print(f'train loss {lr} /n')
                print(f'train inputs{torch.any( torch.isnan( inputs ) ) } /n')
                print(f'train outputs{torch.any( torch.isnan( outputs ) ) } /n')
                if os.path.isfile(modelfile):
                    model.load_state_dict(torch.load(modelfile, map_location=device))
                no_improvement -= 1
                if no_improvement < 0:
                    no_improvement = 0
                break

            loss = loss.to(device)
            loss.backward()
            optimizer.step()

            train_loss_history[0] += loss.detach().item()
            train_loss_history[1] += recon_loss.detach().item()
            train_loss_history[2] += kl_loss.detach().item()

        train_loss_history /= (i+1)

        model.eval()
        torch_seed(seed=SEED)
        val_loss_history = torch.zeros(3)
        with torch.no_grad():
            for i, (inputs, label) in enumerate(testloader):

                optimizer.zero_grad()
                inputs = inputs.to(device).float()
                label = label.to(device).float()
                outputs, mean, log_var, z = model(inputs,label)
                recon_loss = recon_loss_func(inputs,outputs.float())
                kl_loss = kl_loss_func(mean, log_var)

                loss = a * recon_loss+beta*kl_loss
                loss = loss.to(device)

                if torch.any( torch.isnan( loss ) ) or torch.any( torch.isinf( loss ) ):
                    if os.path.isfile(modelfile):
                        model.load_state_dict(torch.load(modelfile, map_location=device))
                    lr = lr * 0.999
                    print(f'test loss nan {i} {n_latent} {lr} /n')
                    print(f'test inputs {torch.any( torch.isnan( inputs ) ) } /n')
                    print(f'test outputs {torch.any( torch.isnan( outputs ) ) } /n')
                    no_improvement -= 1
                    if no_improvement < 0:
                        no_improvement = 0
                    break
                val_loss_history[0] += loss.detach().item()
                val_loss_history[1] += recon_loss.detach().item()
                val_loss_history[2] += kl_loss.detach().item()

            val_loss_history /= (i+1)

        item = np.hstack([epoch, train_loss_history.numpy(), val_loss_history.numpy()])
        history = np.vstack([history,item])

        if nan_count == patience/2:
            model.mean[0] = nn.Linear(1024+n_class, n_latent)
            model.log_var[0] = nn.Linear(1024+n_class, n_latent)
            model = model.to(device)
        if best_loss > val_loss_history[0]:
            print('renew')
            best_loss =  val_loss_history[0]
            torch.save(model.state_dict(), modelfile)
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement > patience/2 and First_rest == 0:
            print('Reset Mean and log_var layer')
            if os.path.isfile(modelfile):
                model.load_state_dict(torch.load(modelfile, map_location=device))
            model.mean[0] = nn.Linear(1024+n_class, n_latent)
            model.log_var[0] = nn.Linear(1024+n_class, n_latent)
            model = model.to(device)
            First_rest += 1
        if no_improvement == patience:
            print('Validation loss has not improved for {} epochs, stopping early.'.format(patience))
            return history
            break
        if val_loss_history[0] > 1e5:
            lr = lr * 0.999

        et = time.time()

        print(
            modelfile,'\n',
            'no_improvement', no_improvement ,'\n',

            'epoch', history[-1][0] ,'\n',

            'loss:', history[-1][1],'\n',
            'val loss:', history[-1][4],'\n',

            'reconstruct loss:', history[-1][2],'\n',
            'reconstruct  val loss:', history[-1][5],'\n',

            'KL loss:', history[-1][3],'\n',
            'KL val loss:', history[-1][6],'\n',

            '{}'.format(np.round( (et-st)/60, 5 )),
        )

def Loss_VAE(
    model,
    testloader,
    test_df,
    error,
    sigma,
    a = 1000,
):
    model.eval()
    torch_seed()
    val_loss_history = torch.zeros(5)
    pattern = r"Image_SED_plate(\d+)_mjd(\d+)_fiber(\d+)"
    with torch.no_grad():
        for i, ( inputs_image, inputs_spectra, filenames) in enumerate(testloader):
            Sigma = []
            Error = []
            for file_name in filenames:
                match = re.search(pattern, file_name)
                if match:
                    # Extract the three numbers from the matched groups
                    plate = int( match.group(1) )
                    mjd = int( match.group(2) )
                    fiber = int( match.group(3) )
                else:
                    print("No match!!",file_name)

                ind = test_df[
                    ( test_df['plate']==plate ) &
                    ( test_df['mjd']==mjd ) &
                    ( test_df['fiber']==fiber )
                ]['index'].values[0]
                Error.append(error[ind])
                Sigma.append(sigma[ind])
            Error = torch.tensor(np.array(Error)).to(device)
            Sigma = torch.tensor(np.array(Sigma)).to(device)
            inputs_image = inputs_image.to(device)
            inputs_spectra = inputs_spectra.to(device)
            recon_image, recon_spectra, mu, log_var, _ = model( inputs_image, inputs_spectra )
            recon_loss_image = F.mse_loss( recon_image, inputs_image )
            recon_loss_spectra = F.mse_loss( recon_spectra, inputs_spectra )
            r = inputs_spectra - recon_spectra
            r[Error==0] = 0
            chi = torch.sum ( r**2 /Sigma )
            KLD = loss_KLD(mu,log_var)
            loss = a * recon_loss_image + a * recon_loss_spectra + KLD

            val_loss_history[0] += loss.detach().item()
            val_loss_history[1] += recon_loss_image.detach().item()
            val_loss_history[2] += recon_loss_spectra.detach().item()
            val_loss_history[3] += KLD.detach().item()
            val_loss_history[4] += chi.detach().item()

    return val_loss_history.numpy()

from tqdm import tqdm
def Save_output_model(
    model,
    modelfile,
    Dataloader,
    save_latent,
    save_filenames,
    save_recon=None,
):
    print(modelfile)
    # Create empty lists to store reconstructed images and spectra
    recon_list = []
    latent_list = []
    filename_list = []
    torch_seed()
    with torch.no_grad():
        model.eval()
        for i, ( inputs, filenames) in tqdm( enumerate(Dataloader) ):
            inputs = inputs.to(device)

            recon_image, mu, log_var,z = model( inputs )

            # Inside the loop
            recon_image_np = recon_image.cpu().detach().numpy()  # Convert tensor to numpy array
            latent_means_np = mu.cpu().detach().numpy()

            # Append the reconstructed image and spectra to the respective lists
            recon_list.append(recon_image_np)
            latent_list.append(latent_means_np)
            filename_list.extend(filenames)

    # After the loop, convert the lists to numpy arrays
    recon_list = np.concatenate(recon_list)
    latent_list = np.concatenate(latent_list)

    if not save_recon == None:
        np.save( save_recon, recon_list )

    np.save( save_latent, latent_list )
    np.save( save_filenames, filename_list )
    np.save( save_latent, latent_list )
