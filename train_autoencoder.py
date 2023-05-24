from pickle import NONE
from random import sample
import torch, argparse
from torch.utils.data import DataLoader,Dataset
import numpy as np
import os, sys
import importlib
from utils.config_load import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

from utils.model_load import get_dynamics, get_model, count_parameters, load_saved_model
DEVICE = 'cuda'
def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--dynamics', default = 'heat_equation', type = str)
    parser.add_argument('--dynamics_samples', default = 200, type = int)
    parser.add_argument('--dynamics_noise', default = 0.0, type = float)
    parser.add_argument('--autoencoder_model_def', default = 'AE1D', type = str)
    parser.add_argument('--autoencoder_model_input_dim', default = 101, type = int)
    parser.add_argument('--autoencoder_model_latent_dim', default = 9, type = int)
    parser.add_argument('--epochs', default = 200, type = int)
    parser.add_argument('--batch_size', default = 100, type = int)
    parser.add_argument('--learning_rate', default = 1e-3, type = float)
    parser.add_argument('--seed', default=2, type=int, help='random seed')
    parser.add_argument('--save_name', default = './saved_models/autoencoder_heat_equation.tar', type = str)
    parser.add_argument('--load_name', default = '', type = str)
    return parser.parse_args()


def MSE(x, y):
    return ((x-y)**2).mean()

def VAE_loss(recon_x, x, mu=None, logvar=None):
    BCE = MSE(recon_x, x)
    if mu == None and logvar == None:
        KLD = 0
    else:
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
    return BCE + KLD*0


def train_autoencoder(model, data, dynamics, args):

    model.train()
    epochs = args.epochs

    x = torch.tensor( data['x'], requires_grad=True, dtype=torch.float32).to(DEVICE)
    test_x = torch.tensor( data['test_x'], requires_grad=True, dtype=torch.float32).to(DEVICE)
    xn1 = torch.Tensor(data['fx']).to(DEVICE)
    test_xn1 = torch.Tensor(data['test_fx']).to(DEVICE)

    print("train data shape:{}".format(x.shape))

    train_tuple = []
    for i in range(x.shape[0]):
      train_tuple.append((x[i],xn1[i]))
    train_loader = DataLoader(train_tuple, batch_size = args.batch_size, shuffle= True)

    test_tuple = []
    for i in range(test_x.shape[0]):
      test_tuple.append((test_x[i],test_xn1[i]))
    test_loader = DataLoader(test_tuple, batch_size = args.batch_size, shuffle= True)


    optim = torch.optim.Adam(model.parameters(), \
            args.learning_rate, weight_decay=0)

    # loss_func = info_nce_loss
    loss_func = MSE

    for epoch in range(epochs):

       
        count = 0
        for x, fx in train_loader:
            count += 1
            optim.zero_grad()
            x_hat = model.forward(x)

            loss = VAE_loss(x_hat,x)
            
            loss.backward()
            optim.step()

        if epoch % 10 == 0:

            mean_losses = 0
            mean_conserve_losses = 0
            mean_accs = 0
            mean_conserve_accs = 0
            count = 0
            for x, fx in test_loader:
                
                x_hat = model.forward(x)
                loss = VAE_loss(x_hat,x)

                mean_losses += loss

                x_numpy = x.cpu().detach().numpy()
                x_numpy = np.reshape(x_numpy, (-1, x_numpy.shape[-1]))  
                count += 1

            print("epoch:{},mean_loss:{}".\
                format(epoch,mean_losses/count))
    
    return model

def save_model(model,args):

    torch.save(model.state_dict(), args.save_name)

    print("save file to:{}".format(args.save_name))


# def visualize_autoencoder(model,dynamics, args):
#     data = dynamics.get_data(dim = 2 , sample= 10)
    
#     subplot_num = 10
#     x = data['x']
#     fx = data['fx']
#     samples = x.shape[0]
#     index = np.random.choice([i for i in range(samples)],subplot_num)
#     x_hat = model(torch.from_numpy(x).float().to(DEVICE)).cpu().detach().numpy()
#     fx_hat = model(torch.from_numpy(fx).float().to(DEVICE)).cpu().detach().numpy()

#     fig,axes = plt.subplots(subplot_num,2, figsize=(4,14))
#     for i in range(subplot_num):
#         axes[i,0].plot(x[index[i]]) 
#         axes[i,0].plot(x_hat[index[i]],linestyle = 'dashed') 
#         axes[i,1].plot(fx[index[i]]) 
#         axes[i,1].plot(fx_hat[index[i]],linestyle = 'dashed') 


#     fig.savefig('{}{}.png'.format('./figs/autoencoder_',args.dynamics))

      
def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    model = get_model(args, model_type = 'autoencoder').to(DEVICE)
    print("model_para_number:"+str(count_parameters(model)))
    
    dynamics = get_dynamics(args)
    data = dynamics.get_data(dim = 2 , sample= args.dynamics_samples, noise=args.dynamics_noise)
    # if args.load_name:
    #     model = get_model(args, model_type='autoencoder').to(DEVICE)
    #     load_saved_model(model, args.load_name)
    
    model = train_autoencoder(model, data, dynamics, args)
    save_model(model,args)
    visualize_autoencoder(model,dynamics, args)
    
    
    return None

if __name__ == "__main__":
    args = get_args()
    main(args)