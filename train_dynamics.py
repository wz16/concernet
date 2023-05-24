from ast import parse
from tkinter import TRUE
# from xmlrpc.client import Boolean, boolean
import torch, argparse
from torch.utils.data import DataLoader,Dataset
import numpy as np
import os, sys
import importlib
from utils.config_load import *
import torch.nn.functional as F
from models.base_wrapper import wrapper
from utils.model_load import get_dynamics, get_model, count_parameters, load_saved_model
import inspect
DEVICE = 'cuda'

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--dynamics', default = 'ideal_spring_mass', type = str)
    parser.add_argument('--dynamics_samples', default = 10, type = int)
    parser.add_argument('--dynamics_noise', default = 0.0, type = float)
    parser.add_argument('--dynamics_time', default = 1.0, type = float)
    parser.add_argument('--wrapper_mode', default = 'default', type = str)
    parser.add_argument('--ae_mode', action='store_true')
    parser.add_argument('--dynamics_model_def', default = 'MLP', type = str)
    parser.add_argument('--dynamics_model_input_dim', default = 2, type = int)
    parser.add_argument('--dynamics_model_output_dim', default = 2, type = int)
    parser.add_argument('--dynamics_model_hidden_dim', default = 100, type = int)
    parser.add_argument('--autoencoder_model_def', default = 'AE1D', type = str)
    parser.add_argument('--autoencoder_model_input_dim', default = 101, type = int)
    parser.add_argument('--autoencoder_model_latent_dim', default = 9, type = int)
    parser.add_argument('--traj_model_def', default = 'MLP', type = str)
    parser.add_argument('--traj_model_input_dim', default = 9, type = int)
    parser.add_argument('--traj_model_output_dim', default = 1, type = int)
    parser.add_argument('--traj_model_hidden_dim', default = 100, type = int)
    parser.add_argument('--load_ae_model', default = '', type = str)
    parser.add_argument('--load_traj_model', default ='', type = str)
    parser.add_argument('--epochs', default = 10, type = int)
    parser.add_argument('--batch_size', default = 20, type = int)
    parser.add_argument('--learning_rate', default = 1e-3, type = float)
    parser.add_argument('--seed', default=2, type=int, help='random seed')
    parser.add_argument('--save_name', default = 'saved_models/dynamics_hnn_ideal_spring_mass.tar', type = str)
    return parser.parse_args()

def train(wrapper_model, data, args):
    wrapper_model.train()
    epochs = args.epochs

    loss_func  = torch.nn.MSELoss(reduction='sum')

    x = torch.tensor( data['x'], requires_grad=True, dtype=torch.float32).to(DEVICE)
    test_x = torch.tensor( data['test_x'], requires_grad=True, dtype=torch.float32).to(DEVICE)
    fx = torch.Tensor(data['fx']).to(DEVICE)
    test_fx = torch.Tensor(data['test_fx']).to(DEVICE)
    if wrapper_model.ae_mode:
        
        noutputs = args.dynamics_model_input_dim
        n_x = x.shape[0]
        x_repeat = x.repeat(noutputs, 1)
        x_repeat.retain_grad()
        x_repeat.requires_grad_(True)
        z = wrapper_model.ae_model.encode(x_repeat)
        z.retain_grad()
        z.backward(torch.eye(noutputs).to(DEVICE).repeat(n_x,1))
        dzdx = x_repeat.grad.data.reshape(n_x, noutputs, -1)
        dzdt = torch.bmm(dzdx,fx.unsqueeze(-1)).squeeze(-1)
        fx = dzdt.detach().data
        x_repeat.grad.data.zero_()
        x = wrapper_model.ae_model.encode(x).detach().data
        
        n_x = test_x.shape[0]
        x_repeat = test_x.repeat(noutputs, 1)
        x_repeat.retain_grad()
        x_repeat.requires_grad_(True)
        z = wrapper_model.ae_model.encode(x_repeat)
        z.retain_grad()
        z.backward(torch.eye(noutputs).to(DEVICE).repeat(n_x,1))
        dzdx = x_repeat.grad.data.reshape(n_x, noutputs, -1)
        dzdt = torch.bmm(dzdx,test_fx.unsqueeze(-1)).squeeze(-1)
        test_fx = dzdt
        x_repeat.grad.data.zero_()
        test_x = wrapper_model.ae_model.encode(test_x).detach().data

    print("train data shape:{}".format(x.shape))

    train_tuple = []
    for i in range(x.shape[0]):
      train_tuple.append((x[i],fx[i]))
    train_loader = DataLoader(train_tuple, batch_size =  args.batch_size, shuffle= True)

    test_tuple = []
    for i in range(test_x.shape[0]):
      test_tuple.append((test_x[i],test_fx[i]))
    test_loader = DataLoader(test_tuple, batch_size =  args.batch_size , shuffle= True)

    if args.wrapper_mode == 'default':
        optim = torch.optim.Adam(wrapper_model.dynamics_model.parameters(), \
                args.learning_rate, weight_decay=0)
    elif args.wrapper_mode == 'project_with_knowngx':
        optim = torch.optim.Adam(wrapper_model.dynamics_model.parameters(), \
                args.learning_rate, weight_decay=0)
    elif args.wrapper_mode == 'half_project_with_knowngx':
        optim = torch.optim.Adam(wrapper_model.dynamics_model.parameters(), \
                args.learning_rate, weight_decay=0)
    for epoch in range(epochs):
        mean_loss = 0
        count = 0
        for x, fx in train_loader:

            count += x.shape[0]
            optim.zero_grad()
            if x.shape != fx.shape:
                fx_hat = wrapper_model.forward((x, fx.shape[-2]))
            else:
                fx_hat = wrapper_model.forward(x)
            
            loss = loss_func(fx_hat,fx)
            loss.backward()
            optim.step()
            
        if epoch % 100 == 0:
            mean_losses = 0
            count = 0
            for x, fx in test_loader:
                
                count += x.shape[0]
                if x.shape != fx.shape:
                    fx_hat = wrapper_model.forward((x, fx.shape[-2]))
                else:
                    fx_hat = wrapper_model.forward(x)

                loss = loss_func(fx_hat,fx)
                mean_losses += loss.item()
            print("epoch: {}. mean loss:{}".format(epoch, mean_losses/count))
    return wrapper_model


def save_model(model,args):
    torch.save(model.state_dict(), args.save_name)

def load_saved_model(model, path):
    model.load_state_dict(torch.load(path))
    return
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(args):
    # config = load_config(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dynamics = get_dynamics(args)
    if 'noise' in inspect.getfullargspec(dynamics.get_data).args:
        data = dynamics.get_data(dim = 2 , sample= args.dynamics_samples, noise=args.dynamics_noise)
    else:
        data = dynamics.get_data(dim = 2 , sample= args.dynamics_samples)

    dynamics_model = get_model(args, model_type = "dynamics").to(DEVICE)
    if args.load_traj_model:
        traj_model = get_model(args, model_type='traj').to(DEVICE)
        load_saved_model(traj_model, args.load_traj_model)
    else:
        traj_model = None
    
    if args.wrapper_mode == 'default':
        wrapper_model = wrapper(dynamics_model,  wrapper_mode=args.wrapper_mode)
    elif args.wrapper_mode == 'project_with_knowngx':
        wrapper_model = wrapper(dynamics_model,  wrapper_mode=args.wrapper_mode, gx = traj_model)


    if args.ae_mode:
        ae_model = get_model(args, model_type='autoencoder').to(DEVICE)
        load_saved_model(ae_model, args.load_ae_model)
        wrapper_model.add_ae(ae_model)

    wrapper_model = train(wrapper_model, data, args)
    save_model(wrapper_model.dynamics_model,args)

    return None

if __name__ == "__main__":
    args = get_args()
    main(args)