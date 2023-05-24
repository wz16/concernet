from xmlrpc.client import Boolean
import torch, argparse
from torch.utils.data import DataLoader
import numpy as np
import os, sys
import importlib
from utils.config_load import *
import torch.nn.functional as F
import scipy.integrate
import matplotlib.pyplot as plt
solve_ivp = scipy.integrate.solve_ivp
from models.base_wrapper import wrapper
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils.model_load import get_dynamics, get_model, count_parameters, load_saved_model
import pickle

DEVICE = 'cpu'

def get_args():
    parser = argparse.ArgumentParser(description=None)

    parser.add_argument('--dynamics', default = 'heat_equation', type = str)
    parser.add_argument('--dynamics_noise', default = 0.01, type = float)
    parser.add_argument('--wrapper_mode', default = 'default', type = str)
    parser.add_argument('--ae_mode', action='store_true')
    parser.add_argument('--dynamics_model_def', default = 'MLP', type = str)
    parser.add_argument('--dynamics_model_input_dim', default = 9, type = int)
    parser.add_argument('--dynamics_model_output_dim', default = 9, type = int)
    parser.add_argument('--dynamics_model_hidden_dim', default = 100, type = int)
    parser.add_argument('--traj_model_def', default = 'MLP', type = str)
    parser.add_argument('--traj_model_input_dim', default = 9, type = int)
    parser.add_argument('--traj_model_output_dim', default = 1, type = int)
    parser.add_argument('--traj_model_hidden_dim', default = 100, type = int)
    parser.add_argument('--load_traj_model', default = '', type = str)
    parser.add_argument('--autoencoder_model_def', default = 'AE1D', type = str)
    parser.add_argument('--autoencoder_model_input_dim', default = 101, type = int)
    parser.add_argument('--autoencoder_model_latent_dim', default = 9, type = int)
    parser.add_argument('--load_ae_model', default = 'saved_models/autoencoder_heat_equation.tar', type = str)
    parser.add_argument('--load_dynamics_model', default = 'saved_models/heat_equation_dynamics_noconsv.tar', type = str)
    parser.add_argument('--eval_t_end', default = 1.0, type = float)
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--n_eval', default=10, type=int, help='evaluate trajectory number')
    parser.add_argument('--fig_save_name', default = 'test', type = str)
    return parser.parse_args()


class euler_wrapper:
    def __init__(self, y):
        self.T = y
def forward_euler(fun, t_span, y0, dt = 1e-4, **kwags):
    
    t = t_span[0]
    y = y0
    ys = []
    while t < t_span[-1]:
        ys.append(y)
        y = y + dt*fun(None, y)
        t += dt
    ys = np.asarray(ys)
    t_eval = kwags["t_eval"]
    t_cal = np.linspace(t_span[0], t_span[-1], num=ys.shape[0])
    y_eval = []
    for i in range(ys.shape[1]):
        y_eval.append(np.interp(t_eval, t_cal, ys[:,i]))
    y_eval = np.stack(y_eval).transpose()
    
    return {"y":euler_wrapper(y_eval)}


def eval(model, dynamics, args):

    model.eval()
    if hasattr(dynamics, "random_init"):
        x0s = []
        for i in range(args.n_eval):
            x0s.append(dynamics.random_init())
        x0s = np.concatenate(x0s)
    else:
        raise ValueError("no random init func in dynamics")

    t_end = args.eval_t_end
    t_span = [0,t_end]
    t_eval= np.linspace(0, t_end, 1001)
    kwargs = {'t_eval': t_eval, 'rtol': 1e-10}
    


    def fun(t, np_x):
        x = torch.tensor(np_x, requires_grad=True, dtype=torch.float32).view(1,dynamics.dynamics_dim)

        fx_hat = model.forward(x)
        dx = fx_hat
        dx = (dx).data.numpy().reshape(-1)
        return dx
    true_xs, true_conservations, simu_xs, simu_conservations = [],[],[],[]
    for i in range(args.n_eval):
        x0 = x0s[i]
        true_path = solve_ivp(fun=dynamics.dynamics_fn, t_span=t_span, y0=x0, **kwargs)
        true_x = true_path['y'].T
        simu_path = solve_ivp(fun=fun, t_span=t_span, y0=x0, **kwargs)
        # simu_path = forward_euler(fun=fun, t_span=t_span, y0=x0, **kwargs)dissipation
        simu_x = simu_path['y'].T
        simu_x = np.stack(simu_x, axis=0)
        simu_conservation = dynamics.conservation_fn(simu_x)
        true_conservation = dynamics.conservation_fn(true_x)
        true_xs.append(true_x)
        true_conservations.append(true_conservation)
        simu_xs.append(simu_x)
        simu_conservations.append(simu_conservation)
    true_xs = np.stack(true_xs)
    true_conservations = np.stack(true_conservations)
    simu_xs = np.stack(simu_xs)
    simu_conservations = np.stack(simu_conservations)

    simu_x = simu_xs[0]
    true_x = true_xs[0]
    true_conservation = true_conservations[0]
    simu_conservation = simu_conservations[0]

    # fig, axs = plt.subplots(1, 3,figsize=(12,3))
    # fig.suptitle(args.fig_save_name)

    # if hasattr(dynamics, "conservation_fn"):
    #     simu_conservation = np.stack([dynamics.conservation_fn(c) for c in simu_x])
    #     axs[2].plot(t_eval, simu_conservation, 'b--', label='simu energy', linewidth=1)
    #     true_conservation = np.stack([dynamics.conservation_fn(c) for c in true_x])
    #     axs[2].plot(t_eval, true_conservation, 'b-', label='exact energy', linewidth=1)
    #     axs[2].set_title('exact energy & g(x)', fontsize=10)
    #     axs[2].set_xlabel('t')

    
    # axs[0].plot(t_eval, ((true_x-simu_x)**2).mean(-1))
    # axs[0].set_title("MSE between coordinates")
    # axs[0].set_xlabel('t')
    
    # if args.dynamics_model_output_dim == 2:
    #     axs[1].plot(true_x[:,0], true_x[:,1], 'k-', label='Ground truth', linewidth=0.1,alpha=0.3)
    #     axs[1].plot(simu_x[:,0], simu_x[:,1], 'r-', label='Simulation', linewidth=0.1)
    #     axs[1].axis('equal')
    #     axs[1].legend(fontsize=7)
    #     axs[1].set_title('trajectory', fontsize=10)
    # elif args.dynamics_model_output_dim == 4:

    #     axs[1].plot(true_x[:,0], true_x[:,1], 'k-', label='Ground truth', linewidth=2,alpha=0.3)
    #     axs[1].plot(simu_x[:,0], simu_x[:,1], 'r--', label='Simulation', linewidth=1)
    #     axs[1].axis('equal')
    # elif args.dynamics_model_output_dim == 8:

    #     axs[1].plot(true_x[:,0], true_x[:,1], 'b-', label='Ground truth', linewidth=1)
    #     axs[1].plot(true_x[:,4], true_x[:,5], 'r-', label='Simulation', linewidth=1)
    #     axs[1].plot(simu_x[:,0], simu_x[:,1], 'b--', label='Ground truth', linewidth=1)
    #     axs[1].plot(simu_x[:,4], simu_x[:,5], 'r--', label='Simulation', linewidth=1)
    #     axs[1].axis('equal')

    # plt.tight_layout()
    # # plt.show()
    # fig.savefig('{}{}.png'.format('./figs/', args.fig_save_name))

    simulation = {"true_x":true_x,"simu_x":simu_x,"true_conservation":true_conservation,"simu_conservation":simu_conservation}
    save_simulation(simulation,args)

    with  open("logs/log_error_"+args.dynamics+".txt", "a") as file: 
        file.write("dynamics:{}, noise:{}, seed:{}, wrapper_mode:{}, model:{}, traj_model_output_dim:{},final mse: {}, final conservation: {}, all mse: {}, all conservation: {}\n".format(
            args.dynamics,args.dynamics_noise, args.seed,args.wrapper_mode, args.dynamics_model_def, args.traj_model_output_dim,((true_xs[:,-1,:]-simu_xs[:,-1,:])**2).mean(),((simu_conservations[:,-1]-true_conservations[:,-1])**2).mean(),
            ((true_xs-simu_xs)**2).mean(),((simu_conservations-true_conservations)**2).mean()))


    return 


def save_simulation(simulation, args):
    filename = "./simulations/{}.pickle".format(args.fig_save_name)
    with open(filename, 'wb') as handle:
        pickle.dump(simulation, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return

def eval_ae(model, dynamics, args):

    model.eval()

    if hasattr(dynamics, "random_init"):
        x0s = []
        for i in range(args.n_eval):
            x0s.append(dynamics.random_init())
        x0s = np.concatenate(x0s)
    else:
        raise ValueError("no random init func in dynamics")
    

    print("x0s:{}".format(x0s))
    
    # x0 = x0*5
    t_end = args.eval_t_end
    t_span = [0,t_end]
    t_eval= np.linspace(0, t_end, 1001)
    kwargs = {'t_eval': t_eval, 'rtol': 1e-10}
    def fun(t, np_x):
        x = torch.tensor(np_x, requires_grad=True, dtype=torch.float32).view(1,args.dynamics_model_input_dim)

        fx_hat = model.forward(x)
        dx = fx_hat
        dx = (dx).data.numpy().reshape(-1)
        return dx
    true_xs, true_conservations, simu_xs, simu_conservations = [],[],[],[]
    for i in range(args.n_eval):
        x0 = x0s[i]
        true_path = solve_ivp(fun=dynamics.dynamics_fn, t_span=t_span, y0=x0, **kwargs)
        true_x = true_path['y'].T

        if args.ae_mode:
            x0 = model.ae_model.encode(torch.from_numpy(x0).float()).detach().numpy()
        simu_path = solve_ivp(fun=fun, t_span=t_span, y0=x0, **kwargs)
        # simu_path = forward_euler(fun=fun, t_span=t_span, y0=x0, **kwargs)
        simu_x = simu_path['y'].T
        simu_x = np.stack(simu_x, axis=0)

        if args.ae_mode:
            simu_x = model.ae_model.decode(torch.from_numpy(simu_x).float()).detach().numpy()
            # true_x = model.ae_model.decode(torch.from_numpy(true_x).float()).detach().numpy()
        simu_conservation = dynamics.conservation_fn(simu_x)
        true_conservation = dynamics.conservation_fn(true_x)

        true_xs.append(true_x)
        true_conservations.append(true_conservation)
        simu_xs.append(simu_x)
        simu_conservations.append(simu_conservation)
    true_xs = np.stack(true_xs)
    true_conservations = np.stack(true_conservations)
    simu_xs = np.stack(simu_xs)
    simu_conservations = np.stack(simu_conservations)


    simu_x = simu_xs[0]
    true_x = true_xs[0]
    true_conservation = true_conservations[0]
    simu_conservation = simu_conservations[0]

    simulation = {"true_x":true_x,"simu_x":simu_x,"true_conservation":true_conservation,"simu_conservation":simu_conservation}
    save_simulation(simulation,args)
    with  open("logs/log_error_"+args.dynamics+".txt", "a") as file: 
        file.write("dynamics:{}, noise:{}, seed:{}, wrapper_mode:{}, model:{}, traj_model_output_dim:{},final mse: {}, final conservation: {}, all mse: {}, all conservation: {}\n".format(
            args.dynamics,args.dynamics_noise, args.seed,args.wrapper_mode, args.dynamics_model_def, args.traj_model_output_dim,((true_xs[:,-1,:]-simu_xs[:,-1,:])**2).mean(),((simu_conservations[:,-1]-true_conservations[:,-1])**2).mean(),
            ((true_xs-simu_xs)**2).mean(),((simu_conservations-true_conservations)**2).mean()))

    return 

def load_saved_model(model, path):
    model.load_state_dict(torch.load(path))
    return
      
    
def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dynamics = get_dynamics(args)
    dynamics_model = get_model(args, model_type = "dynamics").to(DEVICE)
    load_saved_model(dynamics_model, args.load_dynamics_model)
    if args.load_traj_model:
        traj_model = get_model(args, model_type='traj').to(DEVICE)
        load_saved_model(traj_model, args.load_traj_model)
    else:
        traj_model = None

    if args.wrapper_mode == 'default':
        wrapper_model = wrapper(dynamics_model,  wrapper_mode=args.wrapper_mode)
    elif args.wrapper_mode == 'project_with_knowngx':
        wrapper_model = wrapper(dynamics_model,  wrapper_mode=args.wrapper_mode, gx = traj_model)
    elif args.wrapper_mode == 'half_project_with_knowngx':
        wrapper_model = wrapper(dynamics_model,  wrapper_mode=args.wrapper_mode, gx = traj_model)

    if args.ae_mode:
        
        ae_model = get_model(args, model_type='autoencoder').to(DEVICE)
        load_saved_model(ae_model, args.load_ae_model)
        wrapper_model.add_ae(ae_model)
        eval_ae(wrapper_model, dynamics, args)
    else:
        eval(wrapper_model, dynamics, args)
    

    # model = get_model(args).to(DEVICE)


    return None

if __name__ == "__main__":
    args = get_args()
    main(args)