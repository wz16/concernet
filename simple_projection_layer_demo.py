import torch, argparse
from torch.utils.data import DataLoader
import numpy as np
from utils.config_load import *
from models.base_wrapper import wrapper
from utils.model_load import get_dynamics, get_model, count_parameters, load_saved_model
import inspect
import scipy.integrate
import matplotlib.pyplot as plt
solve_ivp = scipy.integrate.solve_ivp
# from helper import *


def get_data(data, args):
    """ Load data and create train and test loader """
    # load data
    x = torch.tensor( data['x'], requires_grad=True, dtype=torch.float32).to(DEVICE)
    fx = torch.Tensor(data['fx']).to(DEVICE)
    test_x = torch.tensor( data['test_x'], requires_grad=True, dtype=torch.float32).to(DEVICE)
    test_fx = torch.Tensor(data['test_fx']).to(DEVICE)
    print("train data shape:{}".format(x.shape))

    # create train and test loader
    train_tuple = []
    for i in range(x.shape[0]):
      train_tuple.append((x[i],fx[i]))
    train_loader = DataLoader(train_tuple, batch_size =  args.batch_size, shuffle= True)

    test_tuple = []
    for i in range(test_x.shape[0]):
      test_tuple.append((test_x[i],test_fx[i]))
    test_loader = DataLoader(test_tuple, batch_size =  args.batch_size , shuffle= True)
    return train_loader, test_loader

def plot_traj(t_eval, true_xs, simu_xs):
    """ Plot trajectory of the system wrt time """
    plt.figure()
    true_xs = true_xs[0]
    A_values = true_xs[:, 0]
    C_values = true_xs[:, 1]
    plt.plot(t_eval, A_values, color='b', label="A")
    plt.plot(t_eval, C_values, color='g', label="C")

    simu_xs = simu_xs[0]
    A_values = simu_xs[:, 0]
    C_values = simu_xs[:, 1]
    plt.plot(t_eval, A_values, color='b', label="A_sim", linestyle='--')
    plt.plot(t_eval, C_values, color='g', label="C_sim", linestyle='--')

    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.title("Concentration profile ({})".format(args.wrapper_mode))
    plt.savefig("./figs/conc_profile_{}.png".format(args.wrapper_mode))

def pipeline(args, train_loader, test_loader, eval_x0s):
    # get dynamics and trajectory model
    dynamics_model = get_model(args, model_type = "dynamics").to(DEVICE)

    if args.wrapper_mode == 'default':
        wrapper_model = wrapper(dynamics_model, wrapper_mode=args.wrapper_mode)
    elif args.wrapper_mode == 'project_with_knowngx':
        wrapper_model = wrapper(dynamics_model, wrapper_mode=args.wrapper_mode, gx=dynamics.conservation_fn)

    # Define loss function
    loss_func  = torch.nn.MSELoss(reduction='sum')

    # Define optimizer
    optim = torch.optim.Adam(wrapper_model.dynamics_model.parameters(), args.learning_rate, weight_decay=0)

    # Train
    train(wrapper_model, train_loader, test_loader, optim, loss_func, args)          
    # save_model(wrapper_model.dynamics_model, args)

    eval(wrapper_model, eval_x0s, args)

    # # Plot
    # visualize_2d(wrapper_model, dynamics, args)
    return wrapper_model


def train(wrapper_model, train_loader, test_loader, optim, loss_func, args):
    """ Train the wrapper model """
    wrapper_model.train()
    for epoch in range(args.epochs):
        for x, fx in train_loader:
            optim.zero_grad()
            fx_hat = forward(wrapper_model, x, fx)
            loss = loss_func(fx_hat, fx)
            loss.backward()
            optim.step()
        
        # evaluate
        if epoch % 10 == 0:
            wrapper_model.eval()
            mean_loss = evaluate(wrapper_model, test_loader, loss_func)
            wrapper_model.train()
            print("epoch: {}. mean loss:{}".format(epoch, mean_loss))


def evaluate(wrapper_model, test_loader, loss_func):
    """ Evaluate the model on test data """
    mean_losses = 0
    count = 0
    for x, fx in test_loader:
        count += x.shape[0]
        fx_hat = forward(wrapper_model, x, fx)
        loss = loss_func(fx_hat, fx)
        mean_losses += loss.item()
    return mean_losses/count


def forward(wrapper_model, x, fx):
    """ Forward pass of the wrapper model """
    if x.shape != fx.shape:
        fx_hat = wrapper_model.forward((x, fx.shape[-2]))
    else:
        fx_hat = wrapper_model.forward(x)
    return fx_hat

def eval(model, x0s, args):
    model.eval()
    model.to('cpu')


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

    fig, axs = plt.subplots(1, 3,figsize=(12,3))
    fig.suptitle(args.wrapper_mode)

    if hasattr(dynamics, "conservation_fn"):
        simu_conservation = np.stack([dynamics.conservation_fn(c) for c in simu_x])
        axs[2].plot(t_eval, simu_conservation, 'r--', label='Simu conservation', linewidth=1)
        true_conservation = np.stack([dynamics.conservation_fn(c) for c in true_x])
        axs[2].plot(t_eval, true_conservation, 'r', label='Ground truth conservation', alpha=0.3, linewidth = 3)
        axs[2].set_title('Conservation', fontsize=10)
        axs[2].set_xlabel('time')
        axs[2].legend(fontsize=7)
        axs[2].set_ylim([
            min(np.min(simu_conservation), np.min(true_conservation))-0.1, 
            max(np.max(simu_conservation), np.max(true_conservation))+0.1
            ])

    
    axs[0].plot(t_eval, ((true_x-simu_x)**2).mean(-1))
    axs[0].set_title("MSE between coordinates")
    axs[0].set_xlabel('t')
    
    if args.dynamics_model_output_dim == 2 or args.dynamics_model_output_dim == 3:
        # axs[1].plot(true_x[:,0], true_x[:,1], 'b-', label='Ground truth', linewidth=1, alpha=0.3)
        # axs[1].plot(simu_x[:,0], simu_x[:,1], 'r-', label='Simulation', linewidth=1)

        axs[1].plot(t_eval, true_x[:,0], 'g', label='x1 Ground truth', alpha=0.3, linewidth = 3)
        axs[1].plot(t_eval, true_x[:,1], 'b', label='x2 Ground truth', alpha=0.3, linewidth = 3)
        axs[1].plot(t_eval, simu_x[:,0], 'g', label='x1 Simu {}'.format(args.wrapper_mode), linestyle='--')
        axs[1].plot(t_eval, simu_x[:,1], 'b', label='x2 Simu {}'.format(args.wrapper_mode), linestyle='--')
        axs[1].set_xlabel('time')
        axs[1].legend(fontsize=7)
        axs[1].set_title('System evolution', fontsize=10)

    plt.tight_layout()
    fig.savefig('{}demo_{}.png'.format('./figs/', args.wrapper_mode))
    # plot_traj(t_eval, true_xs, simu_xs)

def get_args():
    parser = argparse.ArgumentParser(description=None)
    # dynamics model is for f(x) = dx/dt
    parser.add_argument('--dynamics', default = 'chemical_kinematics', type = str)
    parser.add_argument('--dynamics_samples', default = 50, type = int)
    parser.add_argument('--dynamics_noise', default = 0.1, type = float)
    parser.add_argument('--dynamics_model_def', default = 'MLP_ReLU', type = str)
    parser.add_argument('--dynamics_model_input_dim', default = 2, type = int)
    parser.add_argument('--dynamics_model_output_dim', default = 2, type = int)
    parser.add_argument('--dynamics_model_hidden_dim', default = 100, type = int)


    # hyperparameters
    parser.add_argument('--epochs', default = 200, type = int)
    parser.add_argument('--batch_size', default = 100, type = int)
    parser.add_argument('--learning_rate', default = 1e-4, type = float)

    # wrapper model
    parser.add_argument('--wrapper_mode', default = 'project_with_knowngx', type = str)
    
    # eval
    parser.add_argument('--n_eval', default = 5, type = int)
    parser.add_argument('--eval_t_end', default = 10.0, type = float)

    parser.add_argument('--seed', default=0, type=int, help='random seed')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # generate data
    dynamics = get_dynamics(args)
    print(dynamics)
    if 'noise' in inspect.getfullargspec(dynamics.get_data).args:
        data = dynamics.get_data(dim = 2 , sample= args.dynamics_samples, noise=args.dynamics_noise)
    else:
        data = dynamics.get_data(dim = 2 , sample= args.dynamics_samples)
    train_loader, test_loader = get_data(data, args)

    if hasattr(dynamics, "random_init"):
        eval_x0s = []
        for i in range(args.n_eval):
            eval_x0s.append(dynamics.random_init())
        eval_x0s = np.concatenate(eval_x0s)
    else:
        raise ValueError("no random init func in dynamics")

    args.default='project_with_knowngx'
    pipeline(args,train_loader,test_loader, eval_x0s)
    args.wrapper_mode='default'
    pipeline(args,train_loader,test_loader, eval_x0s)