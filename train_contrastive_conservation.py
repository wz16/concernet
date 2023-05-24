from pickle import NONE
import torch, argparse
from torch.utils.data import DataLoader,Dataset
import numpy as np
import os, sys
import importlib
from utils.config_load import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from utils.model_load import get_dynamics, get_model, count_parameters, load_saved_model
matplotlib.rcParams.update({'font.size': 12})
import pickle
import inspect
import scipy

DEVICE = 'cuda'


def get_args():
    parser = argparse.ArgumentParser(description=None)

    parser.add_argument('--dynamics', default = 'ideal_spring_mass', type = str)
    parser.add_argument('--dynamics_samples', default = 50, type = int)
    parser.add_argument('--dynamics_noise', default = 0.0, type = float)
    parser.add_argument('--ae_mode', action='store_true', default=False)
    parser.add_argument('--autoencoder_model_def', default = 'AE1D', type = str)
    parser.add_argument('--autoencoder_model_input_dim', default = 101, type = int)
    parser.add_argument('--autoencoder_model_latent_dim', default = 9, type = int)
    parser.add_argument('--load_ae_model', default = 'saved_models/autoencoder_heat_equation.tar', type = str)
    parser.add_argument('--traj_model_def', default = 'MLP', type = str)
    parser.add_argument('--traj_model_input_dim', default = 2, type = int)
    parser.add_argument('--traj_model_output_dim', default = 1, type = int)
    parser.add_argument('--traj_model_hidden_dim', default = 100, type = int)
    parser.add_argument('--loss_func', default = 'square_ratio', type = str)
    parser.add_argument('--epochs', default = 1000, type = int)
    parser.add_argument('--batch_size', default = 10, type = int)
    parser.add_argument('--learning_rate', default = 1e-3, type = float)
    parser.add_argument('--seed', default=20, type=int, help='random seed')
    parser.add_argument('--save_name', default = './saved_models/traj_test.tar', type = str)

    return parser.parse_args()

def draw_matrix_heatmap(M):
    uniform_data = np.random.rand(10, 12)
    fig,axes = plt.subplots(1,1, figsize=(6,6))
    sns.heatmap(M,ax=axes)
    plt.show()
    fig.savefig('{}.png'.format('./figs/similaritymatrix'))

def MSE(x, y):
    return ((x-y)**2).mean(), 0


def square_ratio_loss(features, x = None):
    num_label = features.shape[0]
    len_traj = features.shape[1]
    num_traj_output = features.shape[2]

    assert num_label != 1
    assert len_traj != 1
    labels = [[i for j in range(len_traj)] for i in range(num_label)]
    labels = torch.Tensor(labels).to(torch.int32).flatten()

    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(DEVICE)
    
    features = features.reshape(num_label*len_traj, 1, -1)

    similarity_matrix = torch.sum(-(features- torch.transpose(features,0,1))**2, dim = 2)
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(DEVICE)
    masked_labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    positives = similarity_matrix[masked_labels.bool()].view(masked_labels.shape[0], -1)
    negatives = similarity_matrix[~masked_labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    ratio_loss = (positives/(logits.sum(-1)).unsqueeze(-1)).mean()
        

    return ratio_loss

def train_traj_repre(model, data, dynamics, args, ae_model=None):
    
    
    model.train()
    epochs = args.epochs

    x = torch.tensor( data['x'], requires_grad=True, dtype=torch.float32).to(DEVICE)
    test_x = torch.tensor( data['test_x'], requires_grad=True, dtype=torch.float32).to(DEVICE)
    xn1 = torch.Tensor(data['fx']).to(DEVICE)
    test_xn1 = torch.Tensor(data['test_fx']).to(DEVICE)

    if ae_model:
        x = ae_model.encode(x).detach().data
        test_x = ae_model.encode(test_x).detach().data

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
    if args.loss_func == 'info_nce':
        loss_func = info_nce_loss
    elif  args.loss_func == "square_ratio":
        loss_func = square_ratio_loss
    else:
        raise Exception("invalid loss function")

    for epoch in range(epochs):

       
        count = 0
        for x, fx in train_loader:
            count += 1

            optim.zero_grad()
            fx_hat = model.forward(x)
            loss = loss_func(fx_hat,x)
            loss.backward()
            optim.step()

        if epoch % 100 == 0:

            mean_losses = 0
            count = 0
            for x, fx in test_loader:
                
                fx_hat = model.forward(x)
                loss= loss_func(fx_hat,x)


                def normalize(cc):
                    m = torch.mean(cc)
                    std = torch.std(cc)
                    return (cc-m)/std
                
                mean_losses += loss
                x_numpy = x.cpu().detach().numpy()
                x_numpy = np.reshape(x_numpy, (-1, x_numpy.shape[-1]))
                count += 1

            print("epoch:{}, mean_loss:{} ".\
                format(epoch, mean_losses/count))
    
    return model


def find_fitting_coef(model, data, dynamics, args, ae_model=None):
    test_x = torch.tensor( data['test_x'], requires_grad=True, dtype=torch.float32).to(DEVICE)
    test_xn1 = torch.Tensor(data['test_fx']).to(DEVICE)

    if ae_model:
        # x = ae_model.encode(x).detach().data
        test_x = ae_model.encode(test_x).detach().data
    test_tuple = []
    for i in range(test_x.shape[0]):
      test_tuple.append((test_x[i],test_xn1[i]))
    test_loader = DataLoader(test_tuple, batch_size = args.batch_size, shuffle= True)

    fx_hat_record = []
    conservation_sum = []
    for x, fx in test_loader:
        fx_hat = model.forward(x)
        fx_hat_record.append(fx_hat)
        x_numpy = x.detach().cpu().numpy()
        conservation = dynamics.conservation_fn(x_numpy)
        conservation_sum.append(conservation)
    conservation_sum = np.stack(conservation_sum)[...,0].flatten()
    fx_hat_sum = torch.concat(fx_hat_record).detach().cpu().numpy().flatten()
    print("conservation_sum:{},fx_hat_sum:{}".format(conservation_sum.shape,fx_hat_sum.shape))
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(fx_hat_sum, conservation_sum)

    coefs = np.polyfit(fx_hat_sum,conservation_sum, deg = 1)

    err = np.sum((fx_hat_sum*coefs[0]+coefs[1]-conservation_sum)**2)/conservation_sum.shape[0]
    
    with open("logs/log_sq_error.txt", "a") as file: 
        file.write("dynamics:{}, seed:{}, dynamics_samples:{}, batch_size:{} dynamics_noise:{}, err:{}, r2:{}\n".format(args.dynamics, args.seed, args.dynamics_samples, args.batch_size, args.dynamics_noise, err,r_value**2))

def save_model(model,args):

    torch.save(model.state_dict(), args.save_name)

    print("save file to:{}".format(args.save_name))


def visualize_2d(model,dynamics, args):
    delta = 0.01*100
    minx = -0*100
    maxx = 1*100
    x = np.arange(minx,maxx+delta, delta)
    y = np.arange(minx,maxx+delta, delta)
    X, Y = np.meshgrid(x, y)

    input = torch.cat((torch.from_numpy(X).unsqueeze(-1),torch.from_numpy(Y).unsqueeze(-1)),dim = -1).to(torch.float32)
    if input.shape[-1] != args.traj_model_input_dim:
        input_ = torch.zeros((input.shape[0],input.shape[1],args.traj_model_input_dim))
        input_[..., -input.shape[-1]:] = input
    else:
        input_ = input
    output = model(input_.to(DEVICE))[...,0]

    output = output.squeeze(-1)
    Z = output.cpu().detach().numpy()

    fig,axes = plt.subplots(2,1, figsize=(4,7))
    clev = np.arange(Z.min(),Z.max(),100)
    cs = axes[0].contourf(X, Y, Z, cmap=plt.cm.coolwarm,extend='both')
    cs.cmap.set_over('red')
    cs.cmap.set_under('blue')
    cs.changed()

    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(cs, cax=cax, orientation='vertical')
    axes[0].set_xlim(minx,maxx)
    axes[0].set_ylim(minx,maxx)
    axes[0].set_xticks(np.linspace(minx,maxx,3))
    axes[0].set_yticks(np.linspace(minx,maxx,3))
    axes[0].axis('equal')
    axes[0].title.set_text(r'Learned Conservation')
    # axes[0].plt.colorbar(cs)


    if hasattr(dynamics, "conservation_fn"):
        axes[1].title.set_text(r'True Conservation')
        conservation = np.zeros_like(Z)
        input_ = input_.numpy()
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                conservation[i][j] = dynamics.conservation_fn(input_[i][j])[0]
        conservation[np.isneginf(conservation)] = 0
        cs1 = axes[1].contourf(X, Y, conservation, cmap=plt.cm.coolwarm,extend='both')
        cs1.cmap.set_over('red')
        cs1.cmap.set_under('blue')
        cs1.changed()
        
        axes[1].set_xlim(minx,maxx)
        axes[1].set_ylim(minx,maxx)
        axes[1].set_yticks(np.linspace(minx,maxx,3))
        axes[1].set_xticks(np.linspace(minx,maxx,3))
        divider1 = make_axes_locatable(axes[1])
        cax1 = divider1.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(cs1, cax=cax1, orientation='vertical')
        axes[1].axis('equal')
        
        

    fig.savefig('{}{}.png'.format('./figs/conservation_',args.dynamics))

def visualize_4d(model,dynamics, args, data = None):
    sample = 10
    if not data:
        traj = dynamics.get_data(dim = 3 , test_split = 1, sample = sample, noise=0)['x']
    else:
        traj = data['x'][0:sample]
    x = torch.Tensor(traj).to(DEVICE)

    fx_hat = model.forward(x)
    loss = info_nce_loss(fx_hat) 
    print("visualize loss:{}".format(loss))
    t = np.linspace(0,10,traj.shape[1])
    


    num_subplot = [2,5]
    if hasattr(dynamics, "conservation_fn"):
        fig,axes = plt.subplots(num_subplot[0],num_subplot[1], figsize=(15,9))
        true_conservation = np.stack([dynamics.conservation_fn(c) for c in traj]).reshape(sample, -1, 2)
        for i in range(sample):
            axes[1,2].scatter(true_conservation[i,:,0],true_conservation[i,:,1])
            axes[0,0].plot(t,true_conservation[i,:,0])
            axes[0,1].plot(t,true_conservation[i,:,1])
            
            # print("i:{},range dim 0:{},range dim 1:{}".format(i, (np.min(true_conservation[i,:,0])), (np.min(true_conservation[i,:,1]))))
        axes[1,2].set_xlabel("true consv 0 vs true consv 1")
        axes[0,0].set_xlabel("true consv 0 vs t")
        axes[0,1].set_xlabel("true consv 1 vs t")

        traj_torch = torch.from_numpy(traj).float().to(DEVICE)

        learned_conservation = np.stack([model(c).cpu().detach().numpy() for c in traj_torch])
        for i in range(sample):
            for j in range(learned_conservation.shape[-1]):
                axes[0,j+2].plot(t,learned_conservation[i,:,j])
                axes[0,j+2].set_xlabel("learn consv vs t")
        

    for i in range(sample):
        axes[1,0].plot(traj[i,:,0],traj[i,:,1])
    axes[1,0].set_xlabel("traj")
    fig.tight_layout(pad=1.0)

    fig.savefig('{}{}.png'.format('./figs/conservation2_',args.dynamics))

    simulation = {"learned_conservation":learned_conservation,"traj":traj,"true_conservation":true_conservation}
    save_simulation(simulation,args)


def visualize_6d(model,dynamics, args):
    delta = 0.05
    minx = -1
    maxx = 1
    v1 = np.arange(minx,maxx+delta, delta)
    v2 = np.arange(minx,maxx+delta, delta)
    m1 = v1*0+15
    m2 = v1*0+15
    x1 = v1*0+0
    x2 = v1*0+1
    M1, M2 = np.meshgrid(m1, m2)
    M1, M2 = torch.from_numpy(M1).unsqueeze(-1),torch.from_numpy(M2).unsqueeze(-1)
    X1, X2 = np.meshgrid(x1, x2)
    X1, X2 = torch.from_numpy(X1).unsqueeze(-1),torch.from_numpy(X2).unsqueeze(-1)
    V1_np, V2_np = np.meshgrid(v1, v2)
    V1, V2 = torch.from_numpy(V1_np).unsqueeze(-1),torch.from_numpy(V2_np).unsqueeze(-1)

    input = torch.cat((M1,X1,V1,M2,X2,V2),dim = -1).to(torch.float32)
    if input.shape[-1] != args.traj_model_input_dim:
        input_ = torch.zeros((input.shape[0],input.shape[1],args.traj_model_input_dim))
        input_[..., -input.shape[-1]:] = input
    else:
        input_ = input
    output = model(input_.to(DEVICE))[...,0]

    output = output.squeeze(-1)
    Z = output.cpu().detach().numpy()

    fig,axes = plt.subplots(2,1, figsize=(4,7))
    clev = np.arange(Z.min(),Z.max(),100)
    cs = axes[0].contourf(V1_np, V2_np, Z, cmap=plt.cm.coolwarm,extend='both')
    cs.cmap.set_over('red')
    cs.cmap.set_under('blue')
    cs.changed()

    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(cs, cax=cax, orientation='vertical')
    axes[0].set_xlim(minx,maxx)
    axes[0].set_ylim(minx,maxx)
    axes[0].axis('equal')
    axes[0].title.set_text(r'Learned Conservation')
    # axes[0].plt.colorbar(cs)


    if hasattr(dynamics, "conservation_fn"):
        axes[1].title.set_text(r'True Conservation')
        conservation = np.zeros_like(Z)
        input_ = input_.numpy()
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                conservation[i][j] = dynamics.conservation_fn(input_[i][j])[0]
        conservation[np.isneginf(conservation)] = 0
        cs1 = axes[1].contourf(V1_np, V2_np, conservation, cmap=plt.cm.coolwarm,extend='both')
        cs1.cmap.set_over('red')
        cs1.cmap.set_under('blue')
        cs1.changed()
        
        axes[1].set_xlim(minx,maxx)
        axes[1].set_ylim(minx,maxx)
        divider1 = make_axes_locatable(axes[1])
        cax1 = divider1.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(cs1, cax=cax1, orientation='vertical')
        axes[1].axis('equal')
        

    fig.savefig('{}{}.png'.format('./figs/conservation_',args.dynamics))

def visualize_2conservation_2d(model,dynamics, args, ae_model = None, data = None):
    sample = 10
    if not data:
        traj = dynamics.get_data(dim = 3 , test_split = 1, sample = sample, noise=0)['x']
    else:
        traj = data['x'][0:sample]
    x = torch.Tensor(traj).to(DEVICE)
    if ae_model:
        x = ae_model.encode(x).detach().data
    fx_hat = model.forward(x)
    loss = info_nce_loss(fx_hat) 
    print("visualize loss:{}".format(loss))

    


    num_subplot = [2,3]
    if hasattr(dynamics, "conservation_fn"):
        fig,axes = plt.subplots(num_subplot[0],num_subplot[1], figsize=(9,9))
        true_conservation = np.stack([dynamics.conservation_fn(c) for c in traj]).reshape(sample, -1, 2)
        for i in range(sample):
            axes[0,2].scatter(true_conservation[i,:,0],true_conservation[i,:,1])
            axes[0,0].plot(true_conservation[i,:,0])
            axes[0,1].plot(true_conservation[i,:,1])
            
            # print("i:{},range dim 0:{},range dim 1:{}".format(i, (np.min(true_conservation[i,:,0])), (np.min(true_conservation[i,:,1]))))
        axes[0,2].set_xlabel("true consv 0 vs true consv 1")
        axes[0,0].set_xlabel("true consv 0 vs t")
        axes[0,1].set_xlabel("true consv 1 vs t")

        traj_torch = torch.from_numpy(traj).float().to(DEVICE)
        if ae_model:
            learned_conservation = np.stack([model(ae_model.encode(c)).cpu().detach().numpy() for c in traj_torch])
        else:
            learned_conservation = np.stack([model(c).cpu().detach().numpy() for c in traj_torch])
        for i in range(sample):
            axes[1,1].plot(learned_conservation[i,:,0])
        axes[1,1].set_xlabel("learn consv vs t")
        

    for i in range(sample):
        axes[1,0].plot(traj[i,:,0],traj[i,:,1])
    axes[1,0].set_xlabel("traj")
    fig.savefig('{}{}.png'.format('./figs/conservation2_',args.dynamics))

    simulation = {"learned_conservation":learned_conservation,"traj":traj,"true_conservation":true_conservation}
    save_simulation(simulation,args)

    return 


def visualize_conservation_ae(model,dynamics, args, ae_model = None, data = None):
    sample = 10
    if not data:
        traj = dynamics.get_data(dim = 3 , test_split = 1, sample = sample, noise=0)['x']
    else:
        traj = data['x'][0:sample]
    x = torch.Tensor(traj).to(DEVICE)
    if ae_model:
        x = ae_model.encode(x).detach().data
    fx_hat = model.forward(x)
    loss = info_nce_loss(fx_hat) 
    print("visualize loss:{}".format(loss))

    t = np.linspace(0,10,traj.shape[1])

    num_subplot = [1,2]
    if hasattr(dynamics, "conservation_fn"):
        fig,axes = plt.subplots(num_subplot[0],num_subplot[1], figsize=(9,5))
        true_conservation = np.stack([dynamics.conservation_fn(c) for c in traj]).reshape(sample, -1, 1)
        for i in range(sample):
            axes[0].plot(t, true_conservation[i,:,0])
            
            # print("i:{},range dim 0:{},range dim 1:{}".format(i, (np.min(true_conservation[i,:,0])), (np.min(true_conservation[i,:,1]))))
        # axes[0,2].set_xlabel("true consv 0 vs true consv 1")
        axes[0].set_xlabel("true consv 0 vs t")
        traj_torch = torch.from_numpy(traj).float().to(DEVICE)
        if ae_model:
            learned_conservation = np.stack([model(ae_model.encode(c)).cpu().detach().numpy() for c in traj_torch])
        else:
            learned_conservation = np.stack([model(c).cpu().detach().numpy() for c in traj_torch])
        for i in range(sample):
            axes[1].plot(t,learned_conservation[i,:,0])
        axes[1].set_xlabel("learn consv vs t")
        

    fig.savefig('{}{}.png'.format('./figs/conservation2_',args.dynamics))

    simulation = {"learned_conservation":learned_conservation,"traj":traj,"true_conservation":true_conservation}
    save_simulation(simulation,args)

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(learned_conservation.flatten(), true_conservation[...,0].flatten())
    with  open("log_r_value.txt", "a") as file:
        file.write("dynamics:{}, method: normalize_{}, seed:{}, dynamics_samples:{}, batch_size:{} dynamics_noise:{}, r2:{}\n".format(
           args.dynamics, args.nce_normalize, args.seed, args.dynamics_samples, args.batch_size, args.dynamics_noise, r_value**2))
    return 

def save_simulation(simulation, args):
    filename = "./simulations/{}_contrastive.pickle".format(args.dynamics)
    with open(filename, 'wb') as handle:
        pickle.dump(simulation, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def load_saved_model(model, path):
    model.load_state_dict(torch.load(path))
    return
      
def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    model = get_model(args, model_type = 'traj').to(DEVICE)
    print("model_para_number:"+str(count_parameters(model)))        
    
    dynamics = get_dynamics(args)
    
    if 'noise' in inspect.getfullargspec(dynamics.get_data).args:
        data = dynamics.get_data(dim = 3 , sample= args.dynamics_samples, noise=args.dynamics_noise)
    else:
        data = dynamics.get_data(dim = 3 , sample= args.dynamics_samples)
    # if args.load_name:
    #     model = get_model(args, model_type='traj').to(DEVICE)
    #     load_saved_model(model, args.load_name)
    if args.ae_mode:
        print("ae mode = True")
        ae_model = get_model(args, model_type='autoencoder').to(DEVICE)
        load_saved_model(ae_model, args.load_ae_model)
    else:
        ae_model = None
    model = train_traj_repre(model, data, dynamics, args, ae_model)

    data = dynamics.get_data(dim = 3 , sample= 1000, noise=0)
    find_fitting_coef(model,data, dynamics, args, ae_model)
    
    save_model(model,args)        
    return None

if __name__ == "__main__":
    args = get_args()
    main(args)