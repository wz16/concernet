# code reused from https://github.com/greydanus/hamiltonian-nn
# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import torch
import autograd
import autograd.numpy as np

import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

try:
    from .dynamics_utils import base_dynamics
except:
    from dynamics_utils import base_dynamics

import matplotlib.pyplot as plt


class dynamics():
    
    def __init__(self,**kwargs):
        super().__init__()
        self.dynamics_dim = 4
        self.m = 1
        self.k = 1
        return None
    def dynamics_fn(self, t, coords):
        dcoords = np.zeros_like(coords)
        x = coords[0]
        y = coords[1]
        r = np.sqrt(x**2+y**2)
        dcoords[2] = -1/(r**3)*x*self.k/self.m
        dcoords[3] = -1/(r**3)*y*self.k/self.m
        dcoords[0] = coords[2]
        dcoords[1] = coords[3]
        return dcoords

    def conservation_fn(self, coords):
        if len(coords.shape)==1:
            r = np.sqrt(coords[0]**2+coords[1]**2)
            v = np.sqrt(coords[2]**2+coords[3]**2)
            H = self.m*0.5*v**2-1/r*self.k
            L = coords[0]*coords[3]-coords[1]*coords[2]
            result = np.array([H,L])
        else:
            r = np.sqrt(coords[...,0]**2+coords[...,1]**2)
            v = np.sqrt(coords[...,2]**2+coords[...,3]**2)
            H = self.m*0.5*v**2-1/r*self.k
            L = coords[...,0]*coords[...,3]-coords[...,1]*coords[...,2]
            result = np.stack((H,L),axis=-1)
        return result

    def random_init(self, batch_dim = 1):
        y0 = np.random.rand(batch_dim,2)*2-1
        dy0 = np.array([y0[...,1], -y0[...,0]]).T
        radius = np.abs(np.random.rand(batch_dim,1)*0.9 + 0.1)
        y0 = y0 / np.sqrt((y0**2).sum()) * radius

        radius2 = np.sqrt(1/radius*(np.random.rand()*0.3 + 1)*self.k/self.m)
        
        dy0 = dy0 / np.sqrt((dy0**2).sum()) * radius2

        init = np.concatenate((y0,dy0),axis=1)

        return init
    
    def get_trajectory(self, t_span=[0,10], timescale=10, radius=None, y0=None, noise_std=0.1, **kwargs):
        t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])))

        if y0 is None:
            y0 = np.squeeze(self.random_init(), axis = 0)

        spring_ivp = solve_ivp(fun=self.dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10, **kwargs)
        coords = spring_ivp['y'].T
        dcoords = np.stack([self.dynamics_fn(None, y) for y in coords], axis = 0)        

        coords += np.random.randn(*coords.shape)*noise_std
        dcoords += np.random.randn(*dcoords.shape)*noise_std
        return coords,dcoords, t_eval
    
    def get_data(self, sample=50, test_split=0.5, dim = 2, noise = 0.01):

        # x = self.random_init(sample)
        # dxdt = self.dynamics_fn(None, x)
        xs, dxs, ts = [], [], []
        for s in range(sample):
            x, dx, t = self.get_trajectory(noise_std=noise)
            if s == 0:
                draw_traj(x)
            xs.append(x)
            
            e = np.stack([self.conservation_fn(x_ )for x_ in x])

            dxs.append(dx)
            ts.append( np.stack(t).T )
            # dxs.append(e)
        
        data = {}
        if dim == 2:
            data['x'] = np.concatenate(xs)
            data['fx'] = np.concatenate(dxs).squeeze()
            data['t'] = np.concatenate(ts).squeeze()
        elif dim == 3:
            data['x'] = np.stack(xs)
            data['fx'] = np.stack(dxs).squeeze()
            data['t'] = np.stack(ts).squeeze()


        split_ix = int(len(data['x']) * test_split)
        split_data = {}
        for k in ['x', 'fx', 't']:
            split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
        data = split_data
        return data


def draw_traj(x):
    fig, axs = plt.subplots(1, 1,figsize=(4,3))
    axs.scatter(x[:,0],x[:,1])
    fig.savefig('{}{}.png'.format('./figs/', 'kepler'))


def test():
    for i in range(10):
        test_dynamics = dynamics()
        data = test_dynamics.get_data(sample=1)
        

if __name__ == "__main__":
    test()
