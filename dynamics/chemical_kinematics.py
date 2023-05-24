
# code reused from https://github.com/greydanus/hamiltonian-nn
# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import torch
import autograd
import autograd.numpy as np
import math
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp


class dynamics():
    
    def __init__(self,**kwargs):
        super().__init__()
        self.dynamics_dim = 2
        self.EA1 = 5.74
        self.EA2 = 7.47
        self.S = 10
        self.R = 8.31
        self.T = 250
        self.M = 1000
        self.k1 = self.S*math.exp(-self.EA1/self.R/self.T*self.M)
        self.k2 = self.S*math.exp(-self.EA2/self.R/self.T*self.M)

        return None
    def dynamics_fn(self, t, coords):

        S = np.matmul([[-self.k1,self.k2],[self.k1,-self.k2]],coords)
        return S

    def conservation_fn(self, coords):
        x1, x2 = np.split(coords,2,axis=-1)
        mass = x1+x2
        return mass

    def random_init(self, batch_dim = 1):
        y0 = np.random.rand(batch_dim,2)
        radius = np.random.rand(batch_dim,1)*200
        y0 = y0 / np.sqrt((y0**2).sum()) * radius
        return y0

    
    def get_trajectory(self, t_span=[0,3], timescale=10, radius=None, y0=None, noise_std=0.1, **kwargs):
        t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])))
        if y0 is None:
            y0 = np.random.rand(2)*2
        if radius is None:
            radius = np.random.rand()*100 + 0.1 
        y0 = y0 / np.sqrt((y0**2).sum()) * radius

        spring_ivp = solve_ivp(fun=self.dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10, **kwargs)
        y = spring_ivp['y']
        dydt = [self.dynamics_fn(None, y) for y in spring_ivp['y'].T]
        dydt = np.stack(dydt).T
        y += np.random.randn(*y.shape)*noise_std
        dydt += np.random.randn(*dydt.shape)*noise_std
        
        return y, dydt, t_eval

    
    def get_data(self, sample=50, test_split=0.5 ,dim = 2, noise=0.01):

        xs, dxs, ts = [], [], []
        for i in range(sample):
            x, dx, t = self.get_trajectory(noise_std=noise)
            xs.append( np.stack(x).T )
            dxs.append( np.stack(dx).T )
            ts.append( np.stack(t).T )

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

