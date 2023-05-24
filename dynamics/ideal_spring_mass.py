
# code reused from https://github.com/greydanus/hamiltonian-nn
# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import torch
import autograd
import autograd.numpy as np

import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

class dynamics():
    
    def __init__(self,**kwargs):
        super().__init__()
        self.dynamics_dim = 2
        return None
    def dynamics_fn(self, t, coords):
        dcoords = autograd.elementwise_grad(self.conservation_fn)(coords)
        dqdt, dpdt = np.split(dcoords,2)
        S = np.concatenate([dpdt, -dqdt], axis=0)
        return S

    def conservation_fn(self, coords):
        q, p = np.split(coords,2,axis=-1)
        H = p**2 + q**2 # spring hamiltonian (linear oscillator)
        return H

    def random_init(self, batch_dim = 1):
        y0 = np.random.rand(batch_dim,2)*2-1
        radius = np.random.rand(batch_dim,1)*0.9 + 0.3
        y0 = y0 / np.sqrt((y0**2).sum()) * radius
        return y0
    
    def get_trajectory(self, t_span=[0,3], timescale=10, radius=None, y0=None, noise_std=0.01, **kwargs):
        t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])))

        if y0 is None:
            y0 = np.squeeze(self.random_init(), axis = 0)

        spring_ivp = solve_ivp(fun=self.dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10, **kwargs)
        q, p = spring_ivp['y'][0], spring_ivp['y'][1]
        dydt = [self.dynamics_fn(None, y) for y in spring_ivp['y'].T]
        dydt = np.stack(dydt).T
        dqdt, dpdt = np.split(dydt,2)
        
        # add noise
        q += np.random.randn(*q.shape)*noise_std
        p += np.random.randn(*p.shape)*noise_std

        dqdt += np.random.randn(*dqdt.shape)*noise_std
        dpdt += np.random.randn(*dpdt.shape)*noise_std
        return q, p, dqdt, dpdt, t_eval
    
    def get_data(self, sample=50, test_split=0.5, dim = 2, noise=0.01):

        xs, dxs, ts = [], [], []
        for s in range(sample):
            x, y, dx, dy, t = self.get_trajectory(noise_std=noise)
            xs.append( np.stack( [x, y]).T )
            dxs.append( np.stack( [dx, dy]).T )
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

    
    def gx(x):
        return torch.sum(x**2, dim = -1)





def test():
    for i in range(100):
        test_dynamics = dynamics()
        x, y, dx, dy, t = test_dynamics.get_trajectory(sample=1)
        for i in range(x.shape[1]):
            jaco = test_dynamics.get_jacobian(np.stack([x[i], y[i]]).T)
        # print(jaco)
            # jaco=np.asarray([[0,1],[-10,-0.3]])
            evalue, evect = np.linalg.eig(jaco)
            
            evect_inv = np.linalg.inv(evect)
            jaco-evect@np.diag(evalue)@evect_inv

            if max(np.real(evalue))>0:
                print(evalue)
        # print("*********")
        # print(jaco)
if __name__ == "__main__":
    test()
