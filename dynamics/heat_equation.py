
# code reused from https://stackoverflow.com/questions/26393545/python-graphing-the-1d-wave-equation-beginner

# code reused from https://github.com/greydanus/hamiltonian-nn
# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import torch
import autograd
import autograd.numpy as np

import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation


dx=0.1 #space increment
dt=0.05 #time increment
tmin=0.0 #initial time
tmax=2.0 #simulate until
xmin=-5.0 #left bound
xmax=5.0 #right bound...assume packet never reaches boundary
c=1.0 #speed of sound

#set initial pulse shape
def init_fn(x):
    val = np.exp(-(x**2)/0.25)
    if val<.001:
        return 0.0
    else:
        return val

class dynamics():
    
    def __init__(self,**kwargs):
        super().__init__()
        self.dynamics_dim = int((xmax-xmin)/dx+1)
        self.nodes = int((xmax-xmin)/dx+1)
        self.x = np.linspace(xmin, xmax, self.nodes)

        return None
    def dynamics_fn(self, t, coords):
        S = np.zeros_like(coords)
        S_padded = np.hstack([coords[0], coords, coords[-1]])
        S = (S_padded[2:]-2*S_padded[1:-1]+S_padded[0:-2])*c/dx/dx

        return S

    def conservation_fn(self, coords):
        H = np.sum(coords, axis = -1, keepdims=True)
        return H

    def random_init(self, batch_dim = 1):
        y0 = np.zeros((batch_dim,self.nodes))
        center = np.random.uniform(low=-1, high=1)
        scale = np.random.uniform(low=0.5, high=2.0)*0.25

        y0[...,0:self.nodes] = np.exp(-((self.x-center)**2)/scale)
        return y0
    
    def get_trajectory(self, t_span=[0,10], timescale=10, radius=None, y0=None, noise_std=0.1, **kwargs):
        t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])))

        if y0 is None:
            y0 = np.squeeze(self.random_init(), axis = 0)

        spring_ivp = solve_ivp(fun=self.dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10, **kwargs)
        y = spring_ivp['y']
        dydt = [self.dynamics_fn(None, y) for y in spring_ivp['y'].T]
        y = y.T
        dydt = np.stack(dydt)
        y += np.random.randn(*y.shape)*noise_std
        dydt += np.random.randn(*dydt.shape)*noise_std
        
        return y, dydt, t_eval
    
    def get_data(self, sample=50, test_split=0.5, dim = 3, noise=0.01):


        xs, dxs = [], []
        for s in range(sample):
            x, dx, t = self.get_trajectory(noise_std=noise)
            xs.append( np.stack(x))
            dxs.append( np.stack(dx))
        data = {}
        if dim == 2:
            data['x'] = np.concatenate(xs)
            data['fx'] = np.concatenate(dxs).squeeze()
        elif dim == 3:
            data['x'] = np.stack(xs)
            data['fx'] = np.stack(dxs).squeeze()

        split_ix = int(len(data['x']) * test_split)
        split_data = {}
        for k in ['x', 'fx']:
            split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
        data = split_data
        return data


def draw_wave(x):
    S_padded = x
    S_padded = np.insert(S_padded, 0, 0, axis=1)
    conservation = np.sum(x, axis = -1)
    fig = plt.figure()
    plts = []             # get ready to populate this list the Line artists to be plotted
    # plt.hold("off")
    for i in range(x.shape[0]):
        p, = plt.plot(x[i], 'k')   # this is how you'd plot a single line...
        t = plt.text(0.5, 1.01, "conservation:{}".format(conservation[i]))

        plts.append( [p,t] )           # ... but save the line artist for the animation
    ani = animation.ArtistAnimation(fig, plts, interval=50, repeat_delay=3000)   # run the animation
    ani.save('figs/heat.gif')    # optionally save it to a file


def test():
    for i in range(1):
        test_dynamics = dynamics()
        data = test_dynamics.get_data(sample=2)
        draw_wave(data['x'][0])
if __name__ == "__main__":
    test()
