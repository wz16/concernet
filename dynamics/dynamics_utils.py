import numpy as np
class base_dynamics():
    def __init__(self):
        pass
    def dynamics_fn(self, x, t):
        raise NotImplementedError
    def get_jacobian(self, x):
        n = self.dynamics_dim
        dx = 1e-7
        fx = self.dynamics_fn(None, x)
        jaco = np.zeros((n,n))
        for i in range(n):
            x_ = np.copy(x)
            x_[i] += dx
            fx_ = self.dynamics_fn(None, x_)
            jaco[:,i] = (fx_-fx)
        jaco = jaco/dx
        return jaco

    def get_data(self, samples = 300):
        raise NotImplementedError
    