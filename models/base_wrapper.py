import torch
import torch.nn as nn
import importlib
import torch.nn.functional as F


def gram_schmidt(vv):
    def projection(u, v):
        return (v * u).sum() / (u * u).sum() * u

    nk = vv.size(1)
    uu = torch.zeros_like(vv, device=vv.device)
    uu[:, 0,:] = vv[:, 0,:] .clone()
    for k in range(1, nk):
        vk = vv[:,k,:].clone()
        uk = 0
        for j in range(0, k):
            uj = uu[:,j,:].clone()
            uk = uk + projection(uj, vk)
        uu[:,k,:] = torch.nn.functional.normalize(vk - uk, dim = -1)
    # for k in range(nk):
    #     uk = uu[..., k].clone()
    #     uu[..., k] = torch.nn.functional.normalize(uk, dim = -1)
    return uu

class wrapper(torch.nn.Module):

  def __init__(self, dynamics_model, wrapper_mode = 'default', **kwargs):
    super(wrapper, self).__init__()
    self.wrapper_mode = wrapper_mode
    self.dynamics_model = dynamics_model
    if 'knowngx'in wrapper_mode:
      self.gx = kwargs['gx']
    self.ae_mode = False
  def forward(self, x):

    if self.wrapper_mode == 'default':
      fx_hat = self.dynamics_model(x)
      
    elif self.wrapper_mode == "project_with_knowngx":
      x = torch.tensor(x, requires_grad=True, dtype=torch.float32)
      def get_grad(x):
          gx = self.gx(x)
          dgdxs = []
          for i in range(gx.shape[-1]):
            dgdxs.append(torch.nn.functional.normalize(torch.autograd.grad(gx[...,i].sum(), x, create_graph=True)[0], dim = -1))
          return torch.autograd.grad(gx[:,0].sum(), x, create_graph=True)[0], dgdxs
      h = self.dynamics_model(x)
      dgdx, dgdxs  = get_grad(x)
      dgdxs = torch.stack(dgdxs,dim=1)
      projection_ = torch.squeeze(torch.unsqueeze(h, dim = -2) @ torch.unsqueeze(dgdx, dim = -1)@torch.unsqueeze(dgdx, dim = -2),dim = -2)
      norm_ = torch.unsqueeze(torch.norm(dgdx, dim = -1),dim = -1)**2
      projection = torch.div(projection_,norm_)
      fx_hat_ = h -  projection
      dgdxs_ortho = gram_schmidt(dgdxs)
      fx_hat = h
      for i in range(dgdxs_ortho.shape[1]):
        dgdx_ortho = dgdxs_ortho[:,i,:].clone()
        fx_hat = fx_hat - torch.squeeze(torch.unsqueeze(h, dim = -2) @ torch.unsqueeze(dgdx_ortho, dim = -1)@torch.unsqueeze(dgdx_ortho, dim = -2),dim = -2)

    return fx_hat

  def add_ae(self, ae_model):
    self.ae_model = ae_model
    self.ae_mode = True
    

