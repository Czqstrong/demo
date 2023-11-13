import os
import numpy as np
import torch
import random
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import scipy.io as scio
import math
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix
from scipy.interpolate import griddata

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# Define the push forward map
# The RealNVP code is from
# https://github.com/senya-ashukha/real-nvp-pytorch/blob/master/real-nvp-pytorch.ipynb
class RealNVP(nn.Module):
    def __init__(self, nets, nett, mask, prior_std):
        super(RealNVP, self).__init__()
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(self.mask))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(self.mask))])
        self.pi=torch.tensor(np.pi,dtype=torch.float32)
        self.prior_std=torch.tensor(prior_std,dtype=torch.float32)

    def g(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x*self.mask[i]
            s = self.s[i](x_)*(1 - self.mask[i])
            t = self.t[i](x_)*(1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def f(self, x):
        # x=torch.vstack((x1,x2)).T
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1-self.mask[i])
            t = self.t[i](z_) * (1-self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J

    def log_prob(self, x):
        z, logp = self.f(x)
        log_prob = -torch.log(2*self.pi*self.prior_std**2)-torch.sum(z**2/2, dim=1)/(self.prior_std**2)
        return log_prob + logp


class dataflow(Dataset):
    def __init__(self, x_train):
        self.x = x_train

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return self.x.size(0)


def get_prob_flux_sparse(f, eps, D, xrange, yrange, Nx, Ny, px=[0, 0]):
    Lx = xrange[1]-xrange[0]
    Ly = yrange[1]-yrange[0]
    N = Nx*Ny
    hx = Lx/Nx
    hy = Ly/Ny
    D11, D22 = D[0][0], D[1][1]

    def idx(i, j, Nx=Nx, Ny=Ny):
        return (i-1)*Ny+j-1

    def pos(i, j, hx=hx, hy=hy):
        return np.array([xrange[0]+i*hx, yrange[0]+j*hy])
    row = []
    col = []
    data = []

    # construct the system
    for i in range(1, Nx+1):
        for j in range(1, Ny+1):
            # J1(i,j-1/2)/hx
            ff = f(np.vstack([pos(i, j-1/2), pos(i-1, j-1/2), pos(i-1/2, j), pos(i-1/2, j-1)]))
            if i < Nx:
                row += [idx(i, j), idx(i, j)]
                col += [idx(i,  j), idx(i+1, j)]
                data += [(-.5*ff[0][0]-eps*D11/hx)/hx,
                         (-.5*ff[0][0]+eps*D11/hx)/hx]

            # -J1(i-1,j-1/2)/hx
            if i > 1:
                row += [idx(i, j), idx(i, j)]
                col += [idx(i-1, j), idx(i, j)]
                data += [-(-.5*ff[1][0]-eps*D11/hx)/hx,
                         -(-.5*ff[1][0]+eps*D11/hx)/hx]

            # J2(i-1/2,j)/hy
            if j < Ny:
                row += [idx(i, j), idx(i, j)]
                col += [idx(i, j), idx(i, j+1)]
                data += [(-.5*ff[2][1]-eps*D22/hy)/hy,
                         (-.5*ff[2][1]+eps*D22/hy)/hy]

            # -J2(i-1/2,j-1)/hy
            if j > 1:
                row += [idx(i, j), idx(i, j)]
                col += [idx(i, j-1), idx(i, j)]
                data += [-(-.5*ff[3][1]-eps*D22/hy)/hy,
                         -(-.5*ff[3][1]+eps*D22/hy)/hy]
    A = csc_matrix((data, (row, col)))
    print(A.astype)

    # solve the system
    idx_ = idx(np.int_((px[0]-xrange[0])/hx+.5), np.int_((px[1]-yrange[0])/hy+.5))
    ei = np.zeros(dtype=np.float64, shape=(N, 1))
    ei[idx_, 0] = 1
    mask = np.reshape(ei == 1, -1)
    ei = csc_matrix(ei, dtype=np.float64)
    b = -A@ei
    mask = ~mask
    A_ = A[:, mask]
    prob_ = spsolve(A_[:-1], b[:-1])
    prob = np.insert(prob_, idx_, np.array([1]), 0)

    # normalization
    prob = np.maximum(prob, 0)
    Z = prob.mean()*Lx*Ly
    prob = prob/Z
    return (np.linspace(xrange[0], xrange[1], Nx+1)[:-1]+np.linspace(xrange[0], xrange[1], Nx+1)[1:])/2,\
           (np.linspace(yrange[0], yrange[1], Ny+1)[:-1]+np.linspace(yrange[0], yrange[1], Ny+1)[1:])/2,\
           np.transpose(prob.reshape(Nx, Ny))


class ATwoDSystem(object):
    def __init__(self, xrange=[-2, 2], yrange=[-3, 3]):
        self.dim = 2
        self.sigma = np.diag([np.sqrt(1./5), 1])
        self.D = self.sigma@np.transpose(self.sigma)
        self.eps = 0.05
        self.xrange = xrange
        self.yrange = yrange

    def get_f(self, X):
        if np.size(X.shape) == 2:
            x, y = X[:, 0][:, None], X[:, 1][:, None]
        else:
            x, y = X[0], X[1]
        return np.hstack([(.1*2*x*(1-x**2) + (1+np.sin(x))*y),
                          (-y + 2*x*(1+np.sin(x))*(1-x**2))])

    def get_FD_sol(self, xrange, yrange, Nx=400, Ny=400, px=[-1, 0]):
        self.FD_Y = []
        xx, yy, prob = get_prob_flux_sparse(self.get_f, self.eps, self.D, xrange, yrange, Nx, Ny, px)
        self.FD_Y.append(prob.reshape(-1))
        # plot_epslog_prob(prob,self.eps,xx,yy,0,0,Vmax=self.eps*15)
        XX, YY = np.meshgrid(xx, yy)
        c = plt.contourf(XX, YY, prob)
        cbar = plt.colorbar(c, format='%.2f', aspect=50)
        self.FD_X = np.concatenate([XX[:, :, None], YY[:, :, None]], axis=-1).reshape(-1, 2)

    def get_P(self, X):
        return np.maximum(griddata(self.FD_X, self.FD_Y[0], X, method='cubic', fill_value=0), 0)


