import os
import argparse
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
from utils import RealNVP
from utils import dataflow
from utils import ATwoDSystem

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

AS = ATwoDSystem()
AS.get_FD_sol(xrange=AS.xrange, yrange=AS.yrange)

N = 200
Lx = 2
Ly = 2
x1 = np.linspace(-Lx, Lx, N)
x2 = np.linspace(-Ly, Ly, N)
X1, X2 = np.meshgrid(x1, x2)
x1 = X1.reshape(-1, 1)
x2 = X2.reshape(-1, 1)
X = np.hstack((x1, x2))
prob_true = AS.get_P(X)
prob_true = prob_true.reshape(N, N)

plt.figure(figsize=(3, 3))
plt.contourf(X1, X2, prob_true)
plt.colorbar()
plt.show()

grid_X = torch.tensor(X, dtype=torch.float32).to(device)
dA = (2 * Lx / N) * (2 * Ly / N)
prob_true_l2norm = np.sqrt(np.sum(prob_true ** 2) * dA)

# ϵ = 0.7
# radius = 3
# λ = 10
# c = 6


def LWS(input_dim, output_dim, scale):
    a = nn.Linear(input_dim, output_dim)
    a.weight.data = a.weight.data * scale
    return a


class FokkerPlanck():
    """
    Solving Fokker-Planck equation using KRnet
    -----------------------------------
    Fokker-Planck equation is a partial differential equation associated with an
    unknown probability density function. Here is an alternative approach to solve
    this PDE. Also, it can give an assessment of the performance of the flow-based model by a
    mathematical model instead of rough computer vision tasks.

    Args:
    -----
        hps: hyper parameters and settings
    """

    def __init__(self, num_sample=10000, num_particle=10000,
                 batchsize=100, dim=32, depth=4, epsilon=0.7, λ=10, c=6,
                 radius=2.5, scale=1.0, prior_std=1.0):
        self.build_model(dim=dim, depth=depth, scale=scale, prior_std=prior_std)
        self.num_sample = num_sample
        self.num_particle = num_particle
        self.batchsize = batchsize
        self.prior_std = prior_std
        self.epsilon = epsilon
        self.λ = λ
        self.c = c
        self.radius = radius

    def build_model(self, dim=64, depth=4, scale=1.0, prior_std=1.0):
        # build the model
        nets = lambda: nn.Sequential(LWS(2, dim, scale), nn.LeakyReLU(),
                                     LWS(dim, dim, scale), nn.LeakyReLU(),
                                     LWS(dim, 2, scale), nn.Tanh())
        nett = lambda: nn.Sequential(LWS(2, dim, scale), nn.LeakyReLU(),
                                     LWS(dim, dim, scale), nn.LeakyReLU(),
                                     LWS(dim, 2, scale))
        masks = torch.from_numpy(np.array([[1, 0], [0, 1]] * depth).astype(np.float32))
        self.pdf_model = RealNVP(nets, nett, masks, prior_std)
        self.pdf_model = self.pdf_model.to(device)

    def get_loss(self, z, ind, use_delta=False, use_wave=False, sigma_param=8):
        x = self.pdf_model.g(z)

        x1 = x[:, 0]
        x2 = x[:, 1]

        f1 = 0.2 * x1 * (1 - x1 ** 2) + x2 * (1 + torch.sin(x1))
        f2 = -x2 + 2 * x1 * (1 - x1 ** 2) * (1 + torch.sin(x1))

        x1_mode = x[ind, 0] + self.epsilon * torch.randn(self.batchsize).to(device)
        x2_mode = x[ind, 1] + self.epsilon * torch.randn(self.batchsize).to(device)

        # loss from boundary condition
        loss_b = self.λ * torch.mean(torch.sigmoid(self.c * (x ** 2 - self.radius ** 2)))

        X1_all = torch.tile(x1, (self.batchsize, 1))
        X2_all = torch.tile(x2, (self.batchsize, 1))

        ref_1 = x1_mode.reshape(-1, 1)
        ref_2 = x2_mode.reshape(-1, 1)

        # use delta functions
        if use_delta:
            phi_basis = torch.exp(-sigma_param * (X1_all - ref_1) ** 2 - \
                                  sigma_param * (X2_all - ref_2) ** 2)

            c1 = 2 * sigma_param
            c2 = c1 ** 2

            phi_basis_x1 = phi_basis * (-c1 * (X1_all - ref_1))
            phi_basis_x2 = phi_basis * (-c1 * (X2_all - ref_2))

            phi_basis_x1x1 = phi_basis * (-c1 + c2 * (X1_all - ref_1) ** 2)
            phi_basis_x2x2 = phi_basis * (-c1 + c2 * (X2_all - ref_2) ** 2)

            laplace = 0.01 * phi_basis_x1x1 + 0.05 * phi_basis_x2x2
            loss_delta = torch.sum(torch.mean(f1 * phi_basis_x1 + f2 * phi_basis_x2 + laplace, 1) ** 2) / self.batchsize

        return loss_b, loss_delta

    def train(self, num_model=1000, learning_rate=1.0e-4,
              sigma_param=8, every_iter_plot=20, plot_iter_sample=99,
              n_test=10 ** 4):
        n_dim = 2
        optim = torch.optim.Adam(self.pdf_model.parameters(), lr=learning_rate)

        use_delta = True
        use_wave = False

        loss_vs_epoch = []
        lossd_vs_epoch = []
        l2_norm_vs_epoch = []

        for idx_model in range(1, num_model + 1):
            z = self.prior_std * np.random.randn(self.num_sample, n_dim)
            z = Variable(torch.from_numpy(z).float(), requires_grad=True).to(device)

            iteration = 0

            ind = np.random.choice(self.num_sample, self.num_particle)
            ind = torch.tensor(ind).to(device)
            data_flow = dataflow(ind)
            train_indset = DataLoader(data_flow, batch_size=self.batchsize, shuffle=True)

            for train_ind in train_indset:
                optim.zero_grad()
                # loss_gen=self.get_loss(z, train_ind, use_delta, use_wave, sigma_param)
                loss_bb, loss_dd = self.get_loss(z, train_ind, use_delta, use_wave, sigma_param)
                loss_gen = loss_bb + loss_dd
                loss_gen.backward(retain_graph=True)
                optim.step()
                iteration += 1

            if idx_model == 1 or (idx_model % every_iter_plot == 0):
                print('model: %s' % (idx_model))
                prob = torch.exp(self.pdf_model.log_prob(grid_X))
                prob = prob.data.cpu().numpy()
                prob = prob.reshape(N, N)
                l2_relative_error = np.sqrt(np.sum((prob_true - prob) ** 2) * dA) / prob_true_l2norm
                print('loss: %s, lossd: %s, lossb: %s, l2rerror: %s ' % (loss_gen.data.cpu().numpy(), \
                                                                         loss_dd.data.cpu().numpy(),
                                                                         loss_bb.data.cpu().numpy(), l2_relative_error))
                l2_norm_vs_epoch.append(l2_relative_error)
                # plt.figure(figsize=(9,5))
                # plt.subplot(2,3,1)
                # plt.contourf(X1,X2,prob_true)
                # plt.colorbar()
                # plt.subplot(2,3,2)
                # plt.contourf(X1,X2,prob)
                # plt.colorbar()
                # plt.subplot(2,3,3)
                # plt.contourf(X1,X2,prob-prob_true)
                # plt.colorbar()
                # plt.subplot(2,3,4)
                # z_plot = self.prior_std * np.random.randn(self.num_sample,n_dim)
                # z_plot = Variable(torch.from_numpy(z_plot).float(),requires_grad=True).to(device)
                # x_plot = self.pdf_model.g(z_plot)
                # x_plot = x_plot.data.cpu().numpy()
                # plt.hist(x_plot[:,0],bins=100,density=True);
                # plt.subplot(2,3,5)
                # plt.hist(x_plot[:,1],bins=100,density=True);
                # plt.subplot(2,3,6)
                # plt.plot(x_plot[:,0],x_plot[:,1],'.')
                # plt.tight_layout()
                # plt.show()

            loss_vs_epoch.append(loss_gen.data.cpu().numpy())
            lossd_vs_epoch.append(loss_dd.data.cpu().numpy())

        # plt.figure(figsize=(9,3))
        # plt.subplot(1,3,1)
        # plt.semilogy(np.arange(1,num_model+1),loss_vs_epoch)
        # plt.subplot(1,3,2)
        # plt.semilogy(lossd_vs_epoch)
        # plt.subplot(1,3,3)
        # plt.semilogy(l2_norm_vs_epoch)
        # plt.tight_layout()
        # plt.show()


import time


def main(args):
    seed = 5
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # gpu
    np.random.seed(seed)
    random.seed(seed)

    model = FokkerPlanck(num_sample=args.num_sample, num_particle=args.num_particle, batchsize=args.batchsize,
                         dim=args.dim, depth=args.depth, epsilon=args.epsilon, λ=args.plambda, c=args.c,
                         radius=args.radius, scale=1.0, prior_std=1)

    sigma = args.sigma  # std for test functions

    t0 = time.time()
    model.train(num_model=args.num_model, learning_rate=args.learning_rate, sigma_param=1 / (2 * sigma ** 2), \
                every_iter_plot=args.every_iter_plot, plot_iter_sample=5)
    t1 = time.time()
    TrainTime = (t1 - t0) / 3600
    print('train time is {:.4} hours'.format(TrainTime))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('--num_sample', type=int, default=10000, help='the number of sample points of z')
    parser.add_argument('--num_particle', type=int, default=1000, help='the number of test functions')
    parser.add_argument('--batchsize', type=int, default=1000, help='the batchsize of the test functions')
    parser.add_argument('--dim', type=int, default=32, help='the width of the neural networks in Real NVP')
    parser.add_argument('--depth', type=int, default=3, help='the length of the flows')
    parser.add_argument('--epsilon', type=float, default=0.5, help='the noise of the test function')
    parser.add_argument('--c', type=float,default=6,help='the hyperparameter for boundary condition')
    parser.add_argument('--radius', type=float, default=2.5, help='the radius of the boundary')
    parser.add_argument('--plambda', type=float, default=10, help='the penalty of the boundary condtion')
    parser.add_argument('--sigma', type=float, default=0.2, help='the variance of the test function')
    parser.add_argument('--num_model', type=int, default=10000, help='iteration')
    parser.add_argument('--learning_rate', type=float, default=1e-04, help='learning rate')
    parser.add_argument('--every_iter_plot', type=int, default=200, help='the number to print the loss')

    args=parser.parse_args()
    main(args)
