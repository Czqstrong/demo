import numpy as np
import matplotlib.pyplot as plt
import time

class LorenzSystem(object):
  def __init__(self,sigma=np.sqrt(2*20)*np.identity(3),xrange=[-25,25],yrange=[-30,30],zrange=[-10,60]):
    self.dim = 3
    self.sigma = sigma
    self.D = self.sigma@np.transpose(self.sigma)
    self.eps = np.linalg.norm(self.D,2)/2
    self.Dbar = self.D/self.eps/2
    self.D_list = [self.D]
    self.xrange = xrange
    self.yrange = yrange
    self.zrange = zrange
  def get_f(self,X,sigma=10,rho=28,beta=8./3):
    if np.size(X.shape)==1:
      x,y,z=X[0],X[1],X[2]
    else:
      x,y,z=X[:,0],X[:,1],X[:,2]
    return np.stack([sigma*(y-x),x*(rho-z)-y,x*y-beta*z],axis=-1)

LS = LorenzSystem()

def get_uniform_data(N):
  data = np.zeros(dtype=np.float64,shape=(N,LS.dim))
  Xrange = [LS.xrange,LS.yrange,LS.zrange]
  for i in range(LS.dim):
    data[:,i] = np.random.uniform(Xrange[i][0],Xrange[i][1],N)
  return data


def sample_equili_data(N,dt=1e-2,T=10,sigma=LS.sigma,m0=0,m=100,k=1):
    X = []
    x = get_uniform_data(N)
    for i in range(np.int_(T/dt*k)):
        if i>=m0*k and i%(m*k)==0:
          X.append(x+0.)
          if i % m0 == 0:
            print(i)
        x = x + LS.get_f(x)*dt/k + np.sqrt(dt/k)*\
                (np.random.normal(0,1,x.shape)@np.transpose(sigma))
        if (i+1)%np.int_(T/dt*k/10)==0:
            print((i+1)/np.int_(T/dt*k/10))
            fig,ax = plt.subplots(1,3,figsize=(15,4))
            ax[0].scatter(x[:,0],x[:,1],s=0.02)
            ax[1].scatter(x[:,0],x[:,2],s=0.02)
            ax[2].scatter(x[:,1],x[:,2],s=0.02)
            plt.show()
        if i==m0*k:
          print('==============================================')
          print(i)
    return np.reshape(X,(-1,LS.dim))

tt = time.time()
equili_data = sample_equili_data(N=int(1e3),dt=1e-3,T=100000,m0=100000,m=1000)
print(time.time()-tt)
np.save('true_points.npy',equili_data)
