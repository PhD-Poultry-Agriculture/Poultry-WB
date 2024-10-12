import numpy as np
from pylab import scatter

def checkboard(N):
   X=np.random.rand(2,N)
   p=np.mod(np.ceil(X*3),2)
   y0=2.*np.logical_xor(p[0,:],p[1,:])-1.
   return X,y0

def test_checkboard():
   x,y=checkboard(2000)
   scatter(x[0,y==-1.],x[1,y==-1.],c='r')
   scatter(x[0,y==1.],x[1,y==1.],c='b')
