import tensorflow as tf
import numpy      as np

class ising2D:

# =======================================================
  def __init__(self,L):
    f = -1
    self.nspins = L*L
    self.L  = L
    self.a  = 0
    self.su = f*np.eye(L, k=1)
    # add the peroidic condition
    self.su[L-1,0] = f 
    self.sl = f*np.eye(L, k=-1)
    # add the peroidic condition
    self.sl[0,L-1] = f
    print('su')
    print(self.su)
    print('sl')
    print(self.sl)
    self.tfsu = tf.constant(self.su,dtype=tf.float32)
    self.tfsl = tf.constant(self.sl,dtype=tf.float32)

# =======================================================
  def getL(self):
    return(self.L)

# =======================================================
  def energy(self,x):
# tf version
    xl = tf.tensordot(x,self.tfsl,axes=[[2],[0]]) # assume sum_k x_{ijk} tfsl_{kl} gives xl_{ijl}
    xu = tf.transpose(tf.tensordot(x,self.tfsu,axes=[[1],[1]]),perm=[0,2,1])
    xxl = tf.multiply(x,xl)
    xxu = tf.multiply(x,xu)

    # prints the shapes
    print('initial x shape ',x.shape)
    print('xl shape ',xl.shape)
    print('xu shape ',xu.shape)
    print('xxl shape ',xxl.shape)
    print('xxu shape ',xxu.shape)
    e = tf.reduce_sum(xxl,reduction_indices=[1,2]) \
      + tf.reduce_sum(xxu,reduction_indices=[1,2])
    print('e shape ',e.shape)
#
# need to reshape e into [ndata,1]
    es = tf.reshape(e,[-1,1])

# check energy range
#    maxe = tf.reduce_max(es)
#    mine = tf.reduce_min(es)
#    maxpossible = tf.constant( 2*self.nspins)
#    minpossible = tf.constant(-2*self.nspins)
#
#    tf.cond(tf.less(maxpossible,maxe),errfunc)
#    tf.cond(tf.less(mine,minpossible),errfunc)

    return(es)
# =======================================================
  def energy_forloop(self, x):
    # checking energy calculation with for loops
    E = np.zeros((x.shape[0],1))
    for b in range(x.shape[0]):
      for r in range(self.L):
        for c in range(self.L):
          # vertical links
          E[b] -= x[b, r, c] * x[b, (r+1) % self.L, c]
          # horizontal links
          E[b] -= x[b, r, c] * x[b, r, (c+1) % self.L]
      #E[b] /= self.nspins
    return E
# =======================================================
  def norm(self,x):
# np version
#    return np.sum(np.multiply(x,x))/self.nspins
    return(0)

# =======================================================
  def errfunc(self):
    print('energy calculation error')
    quit()
