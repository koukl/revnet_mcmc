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

# numpy version
#    xl = np.matmul(x,self.sl)
#    xu = np.matmul(self.su,x)
#    print('xu')
#    print(xu)
#    print('xl')
#    print(xl)
#    return (np.sum(np.multiply(x,xl)) + np.sum(np.multiply(x,xu)))/self.nspins

# tf version
#    xl = tf.matmul(x,self.tfsl)
#    xu = tf.matmul(self.tfsu,x)
    x_shape = x.shape
    print('energy xshape ',x_shape)
    print('sl shape ',self.tfsl)
#    quit()
    tfsl = tf.constant(np.eye(16,4))
    xl = tf.tensordot(x,tfsl,axes=[[2],[0]]) # assume sum_k x_{ijk} tfsl_{kl} gives xl_{ijl}

#    xl = tf.tensordot(x,self.tfsl,axes=[[2],[0]]) # assume sum_k x_{ijk} tfsl_{kl} gives xl_{ijl}
    print('xl shape ',xl.shape)

#    xu = tf.transpose(tf.tensordot(self.tfsu,x,axes=[[1],[1]]),perm=[1,0,2])
#    print('xl shape ',xl.shape)
#    print('xu shape ',xu.shape)
    quit()
    return tf.reduce_sum(tf.multiply(x,xl),reduction_axes=???) +
tf.reduce_sum(tf.multiply(x,xu),reduction_axes=???)


# =======================================================
  def norm(self,x):
# np version
#    return np.sum(np.multiply(x,x))/self.nspins
    return(0)

