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

    xl = tf.tensordot(x,self.tfsl,axes=[[2],[0]]) # assume sum_k x_{ijk} tfsl_{kl} gives xl_{ijl}
    xu = tf.transpose(tf.tensordot(x,self.tfsu,axes=[[1],[1]]),perm=[0,2,1])
    xxl = tf.multiply(x,xl)
    xxu = tf.multiply(x,xu)

    e = tf.reduce_sum(xxl,reduction_indices=[1,2]) \
      + tf.reduce_sum(xxu,reduction_indices=[1,2])

    # normalize to the lattice size
    e = tf.divide(e,self.nspins)

    es = tf.reshape(e,[-1,1])

    # to force error
#    es = es - 1000
    # check energy range
    maxe = tf.reduce_max(es)
    mine = tf.reduce_min(es)
    maxpossible = tf.constant( 2.0*self.nspins)
    minpossible = tf.constant(-2.0*self.nspins)

    with tf.control_dependencies([tf.assert_less(maxe,maxpossible)]):
      es = es * 1
    with tf.control_dependencies([tf.assert_less(minpossible,mine)]):
      es = es * 1
    return(es)
# =======================================================
  def energy_forloop(self, x):

    # checking energy calculation with for loops
    E = np.zeros((x.shape[0],1))
    for b in range(x.shape[0]):
      for r in range(self.L):
        for c in range(self.L):
	  if (x[b,r,c]>1 or x[b,r,c]<-1):
	    print('spin out of range ',x[b,r,c])
	    quit()
          # vertical links
          E[b] -= x[b, r, c] * x[b, (r+1) % self.L, c]
          # horizontal links
          E[b] -= x[b, r, c] * x[b, r, (c+1) % self.L]
      E[b] /= self.nspins
    return E
# =======================================================
