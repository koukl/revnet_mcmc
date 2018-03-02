import tensorflow as tf

# ------------- define the revnet ---------------------
class cost:

# ======================================================
  def __init__(self,phys_model,lam):
    self.phys_model = phys_model
    self.lam        = lam

# ======================================================
  def get(self,e1,e2,y1,y2): # feed in [ndata,L*L} & [ndata,1]

    L     = self.phys_model.getL()
    y1_2d = tf.reshape(y1,[-1,L,L])
    y2_2d = tf.reshape(y2,[-1,L,L])

    ey1 = self.phys_model.energy(y1_2d) # y1_2d is of shape [ndata,L,L], pass into 
    ey2 = self.phys_model.energy(y2_2d) # pass into ising2d gives shape imcompatible

    ex = tf.add(e1,e2)
    ey = tf.add(ey1,ey2)

    norm1 = tf.reduce_sum(tf.multiply(y1,y1),reduction_indices=[1])
    norm2 = tf.reduce_sum(tf.multiply(y2,y2),reduction_indices=[1]) 

    fedi     = tf.square(ex-ey)

    tflambda = tf.constant(self.lam)
    tfnspins = tf.constant(L*L,dtype=tf.float32)
    norm     = tf.multiply(tflambda,tf.add(norm1,norm2))
    norm     = tf.divide(norm,tfnspins)
    norm     = tf.reshape(norm,[-1,1])

    loss = tf.reduce_mean(tf.add(fedi,norm))

    print('ex shape ',ex.shape)
    print('ey shape ',ey.shape)
    print('fedi shape ',fedi.shape)
    print('norm shape ',norm.shape)
    print('loss shape ',loss.shape)

    return(loss)


