import tensorflow as tf

# ------------- define the revnet ---------------------
class cost:

# ======================================================
  def __init__(self,phys_model,lam):
    self.phys_model = phys_model
    self.lam        = lam

# ======================================================
  def get(self,e1,e2,y1,y2): # feed in [ndata,L*L} & [ndata,1]

# check if e1,e2,y1,y2 are batch
    e_shape = e1.shape
    print('e_shape ',e_shape)
    y_shape = y1.shape
    print('y_shape ',y_shape)

    L = self.phys_model.getL()
# y1_2d = tf.reshape(y1,[-1,L,L])
    y1_2d = tf.reshape(y1,[-1,4,16])
#    y2_2d = tf.reshape(y2,[-1,L,L])
    y2_2d = tf.reshape(y2,[-1,4,16])

    y2_shape = y1_2d.shape
    print('y2d_shape ',y2_shape)
# error these lines
    ey1 = self.phys_model.energy(y1_2d) # y1_2d is of shape [ndata,L,L], pass into 
    ey2 = self.phys_model.energy(y2_2d) # pass into ising2d gives shape imcompatible

    ex = tf.add(e1,e2)
    ey = tf.add(ey1,ey2)

    norm1 = tf.reduce_sum(tf.multiply(y1,y1))
    norm2 = tf.reduce_sum(tf.multiply(y2,y2)) 

    fedi     = tf.square(ex-ey)
    tflambda = tf.constant(self.lam)
    tfnspins = tf.reduce_prod(y1)
    norm     = tf.multiply(tflambda,tf.add(norm1,norm2))
    norm     = tf.divide(norm,tfnspins)

    loss = tf.add(fedi,norm)
    return(loss)


