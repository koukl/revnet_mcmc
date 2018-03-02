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

    norm1 = tf.reduce_mean(tf.multiply(y1,y1),reduction_indices=[1])
    norm2 = tf.reduce_mean(tf.multiply(y2,y2),reduction_indices=[1]) 

    fedi     = tf.square(ex-ey)

    norm     = tf.add(norm1,norm2)
    norm     = tf.reshape(norm,[-1,1])

    loss = tf.reduce_mean(tf.add(fedi,tf.scalar_mul(0.5*self.lam,norm)))
    fedi_mean = tf.reduce_mean(fedi)
    norm_mean = tf.reduce_mean(norm)

    return(loss, fedi_mean, norm_mean, ey1, ey2)

# ======================================================
  def np_energy(self,y1):
    ey1 = self.phys_model.energy_forloop(y1)
    return ey1

