from mlp_net   import mlp_net
from rev_net   import rev_net
from cost      import cost

import numpy as np
import pandas as pd
import tensorflow as tf

class revnet_trial:

# ==================================================================
  def __init__(self,L,phys_model):
    # -------- global constant variables ----------
    self.L             = L
    self.n_spins       = L*L
    self.learning_rate = 0.01
    lam                = -10.

    # ------------- define the MLP and revnet objects ---
    self.F = mlp_net(self.n_spins)
    self.G = mlp_net(self.n_spins)
    self.R = rev_net(self.F,self.G)
    # ------------- define the cost function ------------
    self.ising = phys_model
    self.cost  = cost(self.ising,lam)

# ==================================================================
  def train(self,niter,trainx1,trainx2,traine1,traine2):

    # ------- tf variables ---------------------------
    x1 = tf.placeholder("float", [None, self.n_spins])
    x2 = tf.placeholder("float", [None, self.n_spins])
    e1 = tf.placeholder("float", [None, 1])
    e2 = tf.placeholder("float", [None, 1])
    y1,y2 = self.R.forward(x1,x2)
    y1_2d = tf.reshape(y1,[-1,self.L,self.L])
    y2_2d = tf.reshape(y2,[-1,self.L,self.L])

    # ------------- define the cost function ------------
    loss_op, fedi_mean_op, norm_mean_op, ey1, ey2  = self.cost.get(e1,e2,y1,y2)

    decorrelate1_op = tf.reduce_mean(tf.multiply(x1,y1))
    decorrelate2_op = tf.reduce_mean(tf.multiply(x2,y2))
    # ------------- define the optimizer ------------
    optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    train_op  = optimizer.minimize(loss_op)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

      sess.run(init)
      print(' ---------------- evaluate loss_op ----------------')
      sess.run(loss_op,feed_dict={x1:trainx1,x2:trainx2,e1:traine1,e2:traine2})

      print('------------- check energy calculation ------------')
      y1_np, y2_np, ey1_, ey2_ = sess.run([y1,y2,ey1,ey2], \
	                                  feed_dict={x1:trainx1,x2:trainx2,e1:traine1,e2:traine2})
      ey1_np = self.cost.np_energy(y1_np)
      ey2_np = self.cost.np_energy(y2_np)
      print('checking energy :: mean diff 1 ', np.mean(np.square(ey1_np-ey1_)))
      if(self.rel_error(ey1_np,ey1_) > 1e-5):
        raise ValueError('Error in ey1 calculation')
      print('checking energy :: mean diff 2 ', np.mean(np.square(ey2_np-ey2_)))
      if(self.rel_error(ey2_np,ey2_) > 1e-5):
        raise ValueError('Error in ey2 calculation')
      #----------------------------------------------------------    

      for i in range(niter):
        t = sess.run(train_op, feed_dict={x1:trainx1,x2:trainx2,e1:traine1,e2:traine2})
        if(i%100==0):
          loss, fedi_mean, norm_mean = \
	           sess.run([loss_op,fedi_mean_op,norm_mean_op], \
		             feed_dict={x1:trainx1,x2:trainx2,e1:traine1,e2:traine2})
          print('------------- check cost calculation ------------')
          y1_np, y2_np  = sess.run([y1,y2,], \
                                   feed_dict={x1:trainx1,x2:trainx2,e1:traine1,e2:traine2})
          loss_np, fedi_mean_np, norm_mean_np = self.cost.np_cost(traine1,traine2,y1_np,y2_np)
          #----------------------------------------------------------    
          print('i ',i, ' loss ',loss)   # min loss = lam
          print('fedi_mean ',fedi_mean)   # expect range of [0, 16*nspins]
          print('norm_mean ',norm_mean)   # expect range of [0, 2]
          if(abs(loss-loss_np) > 1e-5):
	    raise ValueError('Error in loss calculation. loss_tf: ' + str(loss) + ', loss_np: ' + str(loss_np))
          if(abs(fedi_mean-fedi_mean_np) > 1e-5):
	    raise ValueError('Error in fedi_mean calculation. fedi_mean_tf: ' + str(fedi_mean) + ', fedi_mean_np: ' + str(fedi_mean_np))
          if(abs(norm_mean-norm_mean_np) > 1e-5):
	    raise ValueError('Error in norm_mean calculation. norm_mean_tf: ' + str(norm_mean) + ', norm_mean_np: ' + str(norm_mean_np))
	  deco1 = sess.run(decorrelate1_op,feed_dict={x1:trainx1,x2:trainx2})
	  deco2 = sess.run(decorrelate2_op,feed_dict={x1:trainx1,x2:trainx2})
          print('decorrelation ',deco1,' : ',deco2)

# ==================================================================
  def rel_error(self,x,y):
    # returns relative error
    return np.max(np.abs(x-y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
# ==================================================================
  def generate(self,testx1,testx2):
    return(0) # do nothing for now
