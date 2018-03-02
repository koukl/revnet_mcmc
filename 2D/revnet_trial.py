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
    lam                = -.5

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
    loss_op, ey1, ey2  = self.cost.get(e1,e2,y1,y2)

    decorrelate1_op = tf.divide(tf.reduce_mean(tf.multiply(x1,y1)),self.n_spins)
    decorrelate2_op = tf.divide(tf.reduce_mean(tf.multiply(x2,y2)),self.n_spins)

    de1_op          = tf.subtract(e1,self.ising.energy(y1_2d))
    de2_op          = tf.subtract(e2,self.ising.energy(y2_2d))
    demean_op       = tf.reduce_mean(tf.square(tf.add(de1_op,de2_op)))

    # ------------- define the optimizer ------------
    optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    train_op  = optimizer.minimize(loss_op)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

      sess.run(init)
      print(' ---------------- evaluate loss_op ----------------')
      sess.run(loss_op,feed_dict={x1:trainx1,x2:trainx2,e1:traine1,e2:traine2})
      print('------------- check energy calculation ------------')
      y1_np, y2_np, ey1_, ey2_ = sess.run([y1,y2,ey1,ey2], feed_dict={x1:trainx1,x2:trainx2,e1:traine1,e2:traine2})
      y1_np = y1_np.reshape((-1, self.L, self.L))
      y2_np = y2_np.reshape((-1, self.L, self.L))
      ey1_np = self.cost.np_energy(y1_np)
      ey2_np = self.cost.np_energy(y2_np)
      print("np ey1_: ", ey1_np)
      print("tf ey1: ", ey1_)
      print("np ey2_: ", ey2_np)
      print("tf ey2: ", ey2_)
      quit()
   #----------------------------------------------------------    
      for i in range(niter):
        t = sess.run(train_op, feed_dict={x1:trainx1,x2:trainx2,e1:traine1,e2:traine2})
        if(i%100==0):
          loss = sess.run(loss_op,feed_dict={x1:trainx1,x2:trainx2,e1:traine1,e2:traine2})
          print('i ',i, ' loss ',loss)
	  deco1 = sess.run(decorrelate1_op,feed_dict={x1:trainx1,x2:trainx2})
	  deco2 = sess.run(decorrelate2_op,feed_dict={x1:trainx1,x2:trainx2})
          print('decorrelation ',deco1,' : ',deco2)
	  de = sess.run(demean_op,feed_dict={x1:trainx1,x2:trainx2,e1:traine1,e2:traine2})
          print('de mean ',de )


# ==================================================================
  def generate(self,testx1,testx2):
    return(0) # do nothing for now
