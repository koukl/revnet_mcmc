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
#    niter              = 8000
    lam                = -.5

    # ------------- define the MLP and revnet objects ---
    self.F = mlp_net(self.n_spins)
    self.G = mlp_net(self.n_spins)
    self.R = rev_net(self.F,self.G)
    # ------------- define the cost function ------------
    self.ising = phys_model
    self.cost  = cost(self.ising,lam)

# ==================================================================
  def train(self,trainx1,trainx2,traine1,traine2):

    # ------- tf variables ---------------------------
    x1 = tf.placeholder("float", [None, self.n_spins])
    x2 = tf.placeholder("float", [None, self.n_spins])
    e1 = tf.placeholder("float", [None, 1])
    e2 = tf.placeholder("float", [None, 1])
    y1,y2 = self.R.forward(x1,x2)
    # ------------- define the cost function ------------
    print(' ---------------- define loss_op ----------------')
    loss_op    = self.cost.get(e1,e2,y1,y2)
    # ------------- define the optimizer ------------
    optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    train_op  = optimizer.minimize(loss_op)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

      sess.run(init)
      print(' ---------------- evaluate loss_op ----------------')
      sess.run(loss_op,feed_dict={x1:trainx1,x2:trainx2,e1:traine1,e2:traine2})

    #  for i in range(10):
    #    t = sess.run(train_op, feed_dict={x1:trainx1,x2:trainx2,e1:traine1,e2:traine2})

# ==================================================================
  def generate(self,testx1,testx2):
    return(0) # do nothing for now
