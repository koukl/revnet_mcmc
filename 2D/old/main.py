from mlp_net2D import mlp_net
from revnet2D  import revnet
from cost      import cost
from ising2D   import ising

import numpy as np
import pandas as pd
import tensorflow as tf

# -------- global constant variables ----------
L             = 8
n_spins       = L*L
learning_rate = 0.01
niter         = 8000
lam           = -.5

# ------ read in data from csv files -----------
trainx1 = np.loadtxt('2DIsing-MonteCarlo/bin/trainx1.dat',delimiter=',')
trainx2 = np.loadtxt('2DIsing-MonteCarlo/bin/trainx2.dat',delimiter=',')
trainy1 = np.loadtxt('2DIsing-MonteCarlo/bin/trainy1.dat')
trainy2 = np.loadtxt('2DIsing-MonteCarlo/bin/trainy2.dat')
# reshape trainy to (Ndata, 1)
trainy1 = trainy1.reshape((trainy1.shape[0], 1))
trainy2 = trainy2.reshape((trainy1.shape[0], 1))

# ------- tf variables ---------------------------
x1 = tf.placeholder("float", [None, n_spins])
x2 = tf.placeholder("float", [None, n_spins])
e1 = tf.placeholder("float", [None, 1])
e2 = tf.placeholder("float", [None, 1])
# ------------- define the MLP and revnet objects ---
F_net = mlp_net(n_spins)
G_net = mlp_net(n_spins)
r_net = revnet(F_net,G_net)
y1,y2 = r_net.forward(x1,x2)
# ------------- define the cost function ------------
phys_model = ising(L)
c          = cost(phys_model,lam)
loss_op    = c.get(e1,e2,y1,y2)
# ------------- define the optimizer ------------
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op  = optimizer.minimize(loss_op)

init = tf.global_variables_initializer()

# test it

with tf.Session() as sess:

  sess.run(init)
  sess.run(loss_op,feed_dict={x1:trainx1,x2:trainx2,e1:trainy1,e2:trainy2})


#  vy1,vy2 = r_net.forward(x1,x2,F_net,G_net)
#  ry1,ry2 = sess.run([vy1,vy2],feed_dict={x1:trainx1,x2:trainx2})
#
# now get back the x variables
#  vx1,vx2 = r_net.backward(y1,y2,F_net,G_net)
#  rx1,rx2 = sess.run([vx1,vx2],feed_dict={y1:ry1,y2:ry2})
#
#  print('reconstruction errors')
#
#  diff1 = trainx1-rx1
#  diff2 = trainx2-rx2
#  d1 = np.sum(diff1**2)
#  d2 = np.sum(diff2**2)
#  print('d1 d2 ',d1,d2)


#  for i in range(10):
#    t = sess.run(train_op, feed_dict={x1:trainx1,x2:trainx2,e1:trainy1,e2:trainy2})

