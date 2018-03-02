import numpy as np
import tensorflow as tf

Ndata = 3
L = 2
X = np.arange(Ndata*L*L).reshape((Ndata, L*L))
print("X", X)

Y = np.arange(Ndata*L*L).reshape((Ndata, L, L))
# make first element different
Y[0,0,0] = 10
print("Y", Y)

Xtf = tf.placeholder(tf.float32, name="X", shape=[None, L*L])
Ytf = tf.placeholder(tf.float32, name="Y", shape=[None, L, L])
# fold into 2D
X2 = tf.reshape(Xtf, [-1, L, L])
loss = tf.reduce_sum(tf.square(tf.subtract(X2, Ytf)))

with tf.Session() as sess:
  loss_ = sess.run(loss, feed_dict={Xtf: X, Ytf:Y})
  print("loss: ", loss_)
