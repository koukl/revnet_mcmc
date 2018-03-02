import numpy as np
import tensorflow as tf

Ndata = 2
L = 2
X = np.arange(Ndata*L*L).reshape((Ndata, L*L))
print("X", X)

Y = np.random.rand(Ndata*L*L).reshape((Ndata, L, L))
print("Y", Y)
#Y = np.arange(Ndata*L*L).reshape((Ndata, L, L))
#Y[0,0,0] = 10

Xtf = tf.placeholder(tf.float32, name="X", shape=[None, L*L])
Ytf = tf.placeholder(tf.float32, name="Y", shape=[None, L, L])
W = tf.Variable(np.random.randn(L*L, L*L)*0.01, dtype=tf.float32)
# add one fully connected layer
X1 = tf.matmul(Xtf, W)
# fold into 2D
X2 = tf.reshape(X1, [-1, L, L])
loss = tf.losses.mean_squared_error(labels=X2, predictions=Ytf)
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.1).minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init)
  epoch = 10
  for i in range(epoch):
    _, loss_, output_ = sess.run([optimizer, loss, X2], feed_dict={Xtf: X, Ytf:Y})
    print("loss: ", loss_)
  
  print("final output: ")
  print(output_)
  print("Y: ")
  print(Y)
