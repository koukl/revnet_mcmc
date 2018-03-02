import tensorflow as tf

class mlp_net:


  # ------------- define the MLP ---------------------
  def __init__(self,n_spins):

    f = 10;
    self.w1 = tf.Variable(tf.truncated_normal([  n_spins,f*n_spins],stddev=.05))
    self.w2 = tf.Variable(tf.truncated_normal([f*n_spins,f*n_spins],stddev=.05))
    self.w3 = tf.Variable(tf.truncated_normal([f*n_spins,f*n_spins],stddev=.05))
    self.wo = tf.Variable(tf.truncated_normal([f*n_spins,  n_spins],stddev=.05))
    self.b1  = tf.Variable(tf.constant(0.0,shape=[f*n_spins]))
    self.b2  = tf.Variable(tf.constant(0.0,shape=[f*n_spins]))
    self.b3  = tf.Variable(tf.constant(0.0,shape=[f*n_spins]))
    self.bo  = tf.Variable(tf.constant(0.0,shape=[  n_spins]))

  # ------------- define the MLP ---------------------
  def net(self,x):

    h1 = tf.nn.tanh(tf.nn.bias_add(tf.matmul(x ,self.w1),self.b1),name='h1')
    h2 = tf.nn.tanh(tf.nn.bias_add(tf.matmul(h1,self.w2),self.b2),name='h2')
    h3 = tf.nn.tanh(tf.nn.bias_add(tf.matmul(h2,self.w3),self.b3),name='h2')
    y_predi = tf.nn.tanh(tf.nn.bias_add(tf.matmul(h3,self.wo),self.bo),name='y_predi')
    
    return(y_predi) 



