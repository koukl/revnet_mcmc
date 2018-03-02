import tensorflow as tf

# u_net architecture
# input: linear vector of spins
# output: linear vector of spins
#
# assume Lx = Ly
#
# input 
#  -> reshape into lattice  : 1 channel
#  -> Conv2D(filters=2,kernel_size=[2,2],strike=2) : make into 2 channels
#  -> Conv2D(filters=2,kernel_size=[2,2],strike=2) : this will make into 4 channels? or remain 2?
#  -> Conv2D(filters=2,kernel_size=[2,2],strike=2) : this will make into 8 channels? or remain 2?
#   . . . number of convolution layers as parameter in the constructor
#  -> some mlp layers
#  -> Conv2DTranspose( . . . ) : eventually get back Lx by Ly lattice
#   . . . final output to be vectorized into 1D vector

class u_net:

  # ------------- define the MLP ---------------------
  def __init__(self,Lx,Ly,number_of_conv2D): # Lx, Ly are the linear dimensions of the input lattice
    self.number_of_conv2D = number_of_conv2D

  # ------------- define the MLP ---------------------
  def net(self,x): # x is a linear vector to be reshaped into 2D array

    xl = tf.reshape(x,[Lx,Ly]) -- or something like that

    # make u_net here, output y_predi is a linear vector
    y_predi = tf.reshape(output,[1]) -- or something like that
    return(y_predi) 



