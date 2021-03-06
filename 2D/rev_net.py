#   x1 ------\------------+------- y1
#             F          /
#              \        G 
#   x2 --------+-------/---------- y2
#
# forward
#
#  y2 = 0.5*( F(x1) + x2 )
#  y1 = 0.5*( G(y2) + x1 )
#
# inverse
#
#  x1 = 2*y1 - G(y2)
#  x2 = 2*y2 - F(x1)

import tensorflow as tf

# ------------- define the revnet ---------------------
class rev_net:

  def __init__(self,Fnet,Gnet):
    self.F = Fnet
    self.G = Gnet

  def forward(self,x1,x2):

    Fx1 = self.F.net(x1)
    y2  = tf.scalar_mul(0.5,tf.add(x2,Fx1))
    Gy2 = self.G.net(y2)
    y1  = tf.scalar_mul(0.5,tf.add(Gy2,x1))

    return y1,y2 

# ------------- define the inverse revnet -------------

  def backward(self,y1,y2):

    Gy2 = self.G.net(y2)
    x1  = tf.subtract(tf.scalar_mul(2,y1),Gy2)
    Fx1 = self.F.net(x1)
    x2  = tf.subtract(tf.scalar_mul(2,y2),Fx1)
    
    return(x1,x2)
