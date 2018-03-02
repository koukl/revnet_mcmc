from revnet_trial import revnet_trial
from ising2D      import ising2D

import numpy as np
import tensorflow as tf

# set random seed
tf.set_random_seed(1)
np.random.seed(0)

# -------- global constant variables ----------
L                    = 8
Ndata		     = 5
revnet_training_time = 10000
ising                = ising2D(L)

# ------ read in data from csv files -----------
trainx1 = np.loadtxt('2DIsing-MonteCarlo/bin/trainx1.dat',delimiter=',')
trainx2 = np.loadtxt('2DIsing-MonteCarlo/bin/trainx2.dat',delimiter=',')
traine1 = np.loadtxt('2DIsing-MonteCarlo/bin/trainy1.dat')
traine2 = np.loadtxt('2DIsing-MonteCarlo/bin/trainy2.dat')
# reshape trainy to (Ndata, 1)
traine1 = traine1.reshape((traine1.shape[0], 1))
traine2 = traine2.reshape((traine2.shape[0], 1))
# crop to Ndata
trainx1 = trainx1[:Ndata]
trainx2 = trainx2[:Ndata]
traine1 = traine1[:Ndata]
traine2 = traine2[:Ndata]

revnet_mc = revnet_trial(L,ising)
revnet_mc.train(revnet_training_time,trainx1,trainx2,traine1,traine2)

