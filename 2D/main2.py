from mcmc         import mcmc 
from ising2D      import ising2D

import numpy as np
import tensorflow as tf

# -------- global constant variables ----------
L             = 8
training_time = 10000
ising         = ising2D(L)

# ------ read in data from csv files -----------
trainx1 = np.loadtxt('2DIsing-MonteCarlo/bin/trainx1.dat',delimiter=',')
trainx2 = np.loadtxt('2DIsing-MonteCarlo/bin/trainx2.dat',delimiter=',')
traine1 = np.loadtxt('2DIsing-MonteCarlo/bin/traine1.dat')
traine2 = np.loadtxt('2DIsing-MonteCarlo/bin/traine2.dat')
# reshape trainy to (Ndata, 1)
traine1 = traine1.reshape((traine1.shape[0], 1))
traine2 = traine2.reshape((traine2.shape[0], 1))

mc = mcmc(L,ising)
mc.train(training_time,trainx1,trainx2,traine1,traine2)

