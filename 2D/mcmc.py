from revnet_trial      import revnet_trial
from single_spin_trial import single_spin_trial


class mcmc:

# =============================================
  def __init__(self,L,ising):
    self.ising        = ising
    self.single_spin  = single_spin_trial(L,L)
    self.revnet_trial = revnet_trial(L,ising) 

# =============================================
  def train(self,niter,trainx1,trainx2,traine1,traine2):
    self.revnet_trial.train(niter,trainx1,trainx2,traine1,traine2)

# =============================================
  def step(self,x1,x2,ex1,ex2,niter,trial_ratio,beta):
    
    ratio = np.rand(niter)
    accpe = np.rand(niter)

    for i in range(niter):

      if(ratio[i]<trial_raito):
        y1,y2 = self.revnet_trial.generate(x1,x2)
        # check for acceptance
        ey1 = ising.energy(y1)
        ey2 = ising.energy(y2)
        de = (ey1+ey2)-(ex1+ex2)
        p = np.exp(-1.*beta*de)
        if(accpe[i]<p):
	  x1 = y1
	  x2 = y2
	  ex1 = ey1
	  ex2 = ey2
      else:
        site1,site2,de1,de2, = self.single_spin_trial.generate(x1,x2)	
        de = de1+de2
        p = np.exp(-1.*beta*de)
        if(accpe[i]<p):
	  x1[site1] = -1*x1[site1]
	  x2[site2] = -1*x2[site2]
	  ex1 = ex1 + de1
	  ex2 = ex2 + de2

    return(x1,x2,ex1,ex2)


