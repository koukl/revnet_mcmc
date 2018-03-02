from revnet_trial      import revent_trial
from single_spin_trial import single_spin_trial


class mcmc:

# =============================================
  def __init__(self,L,ising . ... ):
    self.ising        = ising
    self.single_spin  = single_spin_trial( . . .)
    self.revnet_trial = revnet_trial(L,ising) 

# =============================================
  def step(self,x1,x2,ex1,ex2,niter,trial_ratio,beta, .  . .):
    
    ratio = np.rand(niter)
    accpe = np.rand(niter)

    for i in range(niter):

      if(ratio[i]<trial_raito):
        y1,y2 = self.revnet_trial.generate(x1,x2)
      else:
        y1,y2 = self.single_spin_trial.generate(x1,x2)	
 
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

    return(x1,x2,ex1,ex2)


