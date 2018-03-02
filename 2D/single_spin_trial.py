import numpy as np


class single_spin_trial:

  def __init__(self,Lx,Ly):
    self.randsize = 100000
    self.n_spins  = Lx*Ly
    self.Lx       = Lx
    self.Ly       = Ly
    self.regenerate_rand()

  def regenerate_rand(self):
    self.ranlist = np.random.randint(self.n_spins,size=self.randsize)
    self.rancntr = 0

# make local change and calculate local energy change
  def generate(x1,x2):

    # choose the sites
    if(self.rancntr>=self.randsize): regenerate_rand()

    site1 = self.ranlist[self.rancntr  ]
    site2 = self.ranlist[self.rancntr+1]
    self.rancntr += 2

    x1,y1 = index2coord(site1)
    x2,y2 = index2coord(site1)

    # get the neighborhood indices
    nei1t = get_nei(x1,y1, 0, 1)
    nei1b = get_nei(x1,y1, 0,-1)
    nei1l = get_nei(x1,y1,-1, 0)
    nei1r = get_nei(x1,y1, 1, 0)
    nei2t = get_nei(x2,y2, 0, 1)
    nei2b = get_nei(x2,y2, 0,-1)
    nei2l = get_nei(x2,y2,-1, 0)
    nei2r = get_nei(x2,y2, 1, 0)

    # calculate energy difference
    de1 =  2*x1[site1]*(x1[nei1t]+x1[nei1b]+x1[nei1l]+x1[nei1r])
    de2 =  2*x2[site2]*(x2[nei2t]+x2[nei2b]+x2[nei2l]+x2[nei2r])

    return(site1,site2,de1,de2)

# Connie:
# check that this is consistent with the reshape of numpy
  def index2coord(index): 
    xcoord = index//self.Ly
    ycoord = index%self.Ly
    return(xcoord,ycoord)

# Connie:
# check that this is consistent with the reshape of numpy
  def coord2index(xcoord,ycoord): 
    index = xcoord*self.Ly+ycoord
    return(index)


  def getnei(x,y,dx,dy):

    xp = x+dx
    yp = y+dy

    if(xp< 0      ): xp = self.Lx-1
    if(xp>=self.Lx): xp = 0
    if(yp< 0      ): yp = self.Ly-1
    if(yp>=self.Ly): yp = 0

    index = coord2index(xp,yp)
    return(index)


