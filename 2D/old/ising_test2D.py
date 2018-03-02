from ising import ising

import numpy as np

L = 3

ising_model = ising(L)

s = np.random.rand(L,L)
si = np.where(s>0.5,1,-1)
print(si)

# now calculate the energy

e = ising_model.energy(si)

print('e',e)

n = ising_model.norm(si)

print('n',n)


