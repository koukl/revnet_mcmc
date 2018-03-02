
# make a reshape that follows this convention
# or change the c++ code to follow tensorflow reshape convention
#
#109     // index operator, follow C convention
#110     int operator()(std::size_t x, std::size_t y) const {
#111       return lattice[x*L+y];
#112     }
#
# given index, coordinates are x = index/L, y = index%L

# takes in spins as a 2D tensor
def coord2index(x):

  return(index) # returns spins as a vector

# takes in spins as a vector
def index2coord(i):

  return(coord) # returns spins as a 2D tensor

