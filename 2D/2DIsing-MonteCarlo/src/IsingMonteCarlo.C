#include <iostream>
#include <fstream>
#include <vector>
#include <cstddef>
#include <map>
#include <cmath>
#include <algorithm>

#include <Ran2.h>
#include <R1279.h>

/// define the random number generator - use compiler -D flag
//  to define

#ifdef RAN2
typedef psimag::Ran2          UniformRandom;
#endif

#ifdef R1279
typedef psimag::R1279         UniformRandom;
#endif

// -----------------------------------------------
class Lattice {
  public:
    Lattice(std::size_t len) : lattice(len*len,1), L(len) { }

    // index operator
    int operator[](std::size_t index) const {
      return lattice[index];
    }

    // returns sum of neighboring spins at site x,y
    int sum_neighbour(std::size_t index) const {

      std::size_t x,y;

      x = index/L;
      y = index%L;

      std::size_t left_x,right_x,top_x,bottom_x;
      std::size_t left_y,right_y,top_y,bottom_y;

      left_x  = x-1;
      right_x = x+1;
      top_x   = x;
      bottom_x= x;
      left_y  = y;
      right_y = y;
      top_y   = y+1;
      bottom_y= y-1;

      // use peroidic boundary condition
      if     (x==0)   { left_x   = L-1; }
      else if(x==L-1) { right_x  = 0;   }
      if     (y==0)   { bottom_y = L-1; }
      else if(y==L-1) { top_y    = 0;   }

      return (*this)(left_x,left_y) + 
             (*this)(right_x,right_y) +
             (*this)(top_x,top_y) +
             (*this)(bottom_x,bottom_y);
    }

    // flip the spin at index - the only non-constant
    // function of this class
    void flip(std::size_t index) { 
      lattice[index] = -lattice[index]; 
    }

    std::size_t size() const { return lattice.size(); }

    int    magnetization() const {
      int m=0;
      for(std::size_t i=0;i<size();i++) {
	m+=lattice[i];
      }
      return m;
    }

    double energy(double J, double h) const { 
      double exch = 0., zeeman = 0.;
      for(std::size_t i=0;i<size();i++) {
	exch += -lattice[i]*J*sum_neighbour(i);
	zeeman += -h*lattice[i];
      }
      if(0.5*exch<-2.*J*size() || 0.5*exch>2.*J*size()) {
        std::cerr<<__FILE__<<__LINE__<<" wrong energy "
                 <<0.5*exch<<std::endl;
	exit(1);
      }
      return (0.5*exch + zeeman)/size();
    }

    // print as vectorize config
    void print_config(std::ofstream& out) {

      for(unsigned int i=0;i<lattice.size()-1;++i) {
	out<<lattice[i]<<",";
      }
      out<<lattice[lattice.size()-1]<<std::endl;
    }
    void print_energy(std::ofstream& oute) { oute<<energy(1,0)<<std::endl; }

  private:
    std::vector<int>  lattice;
    const std::size_t L;

    // index operator, follow C convention
    int operator()(std::size_t x, std::size_t y) const {
      return lattice[x*L+y];
    }
};

// ----------------------------------------------
class IsingFlip {
  public:

    template<class RNG>
    std::size_t operator()(RNG& ran,Lattice& lat,
                           double T,double h,double J) {

      // choose a spin
      std::size_t chosen_spin =
	   static_cast<std::size_t>(lat.size()*ran());

      // calculate change in energy
      double de = -2.0*lat[chosen_spin]*
                  (-1.0*J*lat.sum_neighbour(chosen_spin)-h);

      // flip if accepted
      if(ran()<std::exp(-de/T)) {
	lat.flip(chosen_spin);
	return 1;
      }
      // reject
      return 0;
    }
};

// ----------------------------------------------
// a class to do data analysis, store measured values
// in a histogram and then provide methods to find
// moments : mean, err etc
template<class ValueType>
class DataAnalysis {
  public:

    /// insert an element
    void insert(const ValueType& value) { 
      if(histogram.find(value)==histogram.end()) {
	histogram[value]=1;
      }
      else { histogram[value]++; }
    }

    template<std::size_t M>
    double mean() const {
      typename Histogram::const_iterator hist_itr;
      double ret=0.;
      for(hist_itr=histogram.begin();
	  hist_itr!=histogram.end();++hist_itr) {
	ret+=power<M>(hist_itr->first)*hist_itr->second;
      }
      return ret/count();
    }

    double err() const {
      double factor = 1./sqrt(count()-1);
      if(count()<=1) factor = 1;
      return sqrt(mean<2>() - mean<1>()*mean<1>())*factor;
    }

    std::size_t count() const {
      typename Histogram::const_iterator hist_itr;
      std::size_t cnt=0;
      for(hist_itr=histogram.begin();
	  hist_itr!=histogram.end();++hist_itr) {
	cnt += hist_itr->second;
      }
      return cnt;
    }

    void clear() { histogram.clear(); }

  private:

    template<std::size_t M>
    double power(const ValueType& value) const {
      double ret=1.;
      for(std::size_t i=0;i<M;i++) {
	ret *= value;
      }
      return ret;
    }
    
    typedef std::map<ValueType,std::size_t>  Histogram;
    
    Histogram                          histogram;

};

// ----------------------------------------------

int main ( int argc, char * argv[]) {

  // set for ferromagnetic Ising model
  const double J = 1.0;

  if(argc!=5) {
    std::cerr<<"usage <program> <L> ";
    std::cerr<<"<Equilibration> <MC-steps> ";
    std::cerr<<"<start T> "
             <<std::endl;
    exit(1);
  }

  std::size_t L = atoi(argv[1]);
  std::size_t equilibration = atoi(argv[2]);
  std::size_t MCsteps = atoi(argv[3]);

  double      startT = atof(argv[4]);
  double      h      = 0;// atof(argv[8]);

  // print input for documentation
  std::cout<<"#L "<<L<<std::endl;
  std::cout<<"#equilibration "<<equilibration<<std::endl;
  std::cout<<"#MCSteps "<<MCsteps<<std::endl;
  std::cout<<"#startT "<<startT<<std::endl;
  std::cout<<"#h "<<h<<std::endl;

  IsingFlip                  metropolis;
  Lattice                    lattice(L);

  DataAnalysis<int>          dataM,dataMsq;
  DataAnalysis<double>       statM,statMsq;
  DataAnalysis<double>       dataE,dataEsq;
  DataAnalysis<double>       statE,statEsq;
  DataAnalysis<std::size_t>  diagnose;

  /// random number generator, typedef at top of file
  UniformRandom               ran;

  std::ofstream outc1("trainx1.dat");
  std::ofstream outc2("trainx2.dat");
  std::ofstream oute1("trainy1.dat");
  std::ofstream oute2("trainy2.dat");

  double T=startT;

  std::cout<<"#T , M, Merr , Msq , Msq err , E \
              , Eerr , Esq , Esq err , accept ratio , ratio err "
           <<std::endl;

//  do {

      for(std::size_t i=0;i<equilibration*lattice.size();i++) {
	metropolis(ran,lattice,T,h,J);
      }
      for(std::size_t i=0;i<MCsteps;i++) {
        for(std::size_t j=0;j<100*lattice.size();j++) {
	  diagnose.insert(metropolis(ran,lattice,T,h,J));
	}
	// print lattice configuration
	lattice.print_config(outc1);
	lattice.print_energy(oute1);
      }
      for(std::size_t i=0;i<MCsteps;i++) {
        for(std::size_t j=0;j<100*lattice.size();j++) {
	  diagnose.insert(metropolis(ran,lattice,T,h,J));
	}
	// print lattice configuration
	lattice.print_config(outc2);
	lattice.print_energy(oute2);
      }


//    T+=dT;
//  } while(T<endT);

}

