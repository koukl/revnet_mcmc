//-*-C++-*-

/**
 * \author Hwee Kuan Lee
 */

#ifndef PSIMAG_Ran2_H
#define PSIMAG_Ran2_H

#include "Real.h"

namespace psimag {

class Ran2 {

    static const int RANA = 32; // define RANA = NTAB in operator ()

  public:

    typedef Real result_type;
    typedef long seed_type;
    static const bool has_fixed_range = true;
    static result_type min() { return 0; }
    static result_type max() { return 1; }

    explicit Ran2(seed_type s=1) : idum(s), idum2(123456789), iy(0) {
      if(s>=0) idum = -s-1;
    }
    
    void seed(seed_type s) 
    { 
      if(s>=0) idum = -s-1; 
      else idum = s; 
    }
    
    result_type operator () ();
    
    result_type operator () (result_type n)
    {
      return (*this)() * n;
    }

    friend bool operator == (const Ran2& a, const Ran2& b)
    { 
      return (a.idum == b.idum) && (a.idum2 == b.idum2) && (a.iy == b.iy); 
    }
    
  private:
    long idum;
    long idum2;
    long iy;
    long iv[RANA];
  };


#define IM1 2147483563
#define IM2 2147483399
#define AM (1.0/IM1)
#define IMM1 (IM1-1)
#define IA1 40014
#define IA2 40692
#define IQ1 53668
#define IQ2 52774
#define IR1 12211
#define IR2 3791
#define NTAB RANA
#define NDIV (1+IMM1/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)

  Ran2::result_type Ran2::operator () () {
    int j;
    long k;
    Real temp;
      
    if (idum <= 0) {
      if (-(idum) < 1) idum=1;
      else idum = -(idum);
      idum2=idum;
      for (j=NTAB+7;j>=0;j--) {
	k=idum/IQ1;
	idum=IA1*(idum-k*IQ1)-k*IR1;
	if (idum < 0) idum += IM1;
	if (j < NTAB) iv[j] = idum;
      }
      iy=iv[0];
    }
    k=idum/IQ1;
    idum=IA1*(idum-k*IQ1)-k*IR1;
    if (idum < 0) idum += IM1;
    k=idum2/IQ2;
    idum2=IA2*(idum2-k*IQ2)-k*IR2;
    if (idum2 < 0) idum2 += IM2;
    j=iy/NDIV;
    iy=iv[j]-idum2;
    iv[j] = idum;
    if (iy < 1) iy += IMM1;
    if ((temp=AM*iy) > RNMX) {
      return RNMX;
    }
    else {
      return temp;
    }
  }

#undef IM1
#undef IM2
#undef AM
#undef IMM1
#undef IA1
#undef IA2
#undef IQ1
#undef IQ2
#undef IR1
#undef IR2
#undef NTAB
#undef NDIV
#undef EPS
#undef RNMX
  
} /* namespace psimag */


#endif /* PSIMAG_Ran2_H */
