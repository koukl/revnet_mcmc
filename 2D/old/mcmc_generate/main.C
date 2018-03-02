#include <stdlib.h>
#include <fstream>
#include <vector>
#include <algorithm>

// generate starting from all spin aligned at infinite temperature
// for some time T1 and T2, extract spin config at T1 and T2

// -------------------------------------------------------
void mcs(std::vector<int>& s,const unsigned int T) {
  for(unsigned int t=0;t<T;++t) {
    unsigned int chosen_site = (unsigned int)(drand48()*s.size());
    if(chosen_site==s.size()) chosen_site = s.size()-1;
    s[chosen_site] = -1*s[chosen_site];
  }
}
// -------------------------------------------------------
int get_energy(std::vector<int>& s) {
  int e = 0;
  for(unsigned int i=0;i<s.size()-1;++i) {
    e += -1*s[i]*s[i+1];
  }
  return e;
}
// -------------------------------------------------------
void print_header(std::ofstream& out,const std::vector<int>& s) {

  std::for_each(s.begin(),s.end()-1,[&out](int v) { out<<"x,"; });
  out<<"x";
  out<<std::endl;
}
// -------------------------------------------------------
void print_config(std::ofstream& out,const std::vector<int>& s) {

  std::for_each(s.begin(),s.end()-1,[&out](int v) { out<<v<<","; });
  out<<s[s.size()-1];
  out<<std::endl;
}
// -------------------------------------------------------
void generate_data(const unsigned int L,const unsigned int ndata,
                   const std::string& outfilex,const std::string& outfiley) {

  const unsigned int T=L/4;
  std::ofstream xout(outfilex);
  std::ofstream yout(outfiley);

//  std::vector<int> header_placeholder(L,1);
//  print_header(xout,header_placeholder);
//  yout<<"E\n";

  for(unsigned int d=0;d<ndata;++d) {
    std::vector<int>   s(L,1); // initialize all ising spin up
    mcs(s,T);
    print_config(xout,s);
    yout<<get_energy(s)<<std::endl;
  }
  yout.close();
  xout.close();
}
// -------------------------------------------------------
int main ( ) {

  const unsigned int L=16;
  const unsigned int ndata=20;

  const unsigned int seed = 89417;
  srand48(seed);

  generate_data(L,ndata,std::string("trainx1.dat"),std::string("trainy1.dat"));
  generate_data(L,ndata,std::string("trainx2.dat"),std::string("trainy2.dat"));
//  generate_data(L,ndata,std::string("testx1.dat"),std::string("testy1.dat"));
}

// -------------------------------------------------------

