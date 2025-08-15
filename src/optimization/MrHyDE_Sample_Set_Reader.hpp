/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.

 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
************************************************************************/

#ifndef ROL_SAMPLE_SET_READER_HPP
#define ROL_SAMPLE_SET_READER_HPP

#include "ROL_SampleGenerator.hpp"
#include "ROL_BatchManager.hpp"
#include <fstream>
#include <iostream>
#include <string>

namespace ROL {

template<typename Real> 
class Sample_Set_Reader : public SampleGenerator<Real> {
private:
  int nSamp_;

  void sample(int n, int dim,
              const ROL::Ptr<BatchManager<Real>> &bman) {
    std::string file_pt = "sample_set.dat";
    nSamp_ = n;
    // Read in full point data and weight data
    std::fstream input_pt;
    input_pt.open(file_pt.c_str(), std::ios::in);
    if (!input_pt.is_open())
    {
      if (bman->batchID() == 0)
      {
        std::cout << "CANNOT OPEN " << file_pt.c_str() << std::endl;
      }
    }
    else
    {
      std::vector<std::vector<Real>> pt(n);
      std::vector<Real> wt(n,1.0/static_cast<Real>(n));
      std::vector<Real> point(dim,0.0);;
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < dim; j++) {
          input_pt >> point[j];
        } 
        pt[i] = point;
      }
      // Get process rankd and number of processes
      int rank  = bman->batchID();
      int nProc = bman->numBatches();
      // Separate samples across processes
      int frac = n/nProc;
      int rem  = n%nProc;
      int N    = frac;
      if ( rank < rem ) N++;
      std::vector<std::vector<Real>> my_pt(N);
      std::vector<Real> my_wt(N,0.0);
      int index = 0;
      for (int i = 0; i < N; i++) {
        index = i*nProc + rank;
        my_pt[i] = pt[index];  
        my_wt[i] = wt[index];
      }
      SampleGenerator<Real>::setPoints(my_pt);
      SampleGenerator<Real>::setWeights(my_wt);
    }
    input_pt.close();
  }

public:
  // For simplicity, this code only considers Monte Carlo samples with equal weights. This can be generalized.
  Sample_Set_Reader(int n,
                    int dim,
                    const ROL::Ptr<BatchManager<Real>> &bman)
      : SampleGenerator<Real>(bman)
  {
    sample(n, dim, bman);
  }

  void refine(void) {}

  int numGlobalSamples(void) const {
    return nSamp_;
  }
};

}

#endif
