#include "mpi.h"
#include "fftw3.h"
#include "fftw3_mkl.h"
#include <stdio.h>
#include <iostream>
#include <cmath>
#include "trilinos.hpp"

int main(int argc, char * argv[]) {

  Teuchos::GlobalMPISession mpiSession(&argc, &argv, 0);
  Teuchos::RCP<MpiComm> Comm = Teuchos::rcp( new MpiComm(MPI_COMM_WORLD) );
  int nProcs = mpiSession.getNProc();
  int myRank = mpiSession.getRank();

  // each MPI process will own a certain number of E and B time series data corresponding
  // to different spatial locations
  
  // they will be transformed to the frequency domain
  // hence we need a bunch of 1D transforms on each MPI process

  fftw_complex *myData;

  int num_histories = 5; // number of time histories owned by the process

  int num_snaps = 10; // number of equispaced snapshots in the time domain
  int space_dim = 3; // spatial dimension
  int num_fields = space_dim*2*num_histories; // E and B fields with space_dim components each

  // we will perform ffts of the columns

  myData = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*num_snaps*num_fields);

  //for (unsigned int i=0; i<num_snaps*num_fields; ++i) myData[i][0] = i*1., myData[i][1] = 0.; 

  double x;
  double dx;

  dx = 2.*PI/(num_snaps);

  // can we transform a sine wave?
  for (unsigned int i=0; i<num_snaps; i++) {
    x = i*dx;
    for (unsigned int j=0; j<num_fields; j++) {
      myData[i*num_fields+j][0] = std::sin(x); myData[i*num_fields+j][1] = 0.;
    }
  }

  int rank = 1; // 1D transform...
  int n[] = {num_snaps}; // of size num_snaps...
  int howmany = num_fields; // of which there are num_fields...
  // TODO I am just starting from the basic FFTW example, this can be modified
  // the beginning of each data line is contiguous in memory
  int idist = 1, odist = 1;
  int istride = num_fields, ostride = num_fields; // distance between elements in a data line
  int *inembed = n, *onembed = n; // TODO don't fully understand this

  fftw_plan p;

  p = fftw_plan_many_dft(rank,n,howmany,myData,inembed,istride,idist,
                                        myData,onembed,ostride,odist,
                                        FFTW_FORWARD, FFTW_ESTIMATE); // forward transform

  fftw_execute_dft(p, myData, myData);

  if (myRank == 0) {
    for (unsigned int i=0; i<num_snaps; ++i) {
      for (unsigned int j=0; j<num_fields; ++j) {
        std::cout << " i,j :: " << i << ", " << j << " " << myData[i*ostride + j][0] << " + j*" << myData[i*ostride + j][1] << std::endl;
      }
    }
  }

  fftw_destroy_plan(p);
  fftw_free(myData);

  return 0;

}
