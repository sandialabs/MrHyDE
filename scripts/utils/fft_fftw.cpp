#include "mpi.h"
#include "fftw3.h"
#include "fftw3_mkl.h"
#include <stdio.h>
#include <iostream>
#include <cmath>
#include "trilinos.hpp"
#include "hdf5.h"

int main(int argc, char * argv[]) {

  Teuchos::GlobalMPISession mpiSession(&argc, &argv, 0);
  Teuchos::RCP<MpiComm> Comm = Teuchos::rcp( new MpiComm(MPI_COMM_WORLD) );
  int nProcs = mpiSession.getNProc();
  int myRank = mpiSession.getRank();

  // each MPI process will own a certain number of E and B time series data corresponding
  // to different spatial locations
  
  // they will be transformed to the frequency domain
  // hence we need a bunch of 1D transforms on each MPI process
 
  // TODO FOO to determine number of histories, load in data, etc.

  int num_sensors = 5; // number of time histories owned by the process

  int num_sensors_total = 20; // TODO also garbage

  int my_sensors_start = myRank*num_sensors;

  int num_snaps = 10; // number of equispaced snapshots in the time domain
  int space_dim = 3; // spatial dimension
  int num_fields = space_dim*num_sensors; // E and B fields with space_dim components each

  // PHDF5 creation
  // TODO SETUP MPI COMMUNICATOR BETWEEN PROCESSES THAT ACTUALLY HAVE SENSORS
  
  herr_t err; // HDF5 return value
  hid_t f_id; // HDF5 file ID

  // file access property list
  hid_t fapl_id;
  fapl_id = H5Pcreate(H5P_FILE_ACCESS);

  // store MPI IO params
  MPI_Comm sensor_comm = MPI_COMM_WORLD;
  MPI_Info info = MPI_INFO_NULL;
  
  err = H5Pset_fapl_mpio(fapl_id, sensor_comm, info);

  // create the file
  f_id = H5Fcreate("test.h5",H5F_ACC_TRUNC, // overwrites file if it exists
                   H5P_DEFAULT,fapl_id);

  // free the file access template
  err = H5Pclose(fapl_id);

  // create the dataspace

  hid_t ds_id;
  hsize_t dims[3] = {num_sensors_total,space_dim,num_snaps};
  ds_id = H5Screate_simple(3, dims, // [sensor_id,spatial_dim,snap]
                           NULL);

  // need to create a new hdf5 datatype which matches fftw_complex
  // TODO not sure about this...
  // TODO change ??
  hsize_t comp_dims[1] = {2};
  hid_t complex_id = H5Tarray_create2(H5T_NATIVE_DOUBLE,1,comp_dims);
  
  // create the B and E frequency domain storage
  hid_t Bfield_id, Efield_id;
  Bfield_id = H5Dcreate2(f_id,"B_freq",complex_id,ds_id,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
  Efield_id = H5Dcreate2(f_id,"E_freq",complex_id,ds_id,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);

  // set up the portion of the files this process will access

  hsize_t start[3] = {my_sensors_start,0,0};
  hsize_t count[3] = {num_sensors,space_dim,num_snaps};

  err = H5Sselect_hyperslab(ds_id,H5S_SELECT_SET,start,NULL, // contiguous
                            count,NULL); // contiguous 

  // Memory, file dataspace??
  // It is my understanding the the file dataspace describes how the data will be laid out in the file
  // The memory dataspace describes how the data is laid out in memory within the application

  // In this example, the FFTW data is a 1D array (of complex) of length
  // num_snaps*num_fields
  hsize_t flattened[] = {num_snaps*num_fields};
  hid_t ms_id = H5Screate_simple(1,flattened,NULL);

  // FFTW example

  fftw_complex *myEData,*myBData;

  // we will perform ffts of the columns

  myEData = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*num_snaps*num_fields);
  myBData = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*num_snaps*num_fields);

  double x;
  double dx;

  dx = 2.*PI/(num_snaps);

  // can we transform a sine wave?
  for (unsigned int i=0; i<num_snaps; i++) {
    x = i*dx;
    for (unsigned int j=0; j<num_fields; j++) {
      myEData[i*num_fields+j][0] = std::sin(x); myEData[i*num_fields+j][1] = 0.;
      myBData[i*num_fields+j][0] = std::cos(x); myBData[i*num_fields+j][1] = 0.;
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

  fftw_plan Ep,Bp;

  Ep = fftw_plan_many_dft(rank,n,howmany,myEData,inembed,istride,idist,
                                         myEData,onembed,ostride,odist,
                                         FFTW_FORWARD, FFTW_ESTIMATE); // forward transform
  Bp = fftw_plan_many_dft(rank,n,howmany,myBData,inembed,istride,idist,
                                         myBData,onembed,ostride,odist,
                                         FFTW_FORWARD, FFTW_ESTIMATE); // forward transform

  fftw_execute_dft(Ep, myEData, myEData);
  fftw_execute_dft(Bp, myBData, myBData);

//  if (myRank == 0) {
//    for (unsigned int i=0; i<num_snaps; ++i) {
//      for (unsigned int j=0; j<num_fields; ++j) {
//        std::cout << " i,j :: " << i << ", " << j << " " << myData[i*ostride + j][0] << " + j*" << myData[i*ostride + j][1] << std::endl;
//      }
//    }
//  }

  err = H5Dwrite(Efield_id,complex_id,ms_id,ds_id,H5P_DEFAULT,myEData);
  err = H5Dwrite(Bfield_id,complex_id,ms_id,ds_id,H5P_DEFAULT,myBData);

  err = H5Dclose(Efield_id);
  err = H5Dclose(Bfield_id);

  H5Sclose(ds_id);
  H5Fclose(f_id);

  fftw_destroy_plan(Ep);
  fftw_destroy_plan(Bp);
  fftw_free(myEData);
  fftw_free(myBData);

  return 0;

}
