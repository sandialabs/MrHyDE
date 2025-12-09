/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

/** \file fftInterface.hpp
 *  \brief Contains the interface to the FFTW library.
 *  \author Created by B. Reuter and modified by T. Wildey
 */

#ifndef MRHYDE_FFTINTERFACE_H
#define MRHYDE_FFTINTERFACE_H

#include "mpi.h"
#include "fftw3.h"
#include <stdio.h>
#include <iostream>
#include <cmath>
#include "trilinos.hpp"

namespace MrHyDE {

  /**
   * \class fftInterface
   * \brief Interface class providing FFT-related operations using FFTW.
   *
   * The class serves as an abstraction layer for performing FFT operations
   * on multi-dimensional data structures using FFTW, potentially across MPI
   * processes. It provides routines for setup and computation of forward or
   * inverse transforms, as well as handling sensor-based data layouts.
   */
  class fftInterface {
  public:

    // ========================================================================================
    /**
     * \brief Default constructor.
     *
     * Initializes an empty FFT interface object. No FFTW plans or data
     * structures are allocated at this point.
     */
    // ========================================================================================
    fftInterface() {};

    /**
     * \brief Destructor.
     *
     * Frees any allocated FFTW resources if applicable.
     */
    ~fftInterface() {};

    /**
     * \brief Perform FFT computations on the given data.
     *
     * Runs FFT-based operations on the provided 3D data array and sensor ID list.
     * The nature of the FFT (forward/inverse) and additional operations depend
     * on the implementation within this method.
     *
     * \param data  A 3D Kokkos view holding the real-space data values.
     * \param IDs   A 1D Kokkos view containing sensor identifiers for indexing.
     * \param total_sensors  Total number of sensors used for the computation.
     *
     * \return void
     */
    void compute(Kokkos::View<ScalarT***,HostDevice> data,
                 Kokkos::View<int*,HostDevice> IDs,
                 const int & total_sensors) {
      
#endif // MRHYDE_FFTINTERFACE_H
      // each MPI process will own a certain number of time series data corresponding
      // to different spatial locations
    
      // they will be transformed to the frequency domain
      // hence we need a bunch of 1D transforms on each MPI process
  
      int num_sensors = data.extent_int(0); // number of time histories owned by the process
      int num_fields = data.extent_int(1); 
      int num_snaps = data.extent_int(2); // number of equispaced snapshots in the time domain
    
      //int num_sensors_total = total_sensors; // TODO also garbage
    
      // PHDF5 creation
      // TODO SETUP MPI COMMUNICATOR BETWEEN PROCESSES THAT ACTUALLY HAVE SENSORS
      /*
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
      f_id = H5Fcreate("fft_data.h5",H5F_ACC_TRUNC, // overwrites file if it exists
                      H5P_DEFAULT,fapl_id);

      // free the file access template
      err = H5Pclose(fapl_id);

      // create the dataspace

      hid_t ds_id;
      hsize_t dims[3] = {num_sensors_total,num_fields,num_snaps};
      ds_id = H5Screate_simple(3, dims, // [sensor_id,spatial_dim,snap]
                              NULL);

      // need to create a new hdf5 datatype which matches fftw_complex
      // TODO not sure about this...
      // TODO change ??
      hsize_t comp_dims[1] = {2};
      hid_t complex_id = H5Tarray_create2(H5T_NATIVE_DOUBLE,1,comp_dims);
      
      // create the B and E frequency domain storage
      hid_t field_id;
      field_id = H5Dcreate2(f_id,"B_freq",complex_id,ds_id,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
      */

      fftw_complex *myData, *myOutData;
      //myData = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*num_snaps*num_fields);
      //myOutData = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*num_snaps*num_fields);
      myData = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*num_snaps);
      myOutData = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*num_snaps);

      //---------------------------------------------------
      // Perform FFT
      //---------------------------------------------------
      fftw_plan plan;
      plan = fftw_plan_dft_1d(num_snaps, myData, myOutData, FFTW_FORWARD, FFTW_MEASURE);
      
      for (int sens=0; sens<num_sensors; ++sens) {

        
        for (int i=0; i<num_fields; i++) {
          for (int j=0; j<num_snaps; j++) {
            myData[j][0] = data(sens,i,j); 
            myData[j][1] = 0.0;
          }
          fftw_execute_dft(plan, myData, myOutData);

          for (int j=0; j<num_snaps; j++) {
            data(sens,i,j) = myOutData[j][0];
          }
        }
        
      }
      fftw_destroy_plan(plan);

      /*
      for (int sens=0; sens<num_sensors; ++sens) {

        
        for (unsigned int i=0; i<num_fields; i++) {
          for (unsigned int j=0; j<num_snaps; j++) {
            myData[i*num_snaps+j][0] = data(sens,i,j); 
            myData[i*num_snaps+j][1] = 0.0;
          }
        }
      
        int rank = 1; // 1D transform...
        int n[] = {num_snaps}; // of size num_snaps...
        int howmany = num_fields; // of which there are num_fields...
        // TODO I am just starting from the basic FFTW example, this can be modified
        // the beginning of each data line is spaced by num_fields in memory
        int idist = num_fields, odist = num_fields;
        int istride = 1, ostride = 1; // distance between elements in a data line
        int *inembed = n, *onembed = n; // TODO don't fully understand this

        fftw_plan plan = fftw_plan_many_dft(rank,n,howmany,myData,inembed,istride,idist,
                                            myData,onembed,ostride,odist,
                                            FFTW_FORWARD, FFTW_ESTIMATE); // forward transform
      
        fftw_execute_dft(plan, myData, myOutData);

        for (unsigned int i=0; i<num_fields; i++) {
          for (unsigned int j=0; j<num_snaps; j++) {
            data(sens,i,j) = myOutData[i*num_snaps+j][0];
            //cout << myData[i*num_snaps+j][0] << "  " << myData[i*num_snaps+j][1] << endl;
          }
        }
        fftw_destroy_plan(plan);
      */
        //---------------------------------------------------
        // Write this data to the HDF5 file
        //---------------------------------------------------
        /*
        // set up the portion of the files this process will access

        hsize_t start[3] = {IDs(sens),0,0};
        //hsize_t count[3] = {num_sensors,num_fields,num_snaps};
        hsize_t count[3] = {1,num_fields,num_snaps};

        err = H5Sselect_hyperslab(ds_id,H5S_SELECT_SET,start,NULL, // contiguous
                                  count,NULL); // contiguous 
      
        // Memory, file dataspace??
        // It is my understanding the the file dataspace describes how the data will be laid out in the file
        // The memory dataspace describes how the data is laid out in memory within the application

        // In this example, the FFTW data is a 1D array (of complex) of length
        // num_snaps*num_fields
        hsize_t flattened[] = {num_snaps*num_fields};
        hid_t ms_id = H5Screate_simple(1,flattened,NULL);
        err = H5Dwrite(field_id,complex_id,ms_id,ds_id,H5P_DEFAULT,myData);
        */
      //}

      /*
      err = H5Dclose(field_id);
      
      H5Sclose(ds_id);
      H5Fclose(f_id);
      */

      
      fftw_free(myData);

    }

    //Teuchos::RCP<Teuchos::ParameterList> settings,
    //Teuchos::RCP<MpiComm> Comm;
    //int nProcs, myRank;
  };
  
  
}

#endif
