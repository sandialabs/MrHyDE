/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.

 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
 ************************************************************************/

// ========================================================================================
// Called after each time step from the solver manager
// Checks if output is required at this time, and write to file or saves for output later
// ========================================================================================

template <class Node>
void PostprocessManager<Node>::record(vector<vector_RCP> &current_soln, const ScalarT &current_time,
                                      const int &stepnum) {

  // Determine if we want to collect QoI, objectives, etc.
  bool write_this_step = false;
  if (stepnum % write_frequency == 0) {
    write_this_step = true;
  }

  // Determine if we want to write to exodus on this time step
  bool write_exodus_this_step = false;
  if (stepnum % exodus_write_frequency == 0) {
    write_exodus_this_step = true;
  }

  // Write to exodus if requested and within user-defined time window for output
  if (write_exodus_this_step && current_time + 1.0e-100 >= exodus_record_start && current_time - 1.0e-100 <= exodus_record_stop) {
    if (write_solution) {
      this->writeSolution(current_soln, current_time);
    }
  }

  // Write all other output if requested and within user-defined time window for output
  if (write_this_step && current_time + 1.0e-100 >= record_start && current_time - 1.0e-100 <= record_stop) {

    if (compute_error) {
      this->computeError(current_soln, current_time);
    }
    if (compute_response || compute_objective) {
      this->computeObjective(current_soln, current_time);
    }
    if (compute_flux_response) {
      this->computeFluxResponse(current_soln, current_time);
    }
    if (compute_integrated_quantities) {
      this->computeIntegratedQuantities(current_soln, current_time);
    }
    if (compute_weighted_norm) {
      this->computeWeightedNorm(current_soln);
    }
    if (store_sensor_solution) {
      this->computeSensorSolution(current_soln, current_time);
    }
  }

  // We only store the full forward state if running optimization, or if user requested it
  if (save_solution) {
    for (size_t set = 0; set < soln.size(); ++set) {
      soln[set]->store(current_soln[set], current_time, 0);
    }
  }
  
  if (write_solution_to_file) {
    linalg->writeStateToFile(current_soln, solution_storage_file, stepnum);
  }
}

// ========================================================================================
// After simulation has completed, write to file or screen output all data saved
// ========================================================================================

template <class Node>
void PostprocessManager<Node>::report()
{

  Teuchos::TimeMonitor localtimer(*reportTimer);

  ////////////////////////////////////////////////////////////////////////////
  // Report the responses
  ////////////////////////////////////////////////////////////////////////////

  if (compute_response && write_response)
  {

    if (Comm->getRank() == 0)
    {
      if (verbosity > 0)
      {
        cout << endl
             << "*********************************************************" << endl;
        cout << "***** Writing responses ******" << endl;
        cout << "*********************************************************" << endl;
      }
    }
    for (size_t obj = 0; obj < objectives.size(); ++obj)
    {
      if (objectives[obj].type == "sensors")
      {
        // First case: sensors just computed states (faster than other case)
        if (objectives[obj].compute_sensor_soln || objectives[obj].compute_sensor_average_soln)
        {

          Kokkos::View<ScalarT ***, HostDevice> sensor_data;
          Kokkos::View<int *, HostDevice> sensorIDs;
          size_t numtimes = 0;
          int numfields = 0;

          int numsensors = objectives[obj].numSensors;

          if (numsensors > 0)
          {
            sensorIDs = Kokkos::View<int *, HostDevice>("sensor IDs owned by proc", numsensors);
            size_t sprog = 0;
            auto sensor_found = objectives[obj].sensor_found;
            for (size_type s = 0; s < sensor_found.extent(0); ++s)
            {
              if (sensor_found(s))
              {
                sensorIDs(sprog) = s;
                ++sprog;
              }
            }
            if (objectives[obj].output_type == "dft") {
              auto dft_data = objectives[obj].sensor_solution_dft;
              size_type numfreq = dft_data.extent(3);
              int numsols = objectives[obj].sensor_solution_data[0].extent_int(1); // does assume this does not change in time, which it shouldn't
              int numdims = objectives[obj].sensor_solution_data[0].extent_int(2);
              numfields = numsols * numdims;
              sensor_data = Kokkos::View<ScalarT ***, HostDevice>("sensor data", numsensors, numfields, numfreq);
              for (size_t t = 0; t < numfreq; ++t) {
                for (int sens = 0; sens < numsensors; ++sens) {
                  size_t solprog = 0;
                  for (int sol = 0; sol < numsols; ++sol) {
                    for (int d = 0; d < numdims; ++d) {
                      sensor_data(sens, solprog, t) = dft_data(sens, sol, d, t).real();
                      solprog++;
                    }
                  }
                }
              }
            }
            else if (objectives[obj].output_type == "integrated dft")
            {
              
              auto dft_data = objectives[obj].sensor_solution_dft;
              
              // Grab some parameters from settings
              int numtheta = settings->sublist("Postprocess").get("NF2FF number theta", 1);
              ScalarT mintheta = settings->sublist("Postprocess").get("NF2FF min theta", 0.0);
              ScalarT maxtheta = settings->sublist("Postprocess").get("NF2FF max theta", 0.0);
              int numphi = settings->sublist("Postprocess").get("NF2FF number theta", 1);
              ScalarT minphi = settings->sublist("Postprocess").get("NF2FF min phi", 0.0);
              ScalarT maxphi = settings->sublist("Postprocess").get("NF2FF max phi", 0.0);
              ScalarT k0 = settings->sublist("Postprocess").get("NF2FF wave number", 0.0);
              ScalarT N0 = settings->sublist("Postprocess").get("NF2FF freespace impedence", 0.0);
              
              // Create the vectors of PHI and THETA
              vector<ScalarT> THETA(numtheta), PHI(numphi);
              if (numtheta>1) {
                ScalarT dtheta = (maxtheta-mintheta)/(numtheta-1);
                for (size_t k=0; k<numtheta; ++k) {
                  THETA[k] = mintheta + k*dtheta;
                }
              }
              else {
                THETA[0] = mintheta;
              }
              
              if (numphi>1) {
                ScalarT dphi = (maxphi-minphi)/(numphi-1);
                for (size_t k=0; k<numphi; ++k) {
                  PHI[k] = minphi + k*dphi;
                }
              }
              else {
                PHI[0] = minphi;
              }
              
              auto numfreq = dft_data.extent(3);
              // Initialize vector potentials
              Kokkos::View<std::complex<ScalarT>***,HostDevice> A_th = Kokkos::View<std::complex<ScalarT>***,HostDevice>("NF2FF A_th", numfreq, numtheta, numphi);
              Kokkos::View<std::complex<ScalarT>***,HostDevice> A_ph = Kokkos::View<std::complex<ScalarT>***,HostDevice>("NF2FF A_ph", numfreq, numtheta, numphi);
              Kokkos::View<std::complex<ScalarT>***,HostDevice> F_th = Kokkos::View<std::complex<ScalarT>***,HostDevice>("NF2FF F_th", numfreq, numtheta, numphi);
              Kokkos::View<std::complex<ScalarT>***,HostDevice> F_ph = Kokkos::View<std::complex<ScalarT>***,HostDevice>("NF2FF F_ph", numfreq, numtheta, numphi);
              
              // Need to make a few assumptions here:
              // 1. The sensors are the boundary quadrature points on a given block
              // 2. They are in the same order as looping through the boundary groups and quadrature points
              // 3. A given block has all of its boundary quadrature points as sensors
              // 4. The data stored in dft_data is the dft of E at the quadrature points
              // 5. The problem is 3D, not 1D or 2D
              
              size_t iblock = objectives[obj].block;
              int prog = 0; // increments sensors
              for (size_t grp=0; grp<assembler->boundary_groups[iblock].size(); ++grp) {
                // These arrays live on the AssemblyDevice, which may not be the Host, i.e., will not run on GPU without data transfer
                vector<View_Sc2> ip = assembler->boundary_groups[iblock][grp]->ip;
                vector<View_Sc2> normals = assembler->boundary_groups[iblock][grp]->normals;
                View_Sc2 wts = assembler->boundary_groups[iblock][grp]->wts;
                
                for (int nt=0; nt<numtheta; ++nt) {
                  for (int np=0; np<numphi; ++np) {
                    
                    // Cartesian to spherical transform
                    vector<ScalarT> r_hat = {std::sin(THETA[nt])*std::cos(PHI[np]), std::sin(THETA[nt])*std::sin(PHI[np]), std::cos(THETA[nt])};
                    vector<ScalarT> theta_hat = {std::cos(THETA[nt])*std::cos(PHI[np]), std::cos(THETA[nt])*std::sin(PHI[np]), -std::sin(THETA[nt])};
                    vector<ScalarT> phi_hat = {-std::sin(PHI[np]), std::cos(PHI[np]), 0.0};
                    
                    // Compute phase at quadrature points (assumes 3D)
                    for (size_type elem=0; elem<wts.extent(0); ++elem) {
                      for (size_type pt=0; pt<wts.extent(1); ++pt) {
                        
                        vector<ScalarT> phase = {k0*ip[0](elem,pt)*r_hat[0], k0*ip[1](elem,pt)*r_hat[1], k0*ip[2](elem,pt)*r_hat[2] };
                        vector<std::complex<ScalarT> > phasor;
                        phasor.push_back(std::complex<ScalarT>(std::cos(phase[0]), -std::sin(phase[0])));
                        phasor.push_back(std::complex<ScalarT>(std::cos(phase[1]), -std::sin(phase[1])));
                        phasor.push_back(std::complex<ScalarT>(std::cos(phase[2]), -std::sin(phase[2])));
                        
                        vector<std::complex<ScalarT>> E_theta = { theta_hat[0]*phasor[0], theta_hat[1]*phasor[1], theta_hat[2]*phasor[2]};
                        vector<std::complex<ScalarT>> E_phi = { phi_hat[0]*phasor[0], phi_hat[1]*phasor[1], phi_hat[2]*phasor[2]};
                        
                        for (size_t t=0; t<numfreq; ++t) {
                          vector<std::complex<ScalarT>> Esrc = {dft_data(prog,0,0,t), dft_data(prog,0,1,t), dft_data(prog,0,2,t)};
                          
                          // Compute normal x E_theta, normal x Escr, normal x E_phi
                          vector<std::complex<ScalarT>> n_x_E_theta = {normals[1](elem,pt)*E_theta[2] - normals[2](elem,pt)*E_theta[1], normals[2](elem,pt)*E_theta[0] - normals[0](elem,pt)*E_theta[2], normals[0](elem,pt)*E_theta[1] - normals[1](elem,pt)*E_theta[0]};
                          vector<std::complex<ScalarT>> n_x_Esrc = {normals[1](elem,pt)*Esrc[2] - normals[2](elem,pt)*Esrc[1], normals[2](elem,pt)*Esrc[0] - normals[0](elem,pt)*Esrc[2], normals[0](elem,pt)*Esrc[1] - normals[1](elem,pt)*Esrc[0]};
                          vector<std::complex<ScalarT>> n_x_E_phi = {normals[1](elem,pt)*E_phi[2] - normals[2](elem,pt)*E_phi[1], normals[2](elem,pt)*E_phi[0] - normals[0](elem,pt)*E_phi[2], normals[0](elem,pt)*E_phi[1] - normals[1](elem,pt)*E_phi[0]};
                          
                          // Sum into the vector potentials
                          A_th(t,nt,np) += -1.0/N0*wts(elem,pt)*(n_x_E_theta[0]*n_x_Esrc[0] + n_x_E_theta[1]*n_x_Esrc[1] + n_x_E_theta[2]*n_x_Esrc[2]);
                          A_ph(t,nt,np) += -1.0/N0*wts(elem,pt)*(n_x_E_phi[0]*n_x_Esrc[0] + n_x_E_phi[1]*n_x_Esrc[1] + n_x_E_phi[2]*n_x_Esrc[2]);
                          F_th(t,nt,np) += -1.0/N0*wts(elem,pt)*(E_theta[0]*n_x_Esrc[0] + E_theta[1]*n_x_Esrc[1] + E_theta[2]*n_x_Esrc[2]);
                          F_ph(t,nt,np) += -1.0/N0*wts(elem,pt)*(E_phi[0]*n_x_Esrc[0] + E_phi[1]*n_x_Esrc[1] + E_phi[2]*n_x_Esrc[2]);
                        }
                      }
                    }
                    prog++;
                  }
                }
                
              }
              /*
              auto dft_data = objectives[obj].sensor_solution_dft;
              size_type numfreq = dft_data.extent(3);
              int numsols = objectives[obj].sensor_solution_data[0].extent_int(1); // does assume this does not change in time, which it shouldn't
              int numdims = objectives[obj].sensor_solution_data[0].extent_int(2);
              numfields = numsols * numdims;
              // need number of integrals to compute
              sensor_data = Kokkos::View<ScalarT ***, HostDevice>("sensor data", numsensors, numfields, numfreq);
              for (size_t t = 0; t < numfreq; ++t) {
                for (int sens = 0; sens < numsensors; ++sens) {
                  size_t solprog = 0;
                  for (int sol = 0; sol < numsols; ++sol) {
                    for (int d = 0; d < numdims; ++d) {
                      sensor_data(sens, solprog, t) = dft_data(sens, sol, d, t).real();
                      solprog++;
                    }
                  }
                }
              }
               */
              //sensor_data = Kokkos::View<ScalarT ***, HostDevice>("sensor data", numsensors, 7, numfreq*numtheta*numphi);
              /*
              ScalarT r = 1.0; // distance into the far field
              int prog = 0;
              for (size_t t=0; t<numfreq; ++t) {
                for (int nt=0; nt<numtheta; ++nt) {
                  for (int np=0; np<numphi; ++np) {
                  }
                }
              }
              DAT.Eth = k0./(4*pi*r)*abs(N0.*A_th + F_ph);
              DAT.Eph = k0./(4*pi*r)*abs(N0.*A_ph - F_th);

              DAT.Pth = k0^2/(4*pi)^2 * 1/N0 * abs(N0.*A_th + F_ph).^2;
              DAT.Pph = k0^2/(4*pi)^2 * 1/N0 * abs(N0.*A_ph - F_th).^2;

              % COMPUTE BISTATIC RCS (same as directivity but with Pinc instead of Prad)
              % Einc = [0;0;1]; %Incident electric field
              % % EP = te*ate + tm*atm; %this is Einc
              % Pinc = norm(Einc).^2./(2*N0);  %Polarization vector
              Pinc = norm(SRC.PlaneWave.EP).^2./(2*N0);

              DAT.RCSth = k0^2/(8*pi*N0*Pinc).*(abs(F_ph + N0.*A_th).^2);
              DAT.RCSph = k0^2/(8*pi*N0*Pinc).*(abs(F_th - N0.*A_ph).^2);
              DAT.RCStot = DAT.RCSth + DAT.RCSph;
              */
            }
            else
            {
              numtimes = objectives[obj].sensor_solution_data.size();              // vector of Kokkos::Views
              int numsols = objectives[obj].sensor_solution_data[0].extent_int(1); // does assume this does not change in time, which it shouldn't
              int numdims = objectives[obj].sensor_solution_data[0].extent_int(2);
              numfields = numsols * numdims;
              sensor_data = Kokkos::View<ScalarT ***, HostDevice>("sensor data", numsensors, numfields, numtimes);
              for (size_t t = 0; t < numtimes; ++t)
              {
                auto sdat = objectives[obj].sensor_solution_data[t];
                for (int sens = 0; sens < numsensors; ++sens)
                {
                  size_t solprog = 0;
                  for (int sol = 0; sol < numsols; ++sol)
                  {
                    for (int d = 0; d < numdims; ++d)
                    {
                      sensor_data(sens, solprog, t) = sdat(sens, sol, d);
                      solprog++;
                    }
                  }
                }
              }
              if (objectives[obj].output_type == "fft")
              {
#if defined(MrHyDE_ENABLE_FFTW)
                fft->compute(sensor_data, sensorIDs, global_num_sensors);
#endif
              }
            }
          }

          size_t max_numtimes = 0;
          Teuchos::reduceAll(*Comm, Teuchos::REDUCE_MAX, 1, &numtimes, &max_numtimes);

          int max_numfields = 0;
          Teuchos::reduceAll(*Comm, Teuchos::REDUCE_MAX, 1, &numfields, &max_numfields);

          if (fileoutput == "text" && objectives[obj].output_type != "integrated dft") {

            for (int field = 0; field < max_numfields; ++field) {
              std::stringstream ss;
              size_t blocknum = objectives[obj].block;
              ss << field;
              string respfile = "sensor_solution_field." + ss.str() + "." + blocknames[blocknum] + ".out";
              std::ofstream respOUT;
              if (Comm->getRank() == 0) {
                bool is_open = false;
                int attempts = 0;
                int max_attempts = 100;
                while (!is_open && attempts < max_attempts) {
                  respOUT.open(respfile);
                  is_open = respOUT.is_open();
                  attempts++;
                }
                respOUT.precision(8);
                Teuchos::Array<ScalarT> time_data(max_numtimes + dimension, 0.0);
                for (int dim = 0; dim < dimension; ++dim) {
                  time_data[dim] = 0.0;
                }

                for (size_t tt = 0; tt < max_numtimes; ++tt) {
                  time_data[tt + dimension] = objectives[obj].response_times[tt];
                }

                for (size_t tt = 0; tt < max_numtimes + dimension; ++tt) {
                  respOUT << time_data[tt] << "  ";
                }
                respOUT << endl;
              }

              auto spts = objectives[obj].sensor_points;
              for (size_t ss = 0; ss < objectives[obj].sensor_found.size(); ++ss) {
                Teuchos::Array<ScalarT> series_data(max_numtimes + dimension, 0.0);
                Teuchos::Array<ScalarT> gseries_data(max_numtimes + dimension, 0.0);
                if (objectives[obj].sensor_found[ss]) {
                  size_t sindex = 0;
                  for (size_t j = 0; j < ss; ++j) {
                    if (objectives[obj].sensor_found(j)) {
                      sindex++;
                    }
                  }
                  for (int dim = 0; dim < dimension; ++dim) {
                    series_data[dim] = spts(sindex, dim);
                  }

                  for (size_t tt = 0; tt < max_numtimes; ++tt) {
                    series_data[tt + dimension] = sensor_data(sindex, field, tt);
                  }
                }

                const int numentries = max_numtimes + dimension;
                Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, numentries, &series_data[0], &gseries_data[0]);

                if (Comm->getRank() == 0) {
                  // respOUT << gseries_data[0] << "  " << gseries_data[1] << "  " << gseries_data[2] << "  ";
                  for (size_t tt = 0; tt < max_numtimes + dimension; ++tt) {
                    respOUT << gseries_data[tt] << "  ";
                  }
                  respOUT << endl;
                }
              }
              if (Comm->getRank() == 0) {
                respOUT.close();
              }
            }
          }
#ifdef MrHyDE_USE_HDF5
          else if (fileoutput == "hdf5") {
            // PHDF5 creation
            size_t num_snaps = max_numtimes;
            const size_t alength = num_snaps;
            ScalarT *myData = new ScalarT[alength];

            for (int field = 0; field < max_numfields; ++field) {

              herr_t err; // HDF5 return value
              hid_t f_id; // HDF5 file ID

              // file access property list
              hid_t fapl_id;
              fapl_id = H5Pcreate(H5P_FILE_ACCESS);

              err = H5Pset_fapl_mpio(fapl_id, *(Comm->getRawMpiComm()), MPI_INFO_NULL);

              // create the file
              std::stringstream ss;
              ss << field;
              string respfile = "sensor_solution_field." + ss.str() + ".h5";

              f_id = H5Fcreate(respfile.c_str(), H5F_ACC_TRUNC, // overwrites file if it exists
                               H5P_DEFAULT, fapl_id);

              // free the file access template
              err = H5Pclose(fapl_id);

              // create the dataspace

              hid_t ds_id;
              hsize_t dims[2] = {objectives[obj].sensor_found.size(), num_snaps};
              ds_id = H5Screate_simple(2, dims, // [sensor_id,snap]
                                       NULL);

              // need to create a new hdf5 datatype which matches fftw_complex
              // TODO not sure about this...
              // TODO change ??
              hsize_t comp_dims[1] = {1};
              hid_t complex_id = H5Tarray_create2(H5T_NATIVE_DOUBLE, 1, comp_dims);

              // create the storage
              hid_t field_id;
              field_id = H5Dcreate2(f_id, "soln", complex_id, ds_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

              // set up the portion of the files this process will access
              for (size_type sens = 0; sens < sensorIDs.extent(0); ++sens) {
                hsize_t myID = sensorIDs(sens);
                hsize_t start[2] = {myID, 0};
                hsize_t count[2] = {1, num_snaps};

                err = H5Sselect_hyperslab(ds_id, H5S_SELECT_SET, start, NULL, // contiguous
                                          count, NULL);                       // contiguous

                for (size_t s = 0; s < num_snaps; ++s) {
                  myData[s] = sensor_data(sensorIDs(sens), field, s);
                }
                hsize_t flattened[] = {num_snaps};
                hid_t ms_id = H5Screate_simple(1, flattened, NULL);
                err = H5Dwrite(field_id, complex_id, ms_id, ds_id, H5P_DEFAULT, myData);
              }

              err = H5Dclose(field_id);

              if (err > 0) {
                // say something
              }
              H5Sclose(ds_id);
              H5Fclose(f_id);
            }
            delete[] myData;
          }
#endif // MrHyDE_USE_HDF5
        }
        else { // Second case: sensors computed response functions
          string respfile = objectives[obj].response_file + ".out";
          std::ofstream respOUT;
          if (Comm->getRank() == 0) {
            bool is_open = false;
            int attempts = 0;
            int max_attempts = 100;
            while (!is_open && attempts < max_attempts) {
              respOUT.open(respfile);
              is_open = respOUT.is_open();
              attempts++;
            }
            respOUT.precision(16);
          }

          if (Comm->getRank() == 0) {
            for (size_t tt = 0; tt < objectives[obj].response_times.size(); ++tt) {
              respOUT << objectives[obj].response_times[tt] << "  ";
            }
            respOUT << endl;
          }
          for (size_t ss = 0; ss < objectives[obj].sensor_found.size(); ++ss) {
            for (size_t tt = 0; tt < objectives[obj].response_times.size(); ++tt) {
              ScalarT sslval = 0.0, ssgval = 0.0;
              if (objectives[obj].sensor_found[ss]) {
                size_t sindex = 0;
                for (size_t j = 0; j < ss; ++j) {
                  if (objectives[obj].sensor_found(j)) {
                    sindex++;
                  }
                }

                sslval = objectives[obj].response_data[tt](sindex);
              }
              Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, 1, &sslval, &ssgval);
              if (Comm->getRank() == 0) {
                respOUT << ssgval << "  ";
              }
            }
            if (Comm->getRank() == 0) {
              respOUT << endl;
            }
          }
          if (Comm->getRank() == 0) {
            respOUT.close();
          }
        }
      }
      else if (objectives[obj].type == "integrated response") {
        if (objectives[obj].save_data) {
          string respfile = objectives[obj].response_file + "." + blocknames[objectives[obj].block] + append + ".out";
          std::ofstream respOUT;
          if (Comm->getRank() == 0) {
            bool is_open = false;
            int attempts = 0;
            int max_attempts = 100;
            while (!is_open && attempts < max_attempts) {
              respOUT.open(respfile);
              is_open = respOUT.is_open();
              attempts++;
            }
            respOUT.precision(16);
          }
          for (size_t tt = 0; tt < objectives[obj].response_times.size(); ++tt) {

            if (Comm->getRank() == 0) {
              respOUT << objectives[obj].response_times[tt] << "  ";
            }
            double localval = objectives[obj].scalar_response_data[tt];
            double globalval = 0.0;
            Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, 1, &localval, &globalval);
            if (Comm->getRank() == 0) {
              respOUT << globalval;
              respOUT << endl;
            }
          }
          if (Comm->getRank() == 0) {
            respOUT.close();
          }
        }
      }
    }
  }

  if (compute_flux_response)
  {
    if (Comm->getRank() == 0)
    {
      if (verbosity > 0)
      {
        cout << endl
             << "*********************************************************" << endl;
        cout << "***** Computing Flux Responses ******" << endl;
        cout << "*********************************************************" << endl;
      }
    }

    vector<ScalarT> gvals;

    for (size_t f = 0; f < fluxes.size(); ++f)
    {
      for (size_t tt = 0; tt < fluxes[f].vals.extent(0); ++tt)
      {
        ScalarT lval = fluxes[f].vals(tt);
        ScalarT gval = 0.0;
        Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, 1, &lval, &gval);
        gvals.push_back(gval);
      }
      Kokkos::deep_copy(fluxes[f].vals, 0.0);
    }
    if (Comm->getRank() == 0)
    {
      string respfile = "flux_response.out";
      std::ofstream respOUT;
      if (Comm->getRank() == 0)
      {
        bool is_open = false;
        int attempts = 0;
        int max_attempts = 100;
        while (!is_open && attempts < max_attempts)
        {
          respOUT.open(respfile, std::ios_base::app);
          is_open = respOUT.is_open();
          attempts++;
        }
        respOUT.precision(16);
      }

      for (size_t g = 0; g < gvals.size(); ++g)
      {
        cout << gvals[g] << endl;

        respOUT << " " << gvals[g] << "  ";
      }
      respOUT << endl;
      respOUT.close();
    }
  }

  if (compute_integrated_quantities)
  {
    if (Comm->getRank() == 0)
    {
      if (verbosity > 0)
      {
        cout << endl
             << "*********************************************************" << endl;
        cout << "****** Storing Integrated Quantities ******" << endl;
        cout << "*********************************************************" << endl;
      }
    }

    for (size_t iLocal = 0; iLocal < integratedQuantities.size(); iLocal++)
    {

      // iLocal indexes over the number of blocks where IQs are defined and
      // does not necessarily match the global block ID

      size_t globalBlock = integratedQuantities[iLocal][0].block; // all IQs with same first index share a block

      if (Comm->getRank() == 0)
      {
        cout << endl
             << "*********************************************************" << endl;
        cout << "****** Integrated Quantities on block : " << blocknames[globalBlock] << " ******" << endl;
        cout << "*********************************************************" << endl;
        for (size_t k = 0; k < integratedQuantities[iLocal].size(); ++k)
        {
          std::cout << integratedQuantities[iLocal][k].name << " : "
                    << integratedQuantities[iLocal][k].val(0) << std::endl;
        }
      }

    } // end loop over blocks with IQs requested
    // TODO output something? Make the first print statement true!
    // BWR -- this only happens at end of sim.
  } // end if compute_integrated_quantities

  ////////////////////////////////////////////////////////////////////////////
  // Report the errors for verification tests
  ////////////////////////////////////////////////////////////////////////////

  if (compute_error)
  {
    if (Comm->getRank() == 0)
    {
      cout << endl
           << "*********************************************************" << endl;
      cout << "***** Computing errors ******" << endl
           << endl;
    }

    for (size_t block = 0; block < assembler->groups.size(); block++)
    { // loop over blocks
      for (size_t etype = 0; etype < error_list[block].size(); etype++)
      {

        // for (size_t et=0; et<error_types.size(); et++){
        for (size_t time = 0; time < error_times.size(); time++)
        {
          // for (int n=0; n<numVars[block]; n++) {

          ScalarT lerr = errors[time][block](etype);
          ScalarT gerr = 0.0;
          Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, 1, &lerr, &gerr);
          if (Comm->getRank() == 0)
          {
            string varname = error_list[block][etype].first;
            if (error_list[block][etype].second == "L2" || error_list[block][etype].second == "L2 VECTOR")
            {
              cout << "***** L2 norm of the error for " << varname << " = " << sqrt(gerr) << "  (time = " << error_times[time] << ")" << endl;
            }
            else if (error_list[block][etype].second == "L2 FACE")
            {
              cout << "***** L2-face norm of the error for " << varname << " = " << sqrt(gerr) << "  (time = " << error_times[time] << ")" << endl;
            }
            else if (error_list[block][etype].second == "GRAD")
            {
              cout << "***** L2 norm of the error for grad(" << varname << ") = " << sqrt(gerr) << "  (time = " << error_times[time] << ")" << endl;
            }
            else if (error_list[block][etype].second == "DIV")
            {
              cout << "***** L2 norm of the error for div(" << varname << ") = " << sqrt(gerr) << "  (time = " << error_times[time] << ")" << endl;
            }
            else if (error_list[block][etype].second == "CURL")
            {
              cout << "***** L2 norm of the error for curl(" << varname << ") = " << sqrt(gerr) << "  (time = " << error_times[time] << ")" << endl;
            }
          }
          //}
        }
      }
    }

    // Error in subgrid models
    if (!(Teuchos::is_null(multiscale_manager)))
    {
      if (multiscale_manager->getNumberSubgridModels() > 0)
      {

        for (size_t m = 0; m < multiscale_manager->getNumberSubgridModels(); m++)
        {
          vector<string> sgvars = multiscale_manager->subgridModels[m]->varlist;
          vector<std::pair<string, string>> sg_error_list;
          // A given processor may not have any elements that use this subgrid model
          // In this case, nothing gets initialized so sgvars.size() == 0
          // Find the global max number of sgvars over all processors
          size_t nvars = sgvars.size();
          if (nvars > 0)
          {
            sg_error_list = multiscale_manager->subgridModels[m]->getErrorList();
          }
          // really only works on one block
          size_t nerrs = sg_error_list.size();
          size_t gnerrs = 0;
          Teuchos::reduceAll(*Comm, Teuchos::REDUCE_MAX, 1, &nerrs, &gnerrs);

          for (size_t etype = 0; etype < gnerrs; etype++)
          {
            for (size_t time = 0; time < error_times.size(); time++)
            {
              // Get the local contribution (if processor uses subgrid model)
              ScalarT lerr = 0.0;
              if (subgrid_errors[time][0][m].extent(0) > 0)
              {
                lerr = subgrid_errors[time][0][m](etype); // block is not relevant
              }
              ScalarT gerr = 0.0;
              Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, 1, &lerr, &gerr);

              // Figure out who can print the information (lowest rank amongst procs using subgrid model)
              int myID = Comm->getRank();
              if (nvars == 0)
              {
                myID = 100000000;
              }
              int gID = 0;
              Teuchos::reduceAll(*Comm, Teuchos::REDUCE_MIN, 1, &myID, &gID);

              if (Comm->getRank() == gID)
              {

                string varname = sg_error_list[etype].first;
                if (sg_error_list[etype].second == "L2" || sg_error_list[etype].second == "L2 VECTOR")
                {
                  cout << "***** Subgrid " << m << ": L2 norm of the error for " << varname << " = " << sqrt(gerr) << "  (time = " << error_times[time] << ")" << endl;
                }
                else if (sg_error_list[etype].second == "L2 FACE")
                {
                  cout << "***** Subgrid " << m << ": L2-face norm of the error for " << varname << " = " << sqrt(gerr) << "  (time = " << error_times[time] << ")" << endl;
                }
                else if (sg_error_list[etype].second == "GRAD")
                {
                  cout << "***** Subgrid " << m << ": L2 norm of the error for grad(" << varname << ") = " << sqrt(gerr) << "  (time = " << error_times[time] << ")" << endl;
                }
                else if (sg_error_list[etype].second == "DIV")
                {
                  cout << "***** Subgrid " << m << ": L2 norm of the error for div(" << varname << ") = " << sqrt(gerr) << "  (time = " << error_times[time] << ")" << endl;
                }
                else if (sg_error_list[etype].second == "CURL")
                {
                  cout << "***** Subgrid " << m << ": L2 norm of the error for curl(" << varname << ") = " << sqrt(gerr) << "  (time = " << error_times[time] << ")" << endl;
                }
              }
              //}
            }
          }
        }
      }
    }
  }

  if (compute_weighted_norm)
  {
    if (Comm->getRank() == 0)
    {
      string respfile = "weighted_norms.out";
      std::ofstream respOUT;
      respOUT.open(respfile);
      for (size_t k = 0; k < weighted_norms.size(); ++k)
      {
        respOUT << weighted_norms[k] << endl;
      }
      respOUT << endl;
      respOUT.close();
    }
  }
}

// ========================================================================================
// ========================================================================================

template <class Node>
void PostprocessManager<Node>::computeWeightedNorm(vector<vector_RCP> &current_soln)
{

  debugger->print(1, "**** Starting PostprocessManager::computeWeightedNorm()");

  Teuchos::TimeMonitor localtimer(*computeWeightedNormTimer);

  typedef typename Node::execution_space LA_exec;
  typedef typename Node::device_type LA_device;

  if (!have_norm_weights)
  {
    for (size_t set = 0; set < current_soln.size(); ++set)
    {
      vector_RCP wts_over = linalg->getNewOverlappedVector(set);
      assembler->getWeightVector(set, wts_over);
      vector_RCP set_norm_wts = linalg->getNewVector(set);
      set_norm_wts->putScalar(0.0);
      set_norm_wts->doExport(*wts_over, *(linalg->exporter[set]), Tpetra::REPLACE);
      norm_wts.push_back(set_norm_wts);
    }
    have_norm_weights = true;
  }

  ScalarT totalnorm = 0.0;
  for (size_t set = 0; set < current_soln.size(); ++set)
  {
    // current_soln is an overlapped vector ... we want
    vector_RCP soln = linalg->getNewVector(set);
    soln->putScalar(0.0);
    soln->doExport(*(current_soln[set]), *(linalg->exporter[set]), Tpetra::REPLACE);

    vector_RCP prod = linalg->getNewVector(set);

    auto wts_view = norm_wts[set]->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
    auto prod_view = prod->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
    auto soln_view = soln->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
    parallel_for("assembly insert Jac", RangePolicy<LA_exec>(0, prod_view.extent(0)), MRHYDE_LAMBDA(const int k) { prod_view(k, 0) = wts_view(k, 0) * soln_view(k, 0) * soln_view(k, 0); });

    Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> l2norm(1);
    prod->norm2(l2norm);
    totalnorm += l2norm[0];
  }

  weighted_norms.push_back(totalnorm);

  if (verbosity >= 10 && Comm->getRank() == 0)
  {
    cout << "Weighted norm of solution: " << totalnorm << endl;
  }

  debugger->print(1, "**** Finished PostprocessManager::computeWeightedNorm()");
}


// ========================================================================================
// ========================================================================================

template <class Node>
ScalarT PostprocessManager<Node>::makeSomeNoise(ScalarT stdev)
{
  // generate sample from 0-centered normal with stdev
  // Box-Muller method
  // srand(time(0)); //doing this more frequently than once-per-second results in getting the same numbers...
  ScalarT U1 = rand() / ScalarT(RAND_MAX);
  ScalarT U2 = rand() / ScalarT(RAND_MAX);

  return stdev * sqrt(-2 * log(U1)) * cos(2 * PI * U2);
}

// ========================================================================================
// ========================================================================================

template <class Node>
void PostprocessManager<Node>::setTimeIndex(const int &cindex)
{
  time_index = cindex;
}

// ========================================================================================
// ========================================================================================

template <class Node>
Teuchos::Array<ScalarT> PostprocessManager<Node>::collectResponses()
{

  //
  // May be multiple objectives, which store the responses
  // Each objective can store different types of responses:
  //   1.  Integrated response: scalar over time (each proc stores own contribution)
  //   2.  Sensor response: scalar over time at each sensor (each proc stores only own sensor)
  //   3.  Sensor solution: state variable over time at each sensor (each proc stores only own sensor)
  //

  Teuchos::Array<ScalarT> globalarraydata;

  ////////////////////////////////
  // First, determne how many responses have been computed
  ////////////////////////////////

  int totalresp = 0;
  vector<int> response_sizes;
  for (size_t obj = 0; obj < objectives.size(); ++obj)
  {
    if (objectives[obj].type == "sensors")
    {
      int totalsens = objectives[obj].sensor_found.extent_int(0); // this is actually the global number of sensors
      if (objectives[obj].compute_sensor_soln || objectives[obj].compute_sensor_average_soln)
      {
        size_t numtimes = objectives[obj].sensor_solution_data.size();
        int numsols = objectives[obj].sensor_solution_data[0].extent_int(1);
        int numdims = objectives[obj].sensor_solution_data[0].extent_int(2);
        totalsens *= numtimes * numsols * numdims;
      }
      else
      {
        size_t numtimes = objectives[obj].response_times.size();
        totalsens *= numtimes;
      }
      response_sizes.push_back(totalsens);
      // totalresp += totalsens;
    }
    else if (objectives[obj].type == "integrated response")
    {
      response_sizes.push_back(objectives[obj].response_times.size());
      // totalresp += objectives[obj].response_times.size();
    }
  }

  for (size_t i = 0; i < response_sizes.size(); ++i)
  {
    totalresp += response_sizes[i];
  }

  if (totalresp > 0)
  {
    int glbresp = 0;
    Teuchos::reduceAll(*Comm, Teuchos::REDUCE_MAX, 1, &totalresp, &glbresp);
    Kokkos::View<ScalarT *, HostDevice> newresp("response", glbresp);
    Teuchos::Array<ScalarT> localarraydata(glbresp, 0.0);

    ////////////////////////////////
    // Next, we fill in the responses
    ////////////////////////////////

    size_t overallprog = 0;
    for (size_t obj = 0; obj < objectives.size(); ++obj)
    {
      if (objectives[obj].type == "sensors")
      {
        int numsensors = objectives[obj].numSensors;
        if (numsensors > 0)
        {
          Kokkos::View<int *, HostDevice> sensorIDs("sensor IDs owned by proc", numsensors);
          size_t sprog = 0;
          auto sensor_found = objectives[obj].sensor_found;
          for (size_type s = 0; s < sensor_found.extent(0); ++s)
          {
            if (sensor_found(s))
            {
              sensorIDs(sprog) = s;
              ++sprog;
            }
          }

          if (objectives[obj].compute_sensor_soln || objectives[obj].compute_sensor_average_soln)
          {
            for (int sens = 0; sens < numsensors; ++sens)
            {
              size_t numtimes = objectives[obj].sensor_solution_data.size();
              int numsols = objectives[obj].sensor_solution_data[0].extent_int(1);
              int numdims = objectives[obj].sensor_solution_data[0].extent_int(2);
              int sensnum = sensorIDs(sens);
              size_t startind = overallprog + sensnum * (numsols * numdims * numtimes);
              size_t cprog = 0;
              for (size_t tt = 0; tt < numtimes; ++tt)
              {
                for (int ss = 0; ss < numsols; ++ss)
                {
                  for (int dd = 0; dd < numdims; ++dd)
                  {
                    localarraydata[startind + cprog] = objectives[obj].sensor_solution_data[tt](sens, ss, dd);
                    cprog++;
                  }
                }
              }
            }
          }
          else
          {
            for (int sens = 0; sens < numsensors; ++sens)
            {
              size_t numtimes = objectives[obj].response_data.size();
              int sensnum = sensorIDs(sens);
              size_t startind = overallprog + sensnum * (numtimes);
              size_t cprog = 0;
              for (size_t tt = 0; tt < numtimes; ++tt)
              {
                localarraydata[startind + cprog] = objectives[obj].response_data[tt](sens);
                cprog++;
              }
            }
          }
        }
      }
      else if (objectives[obj].type == "integrated response")
      {
        for (size_t tt = 0; tt < objectives[obj].response_times.size(); ++tt)
        {
          localarraydata[overallprog + tt] = objectives[obj].scalar_response_data[tt];
        }
      }
      overallprog += response_sizes[obj];
    }

    globalarraydata = Teuchos::Array<ScalarT>(glbresp, 0.0);

    const int numentries = totalresp;
    Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, numentries, &localarraydata[0], &globalarraydata[0]);
  }

  return globalarraydata;
}

// ========================================================================================
// ========================================================================================

template <class Node>
void PostprocessManager<Node>::resetSolutions() {
  for (size_t set = 0; set < soln.size(); ++set) {
    soln[set]->reset();
  }
}

// ========================================================================================
// ========================================================================================

template <class Node>
void PostprocessManager<Node>::writeQuadratureData() {
  
  // Separate files for different element blocks - can change later if one file is better
  string ptsfile_base = settings->sublist("Postprocessing").get<string>("quadrature points file", "qpts");
  string wtsfile_base = settings->sublist("Postprocessing").get<string>("quadrature weights file", "qwts");
    
  for (size_t block=0; block<blocknames.size(); ++block) {
    string ptsfile = ptsfile_base + "." + blocknames[block] + ".dat";
    string wtsfile = wtsfile_base + "." + blocknames[block] + ".dat";
    
    // All processors gather their own data
    View_Sc2 qdata = assembler->getQuadratureData(blocknames[block]); // [pts wts]
    Teuchos::Array<ScalarT> data = Teuchos::Array<ScalarT>(qdata.extent(0), 0.0);
    View_Sc2 qdata0;
    
    if (qdata.extent(0) > 0) { // Not all procs will own elements on this block
      // One by one they send to processor 0, which writes the file
      for (size_t rank=0; rank<Comm->getSize(); ++rank) {
        if (rank == 0) {
          qdata0 = qdata;
        }
        else {
          const int host = 0;
          Teuchos::Array<int> numdata(1,qdata.extent(0));
          Teuchos::Array<int> numdata0(1,0);
          if (Comm->getRank() == rank && rank>0) {
            Teuchos::send(*Comm, 1, &numdata[0], host);
          }
          if (Comm->getRank() == 0 && rank>0) {
            Teuchos::receive(*Comm, rank, 1, &numdata0[0]);
            qdata0 = View_Sc2("Data on proc 0", numdata0[0], dimension+1);
          }
          
          for (size_type k=0; k<qdata.extent(1); ++k) {
            if (Comm->getRank() == rank && rank>0) {
              for (size_type i=0; i<qdata.extent(0); ++i) {
                data[i] = qdata(k,i);
              }
              Teuchos::send(*Comm, numdata[0], &data[0], 0);
            }
            if (Comm->getRank() == 0 && rank>0) {
              Teuchos::Array<ScalarT> data0(numdata0[0], 0.0);
              Teuchos::receive(*Comm, rank, numdata[0], &data0[0]);
              for (size_type i=0; i<qdata.extent(0); ++i) {
                qdata0(k,i) = data0[i];
              }
            }
          }
        }
        if (Comm->getRank() == 0) {
          std::ofstream ptsOUT(ptsfile.c_str());
          ptsOUT.precision(12);
          for (size_type i=0; i<qdata0.extent(0); ++i) {
            for (size_type v=0; v<qdata0.extent(1)-1; ++v) {
              ptsOUT << qdata0(i, v) << "  ";
            }
            ptsOUT << endl;
          }
          ptsOUT.close();
          
          std::ofstream wtsOUT(wtsfile.c_str());
          wtsOUT.precision(12);
          for (size_type i=0; i<qdata0.extent(0); ++i) {
            wtsOUT << qdata0(i, qdata0.extent(1)-1) << endl;
          }
          wtsOUT.close();
        }
      }
      
    }
  }
}

// ========================================================================================
// ========================================================================================

template <class Node>
void PostprocessManager<Node>::writeBoundaryQuadratureData() {
  
  // Separate files for different side sets (not element blocks) - can change later if one file is better
  string ptsfile_base = settings->sublist("Postprocessing").get<string>("boundary quadrature points file", "qpts");
  string wtsfile_base = settings->sublist("Postprocessing").get<string>("boundary quadrature weights file", "qwts");
  string nsfile_base = settings->sublist("Postprocessing").get<string>("boundary quadrature normals file", "qns");
  
  for (size_t block=0; block<blocknames.size(); ++block) {
  
    for (size_t side=0; side<sideSets.size(); ++side) {
      
      std::stringstream ptsfile_ss, wtsfile_ss, nsfile_ss;
      ptsfile_ss << ptsfile_base << "." << blocknames[block] << "." << sideSets[side] << "." << Comm->getRank() << ".dat";
      wtsfile_ss << wtsfile_base << "." << blocknames[block] << "." << sideSets[side] << "." << Comm->getRank() << ".dat";
      nsfile_ss << nsfile_base << "." << blocknames[block] << "." << sideSets[side] << "." << Comm->getRank() << ".dat";
      
      string ptsfile = ptsfile_ss.str();
      string wtsfile = wtsfile_ss.str();
      string nsfile = nsfile_ss.str();
      
      // All processors gather their own data
      View_Sc2 qdata = assembler->getboundaryQuadratureData(blocknames[block], sideSets[side]); // [pts wts normals]
      Teuchos::Array<ScalarT> data = Teuchos::Array<ScalarT>(qdata.extent(0), 0.0);
      View_Sc2 qdata0;
      
      if (qdata.extent(0) > 0) { // Not all procs will own elements on this block
        // One by one they send to processor 0, which writes the file
        for (size_t rank=0; rank<Comm->getSize(); ++rank) {
          if (rank == 0) {
            qdata0 = qdata;
          }
          else {
            const int host = 0;
            Teuchos::Array<int> numdata(1,qdata.extent(0));
            Teuchos::Array<int> numdata0(1,0);
            if (Comm->getRank() == rank && rank>0) {
              Teuchos::send(*Comm, 1, &numdata[0], host);
            }
            if (Comm->getRank() == 0 && rank>0) {
              Teuchos::receive(*Comm, rank, 1, &numdata0[0]);
              qdata0 = View_Sc2("Data on proc 0", numdata0[0], 2*dimension+1); // [pts wts normals]
            }
            
            for (size_type k=0; k<qdata.extent(1); ++k) {
              if (Comm->getRank() == rank && rank>0) {
                for (size_type i=0; i<qdata.extent(0); ++i) {
                  data[i] = qdata(k,i);
                }
                Teuchos::send(*Comm, numdata[0], &data[0], 0);
              }
              if (Comm->getRank() == 0 && rank>0) {
                Teuchos::Array<ScalarT> data0(numdata0[0], 0.0);
                Teuchos::receive(*Comm, rank, numdata[0], &data0[0]);
                for (size_type i=0; i<qdata.extent(0); ++i) {
                  qdata0(k,i) = data0[i];
                }
              }
            }
          }
          //if (Comm->getRank() == 0) {
            std::ofstream ptsOUT(ptsfile.c_str());
            ptsOUT.precision(12);
            for (size_type i=0; i<qdata0.extent(0); ++i) {
              for (size_type v=0; v<dimension; ++v) {
                ptsOUT << qdata0(i, v) << "  ";
              }
              ptsOUT << endl;
            }
            ptsOUT.close();
            
            std::ofstream wtsOUT(wtsfile.c_str());
            wtsOUT.precision(12);
            for (size_type i=0; i<qdata0.extent(0); ++i) {
              wtsOUT << qdata0(i, dimension) << endl;
            }
            wtsOUT.close();
            
            std::ofstream nsOUT(nsfile.c_str());
            nsOUT.precision(12);
            for (size_type i=0; i<qdata0.extent(0); ++i) {
              for (size_type v=0; v<dimension; ++v) {
                nsOUT << qdata0(i, v+dimension+1) << "  ";
              }
              nsOUT << endl;
            }
            nsOUT.close();
            
          }
        //}
        
      }
    }
  }
}


// ========================================================================================
// ========================================================================================

template <class Node>
void PostprocessManager<Node>::setForwardStates(vector<vector<vector_RCP> > & fwd_states, vector<ScalarT> & times) {
  for (size_t set = 0; set < fwd_states.size(); ++set) {
    for (size_t t = 0; t < fwd_states[set].size(); ++t) {
      soln[set]->store(fwd_states[set][t], times[t], 0);
    }
  }
}
