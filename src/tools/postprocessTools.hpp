/***********************************************************************
 This is a framework for solving Multi-resolution Hybridized
 Differential Equations (MrHyDE)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
 ************************************************************************/

#ifndef MRHYDE_POSTPROCESS_TOOLS_H
#define MRHYDE_POSTPROCESS_TOOLS_H

#include "trilinos.hpp"
#include "preferences.hpp"

namespace MrHyDE {
  
  
  // ========================================================================================
  // Class for storing a regularization function settings
  // ========================================================================================
  
  class regularization {
  public:
    
    regularization() {};
    
    ~regularization() {};
    
    regularization(Teuchos::ParameterList & regsettings, const string name_,
                   const size_t & block_) {
      name = name_;
      block = block_;
      
      type = regsettings.get<string>("type","integrated");
      function = regsettings.get<string>("function","0.0");
      location = regsettings.get<string>("location","volume");
      boundary_name = regsettings.get<string>("boundary name","");
      weight = regsettings.get<double>("weight",1.0);
      objective_name = regsettings.get<string>("objective name","");

    }
    
    string type, name, location, function, boundary_name, objective_name;
    ScalarT weight;
    size_t block;
  };
  
  // ========================================================================================
  // Class for storing an objective function settings
  // ========================================================================================
  
  class objective {
  public:
    
    objective() {};
    
    ~objective() {};
    
    objective(Teuchos::ParameterList & objsettings, const string name_,
              const size_t & block_) { //, Teuchos::RCP<FunctionManager<AD> > & functionManager_) {
      name = name_;
      block = block_;
      type = objsettings.get<string>("type","none");
      weight = objsettings.get<double>("weight",1.0);
      
      sensor_points_file = "";
      sensor_data_file = "";
      save_data = false;
      response = "";
      target = 0.0;
      function = "";
      use_sensor_grid = false;
      output_type = "";

      if (type == "sensors") {
        sensor_points_file = objsettings.get<string>("sensor points file","sensor_points.dat");
        sensor_data_file = objsettings.get<string>("sensor data file","");
        save_data = objsettings.get<bool>("save sensor data",false);
        use_sensor_grid = objsettings.get<bool>("use sensor grid",false);
        if (use_sensor_grid) {
          sensor_grid_Nx = objsettings.get<int>("grid Nx");
          sensor_grid_Ny = objsettings.get<int>("grid Ny");
          sensor_grid_Nz = objsettings.get<int>("grid Nz");

          sensor_grid_xmin = objsettings.get<double>("grid xmin");
          sensor_grid_xmax = objsettings.get<double>("grid xmax");
          sensor_grid_ymin = objsettings.get<double>("grid ymin");
          sensor_grid_ymax = objsettings.get<double>("grid ymax");
          sensor_grid_zmin = objsettings.get<double>("grid zmin");
          sensor_grid_zmax = objsettings.get<double>("grid zmax");
        }
        response = objsettings.get<string>("response","0.0");
        //functionManager_->addFunction(name+" response",response,"point");
        response_file = objsettings.get<string>("response file","sensor."+name);
        compute_sensor_soln = objsettings.get<bool>("compute sensor solution",false);
        compute_sensor_average_soln = objsettings.get<bool>("compute sensor average solution",false);
        output_type = objsettings.get<string>("output type",""); // "fft" for fft output
        dft_current = 0;
        if (output_type == "dft") {
          // logic and setup for a subset of frequencies ...
          dft_num_freqs = objsettings.get<int>("number of dft frequencies");
        }
        if (compute_sensor_soln && compute_sensor_average_soln) {
          // throw an error
        }      
      }
      else if (type == "integrated response") {
        response = objsettings.get<string>("response","0.0");
        target = objsettings.get<double>("target",0.0);
        //functionManager_->addFunction(name+" response",response,"ip");
        save_data = objsettings.get<bool>("save response data",false);
        response_file = objsettings.get<string>("response file","response."+name);
      }
      else if (type == "integrated control") {
        function = objsettings.get<string>("function","0.0");
        //functionManager_->addFunction(name,function,"ip");
      }
      else if (type == "discrete control") {
        // nothing else is needed
      }
    
      if (objsettings.isSublist("Regularization functions")) {
        Teuchos::ParameterList reg_funs = objsettings.sublist("Regularization functions");
        Teuchos::ParameterList::ConstIterator reg_itr = reg_funs.begin();
        while (reg_itr != reg_funs.end()) {
          Teuchos::ParameterList regsettings = reg_funs.sublist(reg_itr->first);
          regularization newreg(regsettings,reg_itr->first,block);//,functionManager_);
          regularizations.push_back(newreg);
          reg_itr++;
        }
      }
    }
    
    size_t block;
    string name, type, location, response, function, boundary_name, response_file;
    ScalarT weight, target;
    bool save_data;
    vector<regularization> regularizations;
    vector<ScalarT> response_times;
    vector<ScalarT> scalar_response_data; // [time] or [realization]
    vector<Kokkos::View<ScalarT*,HostDevice> > response_data; // [time](sensor) or [realization](sensor)
    
    // Data specific to sensors
    string sensor_points_file, sensor_data_file, output_type;
    size_t numSensors;
    bool use_sensor_grid, compute_sensor_soln, compute_sensor_average_soln;
    int sensor_grid_Nx, sensor_grid_Ny, sensor_grid_Nz, dft_num_freqs;
    double sensor_grid_xmin, sensor_grid_xmax, sensor_grid_ymin, sensor_grid_ymax, sensor_grid_zmin, sensor_grid_zmax;
    Kokkos::View<ScalarT**,AssemblyDevice> sensor_data;   // Ns x Nt
    Kokkos::View<ScalarT**,AssemblyDevice> sensor_points; // Ns x dim
    Kokkos::View<ScalarT*,AssemblyDevice>  sensor_times;  // Nt
    Kokkos::View<int*[2],HostDevice>       sensor_owners; // Ns x (group elem)
    Kokkos::View<bool*,HostDevice>         sensor_found;
    
    vector<Kokkos::View<ScalarT****,AssemblyDevice> > sensor_basis;       //[basis](Ns,dof,pt,dim)
    vector<Kokkos::View<ScalarT****,AssemblyDevice> > sensor_basis_grad;  // [basis](Ns,dof,pt,dim)
    //vector<vector<Kokkos::View<ScalarT***,AssemblyDevice> > >  sensor_basis_div;   // [Ns][basis](elem,dof,pt)
    //vector<vector<Kokkos::View<ScalarT****,AssemblyDevice> > > sensor_basis_curl;  // [Ns][basis](elem,dof,pt,dim)
    vector<Kokkos::View<ScalarT***,HostDevice> > sensor_solution_data; // [time] (sensor,sol,dim)
    Kokkos::View<std::complex<double>****,HostDevice> sensor_solution_dft; // (sensor,sol,dim,freq)
    int dft_current;
  };
  
  // ========================================================================================
  // Class for storing a flux response (not for optimization)
  // ========================================================================================
  
  class fluxResponse {
  public:
    
    fluxResponse() {};
    
    ~fluxResponse() {};
    
    fluxResponse(Teuchos::ParameterList & frsettings, const string & name_,
                 const size_t & block_ ) { //, Teuchos::RCP<FunctionManager<AD> > & functionManager_) {
      name = name_;
      block = block_;
      
      sidesets = frsettings.get<string>("side sets","all");
      weight = frsettings.get<string>("weight","1.0");
      int numfluxes = frsettings.get<int>("number",1);
      
      vals = Kokkos::View<ScalarT*,HostDevice>("flux data",numfluxes);
      
      //functionManager_->addFunction("flux weight "+name,weight,"side ip");
      
    }
    
    string name, sidesets, weight;
    size_t block;
    Kokkos::View<ScalarT*,HostDevice> vals;
  };
  
  
  // ========================================================================================
  // Class for storing an integrated quantity
  // ========================================================================================

  /** integratedQuantity class
   * 
   * Holds the information necessary to compute an integrated quantity along with
   * its value. This is not for optimization.
   *
   */
  
  // TODO -- BWR could potentially make this a parent class and have the others inherit.
  class integratedQuantity {
  public:
    /**
     * @brief Construct storage and information for an integrated quantity requested
     * in the input file
     *
     * @param[in]  iqsettings  Parameter list with the settings from the input file
     * @param[in]  name_  Name for the quantity
     * @param[in]  block_  Mesh block on which to compute
     * @param[in]  functionManager_  The function manager used to store the integrand 
     *
     */
    
    integratedQuantity() {};
    
    ~integratedQuantity() {};
    
    integratedQuantity(Teuchos::ParameterList & iqsettings, const string & name_,
                 const size_t & block_) { //, Teuchos::RCP<FunctionManager<AD> > & functionManager_) {
      name = name_;
      block = block_;
      
      // We assume a volume integral by default
      location = iqsettings.get<string>("location","volume"); 
      // Only used in case of boundary integral
      boundarynames = iqsettings.get<string>("boundary names","all");
      
      val = Kokkos::View<ScalarT*,HostDevice>("integrated quantity data",1);
     
      integrand = iqsettings.get<string>("integrand","0.0");
      
      // Integrand is kept at the appropriate integration points
      //if (location == "volume") {
      //  functionManager_->addFunction(name+" integrand",integrand,"ip"); 
      //} else if (location == "boundary") {
      //  functionManager_->addFunction(name+" integrand",integrand,"side ip");
      //}

    }
    
    /**
     * @brief Construct storage and information for an integrated quantity requested
     * from the physics module
     *
     * @param[in]  integrand  The integrand to be added to the function manager
     * @param[in]  name_  Name for the quantity
     * @param[in]  integralType  The type of integral (boundary or volume)
     * @param[in]  block_  Mesh block on which to compute
     * @param[in]  functionManager_  The function manager used to store the integrand 
     *
     */
    
    integratedQuantity(const string & integrand_, const string & name_, const string & integralType, 
                 const size_t & block_) { //, Teuchos::RCP<FunctionManager<AD> > & functionManager_) {
      integrand = integrand_;
      name = name_;
      block = block_;
      location = integralType; // for consistency with above

      // Only used in case of boundary integral
      boundarynames = "all"; // TODO this could be expanded in the future (again for consistency)
      
      val = Kokkos::View<ScalarT*,HostDevice>("integrated quantity data",1);
     
      // Integrand is kept at the appropriate integration points
      //if (location == "volume") {
      //  functionManager_->addFunction(name+" integrand",integrand,"ip"); 
      //} else if (location == "boundary") {
      //  functionManager_->addFunction(name+" integrand",integrand,"side ip");
      //} else {
      //  // TODO add error message
      //}

    }

    string name, boundarynames, integrand, location;
    size_t block;
    Kokkos::View<ScalarT*,HostDevice> val;
  };

  // ========================================================================================
  // ========================================================================================
  
  
}

#endif
