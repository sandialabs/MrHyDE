/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef CELL_H
#define CELL_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "physics_base.hpp"
#include "discretizationTools.hpp"
#include "physicsInterface.hpp"
#include "workset.hpp"
#include "subgridModel.hpp"

#include <iostream>     
#include <iterator>     

static void cellHelp(const string & details) {
  cout << "********** Help and Documentation for the cells **********" << endl;
}

class cell {
public:
  
  cell() {} ;
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
  cell(const Teuchos::RCP<Teuchos::ParameterList> & settings,
       const Teuchos::RCP<LA_MpiComm> & LocalComm_, const topo_RCP & cellTopo_,
       const Teuchos::RCP<physics> & physics_RCP_, const DRV & nodes_, const size_t & myBlock_,
       const Kokkos::View<int*> & globalID_, const size_t & myLevel_, const bool & memeff_) :
  //settings(settings_),
  LocalComm(LocalComm_), cellTopo(cellTopo_),
  physics_RCP(physics_RCP_), myBlock(myBlock_), 
  globalElemID(globalID_), myLevel(myLevel_), memory_efficient(memeff_), nodes(nodes_){
  
    
    compute_diff = settings->sublist("Postprocess").get<bool>("Compute Difference in Objective", true);
    useFineScale = settings->sublist("Postprocess").get<bool>("Use fine scale sensors",true);
    loadSensorFiles = settings->sublist("Analysis").get<bool>("Load Sensor Files",false);
    writeSensorFiles = settings->sublist("Analysis").get<bool>("Write Sensor Files",false);
    mortar_objective = settings->sublist("Solver").get<bool>("Use Mortar Objective",false);
    
    active = true;
    multiscale = false;
    numElem = nodes_.dimension(0);
    numnodes = nodes_.dimension(1);
    dimension = nodes_.dimension(2);
    //nodes = DRV("nodes", 1,numnodes,dimension);
    //for (int j=0; j<numnodes; j++) {
    //  for (int k=0; k<dimension; k++) {
    //    nodes(0,j,k) = nodes_(j,k);
    //  }
    //}

    if (dimension == 1) {
      shape = "interval";
    }
    if (dimension == 2) {
      if (numnodes == 3) {
        shape = "tri";
      }
      else if (numnodes == 4) {
        shape = "quad";
      }
      else {
        shape = "unknown";
      }
    }
    if (dimension == 3) {
      if (numnodes == 4) {
        shape = "tet";
      }
      else if (numnodes == 8) {
        shape = "hex";
      }
      else {
        shape = "unknown";
      }
    }
    numSides = cellTopo->getSideCount();
    response_type = "global";
    useSensors = false;
    
    have_cell_phi = false;
    have_cell_rotation = false;
    
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void setIP(const DRV & ref_ip) {
    // ip and ref_ip will live on the assembly device
    ip = DRV("ip", numElem, ref_ip.dimension(0), dimension);
    CellTools<AssemblyDevice>::mapToPhysicalFrame(ip, ref_ip, nodes, *cellTopo);
    
    ijac = DRV("ijac", numElem, ref_ip.dimension(0), dimension, dimension);
    CellTools<AssemblyDevice>::setJacobian(ijac, ref_ip, nodes, *cellTopo);
    
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void setSideIP(const DRV & ref_side_ip, const DRV & ref_side_wts) {
    // side ip and ref_side_ip will live on the assembly device
    for (size_t s=0; s<numSides; s++) {
      bool compute = false; // may not need the integration info on this side
      for (size_t e=0; e<sideinfo.dimension(0); e++) { //numElem
        for (size_t n=0; n<sideinfo.dimension(1); n++) { //numVars
          //if (sideinfo(e,n,s,0)>0) {
            compute = true;
          //}
        }
      }
      DRV sip, sijac, wts_side, cnormals;
      //DRV sijac("sijac", numElem, ref_side_ip.dimension(0), dimension, dimension);
      //DRV wts_side("wts_side", numElem, ref_side_ip.dimension(0));
      //DRV cnormals = DRV("normals", numElem, ref_side_ip.dimension(0),dimension);
      
      if (compute) {
        
        sip = DRV("sip", numElem, ref_side_ip.dimension(0), dimension);
        sijac = DRV("sijac", numElem, ref_side_ip.dimension(0), dimension, dimension);
        wts_side = DRV("wts_side", numElem, ref_side_ip.dimension(0));
        cnormals = DRV("normals", numElem, ref_side_ip.dimension(0),dimension);
        
        DRV refSidePoints("refSidePoints", ref_side_ip.dimension(0), dimension);
        CellTools<AssemblyDevice>::mapToReferenceSubcell(refSidePoints, ref_side_ip, dimension-1, s, *cellTopo);
        
        CellTools<AssemblyDevice>::mapToPhysicalFrame(sip, refSidePoints, nodes, *cellTopo);
        
        CellTools<AssemblyDevice>::setJacobian(sijac, refSidePoints, nodes, *cellTopo);
        
        DRV sijacInv("sidejacobInv",numElem, ref_side_ip.dimension(0), dimension, dimension);
        
        CellTools<AssemblyDevice>::setJacobianInv(sijacInv, sijac);
        
        DRV temporary_buffer("temporary_buffer",numElem,ref_side_ip.dimension(0)*dimension*dimension);
        
        if (dimension == 2) {
          FunctionSpaceTools<AssemblyDevice>::computeEdgeMeasure(wts_side, sijac, ref_side_wts, s, *cellTopo, temporary_buffer);
        }
        if (dimension == 3) {
          FunctionSpaceTools<AssemblyDevice>::computeFaceMeasure(wts_side, sijac, ref_side_wts, s, *cellTopo, temporary_buffer);
        }
        CellTools<AssemblyDevice>::getPhysicalSideNormals(cnormals, sijac, s, *cellTopo);
        
        // scale the normal vector (we need unit normal...)
        
        for (int e=0; e<numElem; e++) {
          for( int j=0; j<ref_side_ip.dimension(0); j++ ) {
            ScalarT normalLength = 0.0;
            for (int sd=0; sd<dimension; sd++) {
              normalLength += cnormals(e,j,sd)*cnormals(e,j,sd);
            }
            normalLength = sqrt(normalLength);
            for (int sd=0; sd<dimension; sd++) {
              cnormals(e,j,sd) = cnormals(e,j,sd) / normalLength;
            }
          }
        }
      }
      sideip.push_back(sip);
      sideijac.push_back(sijac);
      normals.push_back(cnormals);
      sidewts.push_back(wts_side);
    }
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
  //void setIndex(const vector<vector<vector<int> > > & index_) {
  void setIndex(Kokkos::View<LO***,AssemblyDevice> & index_, Kokkos::View<LO*,AssemblyDevice> & numDOF_) {
    
    index = Kokkos::View<LO***,AssemblyDevice>("local index",index_.dimension(0),
                                               index_.dimension(1), index_.dimension(2));
    
    // Need to copy the data since index_ is rewritten for each cell
    parallel_for(RangePolicy<AssemblyDevice>(0,index_.dimension(0)), KOKKOS_LAMBDA (const int e ) {
      for (int j=0; j<index_.dimension(1); j++) {
        for (int k=0; k<index_.dimension(2); k++) {
          index(e,j,k) = index_(e,j,k);
        }
      }
    });
    
    // This is common to all cells (within the same block), so a view copy will do
    numDOF = numDOF_;
    
    /*
    for (int i=0; i<index.dimension(1); i++) {
      numDOF_host(i) = index[0][i].size();
    }
    
    Kokkos::View<int*,HostDevice> numDOF_host("numDOF on host",index.dimension(1));
    for (int i=0; i<index.dimension(1); i++) {
      numDOF_host(i) = numDOF_(i));
    }
    numDOF = Kokkos::create_mirror_view(numDOF_host);
    Kokkos::deep_copy(numDOF_host, numDOF);
    */
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
  //void setParamIndex(const vector<vector<vector<int> > > & pindex_) {
  void setParamIndex(Kokkos::View<LO***,AssemblyDevice> & pindex_,
                     Kokkos::View<LO*,AssemblyDevice> & pnumDOF_) {
    
    paramindex = Kokkos::View<LO***,AssemblyDevice>("local param index",pindex_.dimension(0),
                                                    pindex_.dimension(1), pindex_.dimension(2));
    
    // Need to copy the data since index_ is rewritten for each cell
    parallel_for(RangePolicy<AssemblyDevice>(0,pindex_.dimension(0)), KOKKOS_LAMBDA (const int e ) {
      for (int j=0; j<pindex_.dimension(1); j++) {
        for (int k=0; k<pindex_.dimension(2); k++) {
          paramindex(e,j,k) = pindex_(e,j,k);
        }
      }
    });
    
    // This is common to all cells, so a view copy will do
    // This is excessive storage, please remove
    numParamDOF = pnumDOF_;
    
    
    //paramindex = pindex_;
    
    /*
    Kokkos::View<int*,HostDevice> numParamDOF_host("numParamDOF on host",paramindex[0].size());
    for (int i=0; i<paramindex[0].size(); i++) {
      numParamDOF_host(i) = paramindex[0][i].size();
      cout << "npdof(i) = " << numParamDOF_host(i) << endl;
      
    }
    numParamDOF = Kokkos::create_mirror_view(numParamDOF_host);
    Kokkos::deep_copy(numParamDOF_host, numParamDOF);
    */
    
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
  //void setAuxIndex(const vector<vector<int> > & aindex_) {
  void setAuxIndex(Kokkos::View<LO***,AssemblyDevice> & aindex_) { //}, Kokkos::View<LO*,AssemblyDevice> & anumDOF_) {
    
    auxindex = Kokkos::View<LO***,AssemblyDevice>("local aux index",1,aindex_.dimension(1),
                                                  aindex_.dimension(2));
    
    // Need to copy the data since index_ is rewritten for each cell
    parallel_for(RangePolicy<AssemblyDevice>(0,aindex_.dimension(0)), KOKKOS_LAMBDA (const int e ) {
      for (int j=0; j<aindex_.dimension(1); j++) {
        for (int k=0; k<aindex_.dimension(2); k++) {
          auxindex(e,j,k) = aindex_(e,j,k);
        }
      }
    });
    
    // This is common to all cells, so a view copy will do
    // This is excessive storage, please remove
    //numAuxDOF = anumDOF_;
    // Temp. fix
    numAuxDOF = Kokkos::View<int*,HostDevice>("numAuxDOF",auxindex.dimension(1));
    for (int i=0; i<auxindex.dimension(1); i++) {
      numAuxDOF(i) = auxindex.dimension(2);
    }
    
    
    /*
    auxindex = aindex_;
    
    Kokkos::View<int*,HostDevice> numAuxDOF_host("numAuxDOF on host",auxindex.size());
    for (int i=0; i<auxindex.size(); i++) {
      numAuxDOF_host(i) = auxindex[i].size();
    }
    numAuxDOF = Kokkos::create_mirror_view(numAuxDOF_host);
    Kokkos::deep_copy(numAuxDOF_host, numAuxDOF);
    */
    
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Add the aux basis functions at the integration points.
  // This version assumes the basis functions have been evaluated elsewhere (as in multiscale)
  ///////////////////////////////////////////////////////////////////////////////////////

  void addAuxDiscretization(const vector<basis_RCP> & abasis_pointers, const vector<DRV> & abasis,
                            const vector<DRV> & abasisGrad, const vector<vector<DRV> > & asideBasis, 
                            const vector<vector<DRV> > & asideBasisGrad);

  ///////////////////////////////////////////////////////////////////////////////////////
  // Update the regular parameters (everything but discretized)
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void updateParameters(vector<Teuchos::RCP<vector<AD> > > & params, const vector<string> & paramnames);
                        
  ///////////////////////////////////////////////////////////////////////////////////////
  // Add the aux variables
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void addAuxVars(const vector<string> & auxlist_);
    
  ///////////////////////////////////////////////////////////////////////////////////////
  // Define which basis each variable will use
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void setUseBasis(vector<int> & usebasis_, const int & nstages_);
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Define which basis each discretized parameter will use
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void setParamUseBasis(vector<int> & pusebasis_, vector<int> & paramnumbasis_);
    
  ///////////////////////////////////////////////////////////////////////////////////////
  // Define which basis each aux variable will use
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void setAuxUseBasis(vector<int> & ausebasis_);
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Set the local solutions
  ///////////////////////////////////////////////////////////////////////////////////////
  /*
  void setLocalSolns(const vector_RCP & gl_u, const vector_RCP & gl_u_dot,
                     const vector_RCP & gl_phi, const vector_RCP & gl_phi_dot,
                     const vector_RCP & gl_param, //const vector_RCP & gl_aux,
                     const bool & isAdjoint);
  */
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Set the local solutions
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void setLocalSoln(const Teuchos::RCP<Epetra_MultiVector> & gl_u, const int & type,
                    const size_t & entry);

  void setLocalSoln(const vector_RCP & gl_u, const int & type,
                    const size_t & entry);

  ///////////////////////////////////////////////////////////////////////////////////////
  // Map the coarse grid solution to the fine grid integration points
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void computeSolnVolIP(const bool & seedu, const bool & seedudot, const bool & seedparams,
                        const bool & seedaux);
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Map the coarse grid solution to the fine grid integration points
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void computeSolnSideIP(const int & side,
                         const bool & seedu, const bool & seedudot, const bool & seedparams,
                         const bool & seedaux);
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Compute the contribution from this cell to the global res, J, Jdot
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void computeJacRes(const ScalarT & time, const bool & isTransient, const bool & isAdjoint,
                     const bool & compute_jacobian, const bool & compute_sens,
                     const int & num_active_params, const bool & compute_disc_sens,
                     const bool & compute_aux_sens, const bool & store_adjPrev,
                     Kokkos::View<ScalarT***,AssemblyDevice> res,
                     Kokkos::View<ScalarT***,AssemblyDevice> local_J,
                     Kokkos::View<ScalarT***,AssemblyDevice> local_Jdot);
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Update the solution variables in the workset
  ///////////////////////////////////////////////////////////////////////////////////////
 
  void updateSolnWorkset(const Teuchos::RCP<Epetra_MultiVector> & gl_u, const int tindex);
  
  void updateSolnWorkset(const vector_RCP & gl_u, const int tindex);
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Use the AD res to update the scalarT res
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void updateRes(const bool & compute_sens, Kokkos::View<ScalarT***,AssemblyDevice> local_res);

  ///////////////////////////////////////////////////////////////////////////////////////
  // Update the adjoint res
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void updateAdjointRes(const bool & compute_sens, Kokkos::View<ScalarT***,AssemblyDevice> local_res);
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Use the AD res to update the scalarT J
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void updateJac(const bool & useadjoint, Kokkos::View<ScalarT***,AssemblyDevice> local_J);
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Use the AD res to update the scalarT Jdot
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void updateJacDot(const bool & useadjoint, Kokkos::View<ScalarT***,AssemblyDevice> local_Jdot);

  ///////////////////////////////////////////////////////////////////////////////////////
  // Use the AD res to update the scalarT Jparam
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void updateParamJac(Kokkos::View<ScalarT***,AssemblyDevice> local_J);
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Use the AD res to update the scalarT Jdot
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void updateParamJacDot(Kokkos::View<ScalarT***,AssemblyDevice> local_Jdot);
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Use the AD res to update the scalarT Jaux
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void updateAuxJac(Kokkos::View<ScalarT***,AssemblyDevice> local_J);
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Use the AD res to update the scalarT Jdot
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void updateAuxJacDot(Kokkos::View<ScalarT***,AssemblyDevice> local_Jdot);
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Get the initial condition 
  ///////////////////////////////////////////////////////////////////////////////////////
  
  Kokkos::View<ScalarT**,AssemblyDevice> getInitial(const bool & project, const bool & isAdjoint);

  ///////////////////////////////////////////////////////////////////////////////////////
  // Get the mass matrix 
  ///////////////////////////////////////////////////////////////////////////////////////
    
  Kokkos::View<ScalarT***,AssemblyDevice> getMass();
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Compute the error at the integration points given the solution and solve times
  ///////////////////////////////////////////////////////////////////////////////////////

  Kokkos::View<ScalarT**,AssemblyDevice> computeError(const ScalarT & solvetime, const size_t & tindex,
                                                     const bool compute_subgrid, const string & error_type);

  ///////////////////////////////////////////////////////////////////////////////////////
  // Compute the response at a given set of points and time
  ///////////////////////////////////////////////////////////////////////////////////////

  Kokkos::View<AD***,AssemblyDevice> computeResponseAtNodes(const DRV & nodes,
                                                            const int tindex,
                                                            const ScalarT & time);

  ///////////////////////////////////////////////////////////////////////////////////////
  // Compute the response at the integration points given the solution and solve times
  ///////////////////////////////////////////////////////////////////////////////////////
  
  Kokkos::View<AD***,AssemblyDevice> computeResponse(const ScalarT & solvetime,
                                                     const size_t & tindex,
                                                     const int & seedwhat);
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Compute volumetric contribution to the regularization
  ///////////////////////////////////////////////////////////////////////////////////////

  AD computeDomainRegularization(const vector<ScalarT> reg_constants, const vector<int> reg_types,
                                 const vector<int> reg_indices);

  ///////////////////////////////////////////////////////////////////////////////////////
  // Compute boundary contribution to the regularization and nodes located on the boundary
  ///////////////////////////////////////////////////////////////////////////////////////
  
  AD computeBoundaryRegularization(const vector<ScalarT> reg_constants, const vector<int> reg_types,
                                   const vector<int> reg_indices, const vector<string> reg_sides);

  ///////////////////////////////////////////////////////////////////////////////////////
  // Compute the objective function given the solution and solve times
  ///////////////////////////////////////////////////////////////////////////////////////
  
  Kokkos::View<AD**,AssemblyDevice> computeObjective(const ScalarT & solvetime,
                                                     const size_t & tindex,
                                                     const int & seedwhat);
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Compute the target function given the solve times
  ///////////////////////////////////////////////////////////////////////////////////////

  Kokkos::View<AD***,AssemblyDevice> computeTarget(const ScalarT & solvetime);

  ///////////////////////////////////////////////////////////////////////////////////////
  // Compute the weight functino given the solve times
  ///////////////////////////////////////////////////////////////////////////////////////

  Kokkos::View<AD***,AssemblyDevice> computeWeight(const ScalarT & solvetime);

  ///////////////////////////////////////////////////////////////////////////////////////
  // Add sensor information
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void addSensors(const Kokkos::View<ScalarT**,HostDevice> sensor_points, const ScalarT & sensor_loc_tol,
                  const vector<Kokkos::View<ScalarT**,HostDevice> > & sensor_data, const bool & have_sensor_data,
                  const vector<basis_RCP> & basis_pointers,
                  const vector<basis_RCP> & param_basis_pointers);
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Subgrid Plotting
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void writeSubgridSolution(const std::string & filename);
    
  ///////////////////////////////////////////////////////////////////////////////////////
  // Subgrid Plotting
  ///////////////////////////////////////////////////////////////////////////////////////

  void writeSubgridSolution(Teuchos::RCP<panzer_stk::STK_Interface> & globalmesh,
                            string & subblockname, bool & isTD, int & offset);

  ///////////////////////////////////////////////////////////////////////////////////////
  // Compute flux and sensitivity wrt params
  ///////////////////////////////////////////////////////////////////////////////////////

  template<class T>
  void computeFlux(const Teuchos::RCP<T> & u, const Teuchos::RCP<T> & du,
                   const Teuchos::RCP<T> & sub_param,
                   Kokkos::View<ScalarT***,AssemblyDevice> lambda,
                   const ScalarT & time, const int & s, const ScalarT & coarse_h,
                   const bool & compute_sens);

  ///////////////////////////////////////////////////////////////////////////////////////
  // Re-seed the global parameters 
  ///////////////////////////////////////////////////////////////////////////////////////
  /*
  void sacadoizeParams(const bool & seed_active, const int & num_active_params,
                       const vector<int> & paramtypes, const vector<string> & paramnames,
                       const vector<vector<ScalarT> > & paramvals);
    
  */
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Get the discretization/physics info (used for workset construction)
  ///////////////////////////////////////////////////////////////////////////////////////

  vector<int> getInfo() {
    vector<int> info;
    int nparams = 0;
    if (paramindex.dimension(0)>0) {
      nparams = paramindex.dimension(1);
    }
    info.push_back(dimension);
    info.push_back(numDOF.dimension(0));
    info.push_back(nparams);
    info.push_back(auxindex.dimension(1));
    info.push_back(GIDs.dimension(1));
    info.push_back(numElem);
    return info;
  }

  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////

  void setUpAdjointPrev(const int & numDOF) {
    adjPrev = Kokkos::View<ScalarT**,AssemblyDevice>("previous adjoint",numElem,numDOF);
  }

  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////

  void setUpSubGradient(const int & numParams) {
    subgradient = Kokkos::View<ScalarT**,AssemblyDevice>("subgrid gradient",numElem,numParams);
  }

  ///////////////////////////////////////////////////////////////////////////////////////
  // Get the subgrid timers
  ///////////////////////////////////////////////////////////////////////////////////////

  /*
  vector<Teuchos::RCP<Teuchos::Time> > getSubgridTimers() {
    vector<Teuchos::RCP<Teuchos::Time> > subtimers;
    vector<Teuchos::RCP<Teuchos::Time> > subwksettimers;
    if (multiscale) {
      subtimers = subgridModel->timers;
      subwksettimers = subgridModel->wkset[0]->timers;
      for (size_t i=0; i<subwksettimers.size(); i++) {
        subtimers.push_back(subwksettimers[i]);
      }
    }
    return subtimers;
  }
  */
   
  ///////////////////////////////////////////////////////////////////////////////////////
  // Update the subgrid model
  ///////////////////////////////////////////////////////////////////////////////////////

  void updateSubgridModel(vector<Teuchos::RCP<SubGridModel> > & models);
  
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Pass cell data to wkset
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void updateData();
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////

  void resetAdjPrev(const ScalarT & val);

  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////

  // Public and necessary
  //Teuchos::RCP<Teuchos::ParameterList> settings;
  Teuchos::RCP<LA_MpiComm> LocalComm;
  bool active, memory_efficient;
  size_t myBlock, myLevel;
  Kokkos::View<int*> globalElemID;
  //size_t globalElemID;
  Teuchos::RCP<physics> physics_RCP;
  
  // Geometry Information
  int numnodes, numSides, dimension, numElem;
  string shape;
  DRV nodes;
  DRV ip, ijac;
  vector<DRV> sideip, sideijac, normals, sidewts;
  
  DRV nodepert; // perturbation of the nodes from their original location
  topo_RCP cellTopo;
  Kokkos::View<int****,HostDevice> sideinfo; // may need to move this to Assembly
  vector<string> sidenames;
  
  Kokkos::View<GO**,HostDevice> GIDs, paramGIDs, auxGIDs;
  Kokkos::View<LO***,AssemblyDevice> index, paramindex, auxindex;
  
  //vector<vector<int> > GIDs;
  //vector<vector<vector<int> > > index;
  vector<vector<ScalarT> > orientation;
  Kokkos::View<int*,AssemblyDevice> numDOF, numParamDOF, numAuxDOF;
  
  Kokkos::View<ScalarT***,AssemblyDevice> u, u_dot, phi, phi_dot, aux;
  ScalarT current_time;
  int num_stages;
  
  // Discretized Parameter Information
  Kokkos::View<ScalarT***,AssemblyDevice> param;
  //vector<vector<int> > paramGIDs;
  //vector<vector<vector<int> > > paramindex;
  
  // Auxiliary Parameter Information
  // aux variables are handled slightly differently from others
  
  vector<string> auxlist;
  //vector<int> auxGIDs;
  vector<vector<int> > auxoffsets;
  vector<int> auxusebasis;
  //vector<vector<int> > auxindex;
  vector<basis_RCP> auxbasisPointers;
  vector<DRV> auxbasis;
  vector<DRV> auxbasisGrad;
  vector<vector<DRV> > auxside_basis;
  vector<vector<DRV> > auxside_basisGrad;
  
  // Sensor information
  bool useSensors;
  string response_type;
  size_t numSensors;
  vector<Kokkos::View<ScalarT**,HostDevice> > sensorLocations;
  DRV sensorPoints;
  vector<int> sensorElem;
  vector<Kokkos::View<ScalarT**,HostDevice> > sensorData;
  vector<vector<DRV> > sensorBasis, param_sensorBasis;
  vector<vector<DRV> > sensorBasisGrad, param_sensorBasisGrad;
  vector<int> mySensorIDs;
  Kokkos::View<ScalarT**,AssemblyDevice> adjPrev, subgradient;
  Kokkos::View<ScalarT**,AssemblyDevice> cell_data;
  vector<ScalarT> cell_data_distance;
  bool compute_diff, useFineScale, loadSensorFiles, writeSensorFiles;
  bool mortar_objective;
  bool exodus_sensors = false;
  
  // Profile timers
  Teuchos::RCP<Teuchos::Time> computeSolnVolTimer = Teuchos::TimeMonitor::getNewCounter("MILO::cell::computeSolnVolIP()");
  Teuchos::RCP<Teuchos::Time> computeSolnSideTimer = Teuchos::TimeMonitor::getNewCounter("MILO::cell::computeSolnSideIP()");
  Teuchos::RCP<Teuchos::Time> volumeResidualTimer = Teuchos::TimeMonitor::getNewCounter("MILO::cell::computeJacRes() - volume residual");
  Teuchos::RCP<Teuchos::Time> boundaryResidualTimer = Teuchos::TimeMonitor::getNewCounter("MILO::cell::computeJacRes() - boundary residual");
  Teuchos::RCP<Teuchos::Time> jacobianFillTimer = Teuchos::TimeMonitor::getNewCounter("MILO::cell::computeJacRes() - fill local Jacobian");
  Teuchos::RCP<Teuchos::Time> residualFillTimer = Teuchos::TimeMonitor::getNewCounter("MILO::cell::computeJacRes() - fill local residual");
  Teuchos::RCP<Teuchos::Time> transientResidualTimer = Teuchos::TimeMonitor::getNewCounter("MILO::cell::computeJacRes() - transient residual");
  Teuchos::RCP<Teuchos::Time> adjointResidualTimer = Teuchos::TimeMonitor::getNewCounter("MILO::cell::computeJacRes() - adjoint residual");
  Teuchos::RCP<Teuchos::Time> cellFluxGatherTimer = Teuchos::TimeMonitor::getNewCounter("MILO::cell::computeFlux - gather solution");
  Teuchos::RCP<Teuchos::Time> cellFluxWksetTimer = Teuchos::TimeMonitor::getNewCounter("MILO::cell::computeFlux - update wkset");
  Teuchos::RCP<Teuchos::Time> cellFluxAuxTimer = Teuchos::TimeMonitor::getNewCounter("MILO::cell::computeFlux - compute aux solution");
  Teuchos::RCP<Teuchos::Time> cellFluxEvalTimer = Teuchos::TimeMonitor::getNewCounter("MILO::cell::computeFlux - physics evaluation");
  
  vector<Teuchos::RCP<SubGridModel> > subgridModels;
  bool multiscale, have_cell_phi, have_cell_rotation;
  vector<size_t> subgrid_usernum, cell_data_seed, cell_data_seedindex;
  vector<vector<size_t> > subgrid_model_index;
  
  Teuchos::RCP<workset> wkset;
  
};

#endif
