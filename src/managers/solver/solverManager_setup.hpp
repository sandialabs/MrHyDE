/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

// ========================================================================================
// ========================================================================================

template<class Node>
void SolverManager<Node>::completeSetup() {

  debugger->print("**** Starting SolverManager::completeSetup()");
  
  /////////////////////////////////////////////////////////////////////////////
  // Create linear algebra interface
  /////////////////////////////////////////////////////////////////////////////
  
  linalg = Teuchos::rcp( new LinearAlgebraInterface<Node>(Comm, settings, disc, params) );
  
  if (store_vectors) {
    for (size_t set=0; set<setnames.size(); ++set) {
      res.push_back(linalg->getNewVector(set));
      res_over.push_back(linalg->getNewOverlappedVector(set));
      du_over.push_back(linalg->getNewOverlappedVector(set));
      du.push_back(linalg->getNewVector(set));
    }
  }
  this->setupFixedDOFs(settings);

  //---------------------------------------------------
  // Mass matrix (lumped and maybe full) for explicit
  //---------------------------------------------------
  
  if (fully_explicit) {
    this->setupExplicitMass();
  }
  
  if (use_param_mass && params->num_discretized_params > 0) {
    this->setupDiscretizedParamMass();
  }
  
  debugger->print("**** Finished SolverManager::completeSetup()");
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
void SolverManager<Node>::setupExplicitMass() {

  debugger->print("**** Starting SolverManager::setupExplicitMass()");
  
  bool compute_matrix = true;
  if (assembler->lump_mass || assembler->matrix_free) {
    compute_matrix = false;
  }
  
  for (size_t set=0; set<useBasis.size(); ++set) {
    matrix_RCP mass;
    
    assembler->updatePhysicsSet(set);
    if (compute_matrix) {
      explicitMass.push_back(linalg->getNewMatrix(set));
      if (linalg->getHaveOverlapped()) {
        mass = linalg->getNewOverlappedMatrix(set);
      }
      else {
        mass = explicitMass[set];
      }
    }
    
    diagMass.push_back(linalg->getNewVector(set));
    vector_RCP diagMass_over;
    if (linalg->getHaveOverlapped()) {
      diagMass_over = linalg->getNewOverlappedVector(set);
    } 
    else {
      diagMass_over = diagMass[set];
    }
    
    assembler->getWeightedMass(set,mass,diagMass_over);
    
    if (linalg->getHaveOverlapped()) {
      linalg->exportVectorFromOverlapped(set,diagMass[set], diagMass_over);
      if (compute_matrix) {
        linalg->exportMatrixFromOverlapped(set,explicitMass[set], mass);
      }
    }
    
  }

  debugger->print("**** Starting SolverManager::setupExplicitMass() - fillComplete");
  
  for (size_t set=0; set<useBasis.size(); ++set) {
    
    if (compute_matrix) {
      linalg->fillComplete(explicitMass[set]);
    }
    
    if (store_vectors) {
      q_pcg.push_back(linalg->getNewVector(set));
      z_pcg.push_back(linalg->getNewVector(set));
      p_pcg.push_back(linalg->getNewVector(set));
      r_pcg.push_back(linalg->getNewVector(set));
      if (linalg->getHaveOverlapped() && assembler->matrix_free) {
        q_pcg_over.push_back(linalg->getNewOverlappedVector(set));
        p_pcg_over.push_back(linalg->getNewOverlappedVector(set));
      }
    }
  }
  
  debugger->print("**** Finished SolverManager::setupExplicitMass()");
  
}


// ========================================================================================
// ========================================================================================

template<class Node>
void SolverManager<Node>::setupDiscretizedParamMass() {

  debugger->print("**** Starting SolverManager::setupDiscretizedParamMass()");
  
  // Hard coding this to always assemble the matrix
  // Can relax this and allow matrix-free later
  bool compute_matrix = true;
  
  matrix_RCP pmass;
  
  if (compute_matrix) {
    
    paramMass = linalg->getNewParamMatrix();
    
    if (linalg->getHaveOverlapped()) {
      pmass = linalg->getNewOverlappedParamMatrix();
    }
    else {
      pmass = paramMass;
    }
    
  }
  
  diagParamMass = linalg->getNewParamVector();
  vector_RCP diagParamMass_over;
  if (linalg->getHaveOverlapped()) {
    diagParamMass_over = linalg->getNewOverlappedParamVector();
  }
  else { // squeeze out memory for single rank demos
    diagParamMass_over = diagParamMass;
  }
  
  assembler->getParamMass(pmass,diagParamMass_over);
  
  if (linalg->getHaveOverlapped()) {
    linalg->exportParamVectorFromOverlapped(diagParamMass, diagParamMass_over);
    if (compute_matrix) {
      linalg->exportParamMatrixFromOverlapped(paramMass, pmass);
    }
  }
  

  if (compute_matrix) {
    linalg->fillComplete(paramMass);
  }
  
  params->setParamMass(diagParamMass, paramMass);
  
  debugger->print("**** Finished SolverManager::setupDiscretizedParamMass()");
  
}

//========================================================================
//========================================================================

template<class Node>
void SolverManager<Node>::setButcherTableau(const vector<string> & tableau, const int & set) {

  for (size_t block=0; block<assembler->groups.size(); ++block) {

    // TODO the RK scheme cannot be specified block by block

    auto myTableau = tableau[set];

    // only filling in the non-zero entries

    if (myTableau == "BWE" || myTableau == "DIRK-1,1") {
      butcher_A = Kokkos::View<ScalarT**,HostDevice>("butcher_A",1,1);
      butcher_A(0,0) = 1.0;
      butcher_b = Kokkos::View<ScalarT*,HostDevice>("butcher_b",1);
      butcher_b(0) = 1.0;
      butcher_c = Kokkos::View<ScalarT*,HostDevice>("butcher_c",1);
      butcher_c(0) = 1.0;
    }
    else if (myTableau == "FWE") {
      butcher_A = Kokkos::View<ScalarT**,HostDevice>("butcher_A",1,1);
      butcher_b = Kokkos::View<ScalarT*,HostDevice>("butcher_b",1);
      butcher_b(0) = 1.0;
      butcher_c = Kokkos::View<ScalarT*,HostDevice>("butcher_c",1);
    }
    else if (myTableau == "CN") {
      butcher_A = Kokkos::View<ScalarT**,HostDevice>("butcher_A",2,2);
      butcher_A(1,0) = 0.5;
      butcher_A(1,1) = 0.5;
      butcher_b = Kokkos::View<ScalarT*,HostDevice>("butcher_b",2);
      butcher_b(0) = 0.5;
      butcher_b(1) = 0.5;
      butcher_c = Kokkos::View<ScalarT*,HostDevice>("butcher_c",2);
      butcher_c(1) = 1.0;
    }
    else if (myTableau == "SSPRK-3,3") {
      butcher_A = Kokkos::View<ScalarT**,HostDevice>("butcher_A",3,3);
      butcher_A(1,0) = 1.0;
      butcher_A(2,0) = 0.25;
      butcher_A(2,1) = 0.25;
      butcher_b = Kokkos::View<ScalarT*,HostDevice>("butcher_b",3);
      butcher_b(0) = 1.0/6.0;
      butcher_b(1) = 1.0/6.0;
      butcher_b(2) = 2.0/3.0;
      butcher_c = Kokkos::View<ScalarT*,HostDevice>("butcher_c",3);
      butcher_c(1) = 1.0;
      butcher_c(2) = 1.0/2.0;
    }
    else if (myTableau == "RK-4,4") { // Classical RK4
      butcher_A = Kokkos::View<ScalarT**,HostDevice>("butcher_A",4,4);
      butcher_A(1,0) = 0.5;
      butcher_A(2,1) = 0.5;
      butcher_A(3,2) = 1.0;
      butcher_b = Kokkos::View<ScalarT*,HostDevice>("butcher_b",4);
      butcher_b(0) = 1.0/6.0;
      butcher_b(1) = 1.0/3.0;
      butcher_b(2) = 1.0/3.0;
      butcher_b(3) = 1.0/6.0;
      butcher_c = Kokkos::View<ScalarT*,HostDevice>("butcher_c",4);
      butcher_c(1) = 1.0/2.0;
      butcher_c(2) = 1.0/2.0;
      butcher_c(3) = 1.0;
    }
    else if (myTableau == "DIRK-1,2") {
      butcher_A = Kokkos::View<ScalarT**,HostDevice>("butcher_A",1,1);
      butcher_A(0,0) = 0.5;
      butcher_b = Kokkos::View<ScalarT*,HostDevice>("butcher_b",1);
      butcher_b(0) = 1.0;
      butcher_c = Kokkos::View<ScalarT*,HostDevice>("butcher_c",1);
      butcher_c(0) = 0.5;
    }
    else if (myTableau == "DIRK-2,2") { // 2-stage, 2nd order
      butcher_A = Kokkos::View<ScalarT**,HostDevice>("butcher_A",2,2);
      butcher_A(0,0) = 1.0/4.0;
      butcher_A(1,0) = 1.0/2.0;
      butcher_A(1,1) = 1.0/4.0;
      butcher_b = Kokkos::View<ScalarT*,HostDevice>("butcher_b",2);
      butcher_b(0) = 1.0/2.0;
      butcher_b(1) = 1.0/2.0;
      butcher_c = Kokkos::View<ScalarT*,HostDevice>("butcher_c",2);
      butcher_c(0) = 1.0/4.0;
      butcher_c(1) = 3.0/4.0;
    }
    else if (myTableau == "DIRK-2,3") { // 2-stage, 3rd order
      butcher_A = Kokkos::View<ScalarT**,HostDevice>("butcher_A",2,2);
      butcher_A(0,0) = 1.0/2.0 + std::sqrt(3)/6.0;
      butcher_A(1,0) = -std::sqrt(3)/3.0;
      butcher_A(1,1) = 1.0/2.0  + std::sqrt(3)/6.0;
      butcher_b = Kokkos::View<ScalarT*,HostDevice>("butcher_b",2);
      butcher_b(0) = 1.0/2.0;
      butcher_b(1) = 1.0/2.0;
      butcher_c = Kokkos::View<ScalarT*,HostDevice>("butcher_c",2);
      butcher_c(0) = 1.0/2.0 + std::sqrt(3)/6.0;;
      butcher_c(1) = 1.0/2.0 - std::sqrt(3)/6.0;;
    }
    else if (myTableau == "DIRK-3,3") { // 3-stage, 3rd order
      ScalarT p = 0.4358665215;
      butcher_A = Kokkos::View<ScalarT**,HostDevice>("butcher_A",3,3);
      butcher_A(0,0) = p;
      butcher_A(1,0) = (1.0-p)/2.0;
      butcher_A(1,1) = p;
      butcher_A(2,0) = -3.0*p*p/2.0+4.0*p-1.0/4.0;
      butcher_A(2,1) = 3.0*p*p/2.0 - 5.0*p + 5.0/4.0;
      butcher_A(2,2) = p;
      butcher_b = Kokkos::View<ScalarT*,HostDevice>("butcher_b",3);
      butcher_b(0) = -3.0*p*p/2.0+4.0*p-1.0/4.0;
      butcher_b(1) = 3.0*p*p/2.0-5.0*p+5.0/4.0;
      butcher_b(2) = p;
      butcher_c = Kokkos::View<ScalarT*,HostDevice>("butcher_c",3);
      butcher_c(0) = p;
      butcher_c(1) = (1.0+p)/2.0;
      butcher_c(2) = 1.0;
    }
    else if (myTableau == "leap-frog") { // Leap-frog for Maxwells
      butcher_A = Kokkos::View<ScalarT**,HostDevice>("butcher_A",2,2);
      butcher_A(1,0) = 1.0;
      butcher_b = Kokkos::View<ScalarT*,HostDevice>("butcher_b",2);
      butcher_b(0) = 1.0;
      butcher_b(1) = 1.0;
      butcher_c = Kokkos::View<ScalarT*,HostDevice>("butcher_c",2);
      butcher_c(0) = 0.0;
      butcher_c(1) = 0.0;
    }
    else if (myTableau == "custom") {

      string delimiter = ", ";
      string line_delimiter = "; ";
      size_t pos = 0;
      string b_A = settings->sublist("Solver").get<string>("transient Butcher A","1.0");
      string b_b = settings->sublist("Solver").get<string>("transient Butcher b","1.0");
      string b_c = settings->sublist("Solver").get<string>("transient Butcher c","1.0");
      vector<vector<double>> A_vals;
      if (b_A.find(delimiter) == string::npos) {
        vector<double> row;
        row.push_back(std::stod(b_A));
        A_vals.push_back(row);
      }
      else {
        string token;
        size_t linepos = 0;
        vector<string> lines;
        while ((linepos = b_A.find(line_delimiter)) != string::npos) {
          string line = b_A.substr(0,linepos);
          lines.push_back(line);
          b_A.erase(0, linepos + line_delimiter.length());
        }
        lines.push_back(b_A);
        for (size_t k=0; k<lines.size(); k++) {
          string line = lines[k];
          vector<double> row;
          while ((pos = line.find(delimiter)) != string::npos) {
            token = line.substr(0, pos);
            row.push_back(std::stod(token));
            line.erase(0, pos + delimiter.length());
          }
          row.push_back(std::stod(line));
          A_vals.push_back(row);
        }
      }
      // Make sure A is square
      size_t A_nrows = A_vals.size();
      for (size_t i=0; i<A_nrows; i++) {
        if (A_vals[i].size() != A_nrows) {
          TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: custom Butcher A is not a square matrix");
        }
      }

      vector<double> b_vals;
      if (b_b.find(delimiter) == string::npos) {
        b_vals.push_back(std::stod(b_b));
      }
      else {
        string token;
        while ((pos = b_b.find(delimiter)) != string::npos) {
          token = b_b.substr(0, pos);
          b_vals.push_back(std::stod(token));
          b_b.erase(0, pos + delimiter.length());
        }
        b_vals.push_back(std::stod(b_b));
      }

      // Make sure size of b matches A
      if (b_vals.size() != A_nrows) {
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: custom Butcher b does not match size of A");
      }

      vector<double> c_vals;
      if (b_c.find(delimiter) == string::npos) {
        c_vals.push_back(std::stod(b_c));
      }
      else {
        string token;
        while ((pos = b_c.find(delimiter)) != string::npos) {
          token = b_c.substr(0, pos);
          c_vals.push_back(std::stod(token));
          b_c.erase(0, pos + delimiter.length());
        }
        c_vals.push_back(std::stod(b_c));
      }

      // Make sure size of c matches A
      if (c_vals.size() != A_nrows) {
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: custom Butcher c does not match size of A");
      }

      // Create the views
      butcher_A = Kokkos::View<ScalarT**,HostDevice>("butcher_A",A_nrows,A_nrows);
      butcher_b = Kokkos::View<ScalarT*,HostDevice>("butcher_b",A_nrows);
      butcher_c = Kokkos::View<ScalarT*,HostDevice>("butcher_c",A_nrows);
      for (size_t i=0; i<A_nrows; i++) {
        for (size_t j=0; j<A_nrows; j++) {
          butcher_A(i,j) = A_vals[i][j];
        }
        butcher_b(i) = b_vals[i];
        butcher_c(i) = c_vals[i];
      }

    }
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: unrecognized Butcher tableau:" + tableau[set]);
    }
    Kokkos::View<ScalarT**,AssemblyDevice> dev_butcher_A("butcher_A on device",butcher_A.extent(0),butcher_A.extent(1));
    Kokkos::View<ScalarT*,AssemblyDevice> dev_butcher_b("butcher_b on device",butcher_b.extent(0));
    Kokkos::View<ScalarT*,AssemblyDevice> dev_butcher_c("butcher_c on device",butcher_c.extent(0));
  
    auto tmp_butcher_A = Kokkos::create_mirror_view(dev_butcher_A);
    auto tmp_butcher_b = Kokkos::create_mirror_view(dev_butcher_b);
    auto tmp_butcher_c = Kokkos::create_mirror_view(dev_butcher_c);
  
    Kokkos::deep_copy(tmp_butcher_A, butcher_A);
    Kokkos::deep_copy(tmp_butcher_b, butcher_b);
    Kokkos::deep_copy(tmp_butcher_c, butcher_c);
  
    Kokkos::deep_copy(dev_butcher_A, tmp_butcher_A);
    Kokkos::deep_copy(dev_butcher_b, tmp_butcher_b);
    Kokkos::deep_copy(dev_butcher_c, tmp_butcher_c);

    //block_butcher_A.push_back(dev_butcher_A);
    //block_butcher_b.push_back(dev_butcher_b);
    //block_butcher_c.push_back(dev_butcher_c);
  
    int newnumstages = butcher_A.extent(0);

    maxnumstages[set] = std::max(numstages[set],newnumstages);
    numstages[set] = newnumstages;
  
    assembler->setWorksetButcher(set, block, dev_butcher_A, dev_butcher_b, dev_butcher_c);

  } // end for blocks
}

// ========================================================================================
// ========================================================================================

template<class Node>
void SolverManager<Node>::setBackwardDifference(const vector<int> & order, const int & set) { // using order as an input to allow for dynamic changes

  // TODO rearrange this? and setButcher...

  for (size_t block=0; block<assembler->groups.size(); ++block) {

    // TODO currently, the BDF wts cannot be specified block by block

    Kokkos::View<ScalarT*,AssemblyDevice> dev_BDF_wts;
    Kokkos::View<ScalarT*,HostDevice> BDF_wts;

    // Note that these do not include 1/deltat (added in wkset)
    // Not going to work properly for adaptive time stepping if BDForder>1

    auto myOrder = order[set];

    if (isTransient) {

      if (myOrder == 1) {
        BDF_wts = Kokkos::View<ScalarT*,HostDevice>("BDF weights to compute u_dot",2);
        BDF_wts(0) = 1.0;
        BDF_wts(1) = -1.0;
      }
      else if (myOrder == 2) {
        BDF_wts = Kokkos::View<ScalarT*,HostDevice>("BDF weights to compute u_dot",3);
        BDF_wts(0) = 1.5;
        BDF_wts(1) = -2.0;
        BDF_wts(2) = 0.5;
      }
      else if (myOrder == 3) {
        BDF_wts = Kokkos::View<ScalarT*,HostDevice>("BDF weights to compute u_dot",4);
        BDF_wts(0) = 11.0/6.0;
        BDF_wts(1) = -3.0;
        BDF_wts(2) = 1.5;
        BDF_wts(3) = -1.0/3.0;
      }
      else if (myOrder == 4) {
        BDF_wts = Kokkos::View<ScalarT*,HostDevice>("BDF weights to compute u_dot",5);
        BDF_wts(0) = 25.0/12.0;
        BDF_wts(1) = -4.0;
        BDF_wts(2) = 3.0;
        BDF_wts(3) = -4.0/3.0;
        BDF_wts(4) = 1.0/4.0;
      }
      else if (myOrder == 5) {
        BDF_wts = Kokkos::View<ScalarT*,HostDevice>("BDF weights to compute u_dot",6);
        BDF_wts(0) = 137.0/60.0;
        BDF_wts(1) = -5.0;
        BDF_wts(2) = 5.0;
        BDF_wts(3) = -10.0/3.0;
        BDF_wts(4) = 75.0/60.0;
        BDF_wts(5) = -1.0/5.0;
      }
      else if (myOrder == 6) {
        BDF_wts = Kokkos::View<ScalarT*,HostDevice>("BDF weights to compute u_dot",7);
        BDF_wts(0) = 147.0/60.0;
        BDF_wts(1) = -6.0;
        BDF_wts(2) = 15.0/2.0;
        BDF_wts(3) = -20.0/3.0;
        BDF_wts(4) = 225.0/60.0;
        BDF_wts(5) = -72.0/60.0;
        BDF_wts(6) = 1.0/6.0;
      }

      int newnumsteps = BDF_wts.extent(0)-1;

      maxnumsteps[set] = std::max(maxnumsteps[set],newnumsteps);
      numsteps[set] = newnumsteps;

    }
    else { // for steady state solves, u_dot = 0.0*u
      BDF_wts = Kokkos::View<ScalarT*,HostDevice>("BDF weights to compute u_dot",1);
      BDF_wts(0) = 1.0;
      numsteps[set] = 1;
      maxnumsteps[set] = 1;
    }

    dev_BDF_wts = Kokkos::View<ScalarT*,AssemblyDevice>("BDF weights on device",BDF_wts.extent(0));
    Kokkos::deep_copy(dev_BDF_wts, BDF_wts);
    
    assembler->setWorksetBDF(set, block, dev_BDF_wts);
  } // end loop blocks
}

/////////////////////////////////////////////////////////////////////////////
// Worksets
/////////////////////////////////////////////////////////////////////////////

template<class Node>
void SolverManager<Node>::finalizeWorkset() {
  
  debugger->print("**** Starting SolverManager::finalizeWorkset ...");
  
  this->finalizeWorkset(assembler->wkset, params->paramvals_KV, params->paramdot_KV);
#ifndef MrHyDE_NO_AD
  this->finalizeWorkset(assembler->wkset_AD, params->paramvals_KVAD, params->paramdot_KVAD);
  this->finalizeWorkset(assembler->wkset_AD2, params->paramvals_KVAD2, params->paramdot_KVAD2);
  this->finalizeWorkset(assembler->wkset_AD4, params->paramvals_KVAD4, params->paramdot_KVAD4);
  this->finalizeWorkset(assembler->wkset_AD8, params->paramvals_KVAD8, params->paramdot_KVAD8);
  this->finalizeWorkset(assembler->wkset_AD16, params->paramvals_KVAD16, params->paramdot_KVAD16);
  this->finalizeWorkset(assembler->wkset_AD18, params->paramvals_KVAD18, params->paramdot_KVAD18);
  this->finalizeWorkset(assembler->wkset_AD24, params->paramvals_KVAD24, params->paramdot_KVAD24);
  this->finalizeWorkset(assembler->wkset_AD32, params->paramvals_KVAD32, params->paramdot_KVAD32);
#endif
  
  debugger->print("**** Finished SolverManager::finalizeWorkset");
  
  
}

template<class Node>
template<class EvalT>
void SolverManager<Node>::finalizeWorkset(vector<Teuchos::RCP<Workset<EvalT> > > & wkset,
                                          Kokkos::View<EvalT**,AssemblyDevice> paramvals_KV,
                                          Kokkos::View<EvalT**,AssemblyDevice> paramdot_KV) {

  // Determine the offsets for each set as a Kokkos View
  for (size_t block=0; block<wkset.size(); ++block) {
    if (wkset[block]->isInitialized) {
      for (size_t set=0; set<physics->set_names.size(); set++) {
        vector<vector<int> > voffsets = disc->offsets[set][block];
        size_t maxoff = 0;
        for (size_t i=0; i<voffsets.size(); i++) {
          if (voffsets[i].size() > maxoff) {
            maxoff = voffsets[i].size();
          }
        }
        
        Kokkos::View<int**,AssemblyDevice> offsets_view("offsets on assembly device",voffsets.size(),maxoff);
        auto host_offsets = Kokkos::create_mirror_view(offsets_view);
        for (size_t i=0; i<voffsets.size(); i++) {
          for (size_t j=0; j<voffsets[i].size(); j++) {
            host_offsets(i,j) = voffsets[i][j];
          }
        }
        Kokkos::deep_copy(offsets_view,host_offsets);
        wkset[block]->set_offsets.push_back(offsets_view);
        if (set == 0) {
          wkset[block]->offsets = offsets_view;
        }

      }
    }
  }
  
  for (size_t block=0; block<wkset.size(); ++block) {
    if (wkset[block]->isInitialized) {
      
      vector<vector<int> > block_useBasis;
      vector<vector<string> > block_varlist;
      
      for (size_t set=0; set<useBasis.size(); ++set) {
        block_useBasis.push_back(useBasis[set][block]);
        block_varlist.push_back(varlist[set][block]);
      }
      wkset[block]->set_usebasis = block_useBasis;
      wkset[block]->set_varlist = block_varlist;
      wkset[block]->usebasis = block_useBasis[0];
      wkset[block]->varlist = block_varlist[0];
      
    }
  }
  
  for (size_t block=0; block<wkset.size(); ++block) {
    if (wkset[block]->isInitialized) {
      // set defaults for time integration params since these
      // won't get set if the total number of sets is 1
      wkset[block]->butcher_A = wkset[block]->set_butcher_A[0];
      wkset[block]->butcher_b = wkset[block]->set_butcher_b[0];
      wkset[block]->butcher_c = wkset[block]->set_butcher_c[0];
      wkset[block]->BDF_wts = wkset[block]->set_BDF_wts[0];
      // update workset for first physics set
      wkset[block]->updatePhysicsSet(0);

    }
  }
  
  // Parameters do not depend on physics sets
  for (size_t block=0; block<wkset.size(); ++block) {
    if (wkset[block]->isInitialized) {
      size_t maxpoff = 0;
      for (size_t i=0; i<params->paramoffsets.size(); i++) {
        if (params->paramoffsets[i].size() > maxpoff) {
          maxpoff = params->paramoffsets[i].size();
        }
      }
      
      Kokkos::View<int**,AssemblyDevice> poffsets_view("param offsets on assembly device",params->paramoffsets.size(),maxpoff);
      auto host_poffsets = Kokkos::create_mirror_view(poffsets_view);
      for (size_t i=0; i<params->paramoffsets.size(); i++) {
        for (size_t j=0; j<params->paramoffsets[i].size(); j++) {
          host_poffsets(i,j) = params->paramoffsets[i][j];
        }
      }
      Kokkos::deep_copy(poffsets_view,host_poffsets);
      wkset[block]->paramusebasis = params->discretized_param_usebasis;
      wkset[block]->paramoffsets = poffsets_view;
      wkset[block]->param_varlist = params->discretized_param_names;

    }
  }
  
  for (size_t block=0; block<wkset.size(); ++block) {
    if (wkset[block]->isInitialized) {
      wkset[block]->createSolutionFields();
    }
  }
  
  for (size_t block=0; block<wkset.size(); ++block) {
    if (wkset[block]->isInitialized) {
      vector<vector<int> > block_useBasis;
      for (size_t set=0; set<useBasis.size(); ++set) {
        block_useBasis.push_back(useBasis[set][block]);
      }
      assembler->groupData[block]->setSolutionFields(maxnumsteps, maxnumstages);
      for (size_t grp=0; grp<assembler->groups[block].size(); ++grp) {
        assembler->groups[block][grp]->setUseBasis(block_useBasis, maxnumsteps, maxnumstages, false);
        assembler->groups[block][grp]->setUpSubGradient(params->num_active_params);
      }
      
      wkset[block]->params_AD = paramvals_KV;
      wkset[block]->params_dot_AD = paramdot_KV;
      wkset[block]->paramnames = params->paramnames;
      wkset[block]->setTime(current_time);

      if (assembler->boundary_groups.size() > block) { // avoid seg faults
        for (size_t grp=0; grp<assembler->boundary_groups[block].size(); ++grp) {
          if (assembler->boundary_groups[block][grp]->numElem > 0) {
            assembler->boundary_groups[block][grp]->setUseBasis(block_useBasis, maxnumsteps, maxnumstages, false);
          }
        }
      }
    }
  }
  
  
}

// ========================================================================================
// Set up the logicals and data structures for the fixed DOF (Dirichlet and point constraints)
// ========================================================================================

template<class Node>
void SolverManager<Node>::setupFixedDOFs(Teuchos::RCP<Teuchos::ParameterList> & settings) {
  
  Teuchos::TimeMonitor localtimer(*fixeddofsetuptimer);
  
  debugger->print("**** Starting SolverManager::setupFixedDOFs()");
  
  if (!disc->have_dirichlet) {
    usestrongDBCs = false;
  }
  
  size_t numSets = physics->set_names.size();
  
  scalarDirichletData = vector<bool>(numSets,false);
  staticDirichletData = vector<bool>(numSets,true);
  
  if (usestrongDBCs) {
    for (size_t set=0; set<numSets; ++set) {
      fixedDOF_soln.push_back(linalg->getNewOverlappedVector(set));
    }
    for (size_t set=0; set<numSets; ++set) {
    
      scalarDirichletData[set] = settings->sublist("Physics").sublist("Dirichlet conditions").get<bool>("scalar data", false);
      staticDirichletData[set] = settings->sublist("Physics").sublist("Dirichlet conditions").get<bool>("static data", true);
      
      if (scalarDirichletData[set] && !staticDirichletData[set]) {
        if (Comm->getRank() == 0) {
          cout << "Warning: The Dirichlet data was set to scalar and non-static.  This should not happen." << endl;
        }
      }
      
      if (scalarDirichletData[set]) {
        vector<vector<ScalarT> > setDirichletValues;
        for (size_t block=0; block<blocknames.size(); ++block) {
          
          std::string blockID = blocknames[block];
          Teuchos::ParameterList dbc_settings = physics->physics_settings[set][block].sublist("Dirichlet conditions");
          vector<ScalarT> blockDirichletValues;
          
          for (size_t var=0; var<varlist[set][block].size(); var++ ) {
            ScalarT value = 0.0;
            if (dbc_settings.isSublist(varlist[set][block][var])) {
              if (dbc_settings.sublist(varlist[set][block][var]).isParameter("all boundaries")) {
                value = dbc_settings.sublist(varlist[set][block][var]).template get<ScalarT>("all boundaries");
              }
              else {
                Teuchos::ParameterList currdbcs = dbc_settings.sublist(varlist[set][block][var]);
                Teuchos::ParameterList::ConstIterator d_itr = currdbcs.begin();
                while (d_itr != currdbcs.end()) {
                  value = currdbcs.get<ScalarT>(d_itr->first);
                  d_itr++;
                }
              }
            }
            blockDirichletValues.push_back(value);
          }
          setDirichletValues.push_back(blockDirichletValues);
        }
        scalarDirichletValues.push_back(setDirichletValues);
      }
    }
  }
  
  debugger->print("**** Finished SolverManager::setupFixedDOFs()");
  
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template<class Node>
void SolverManager<Node>::finalizeParams() {
  
  //for (size_t block=0; block<blocknames.size(); ++block) {
  //  assembler->wkset[block]->paramusebasis = params->discretized_param_usebasis;
  //  assembler->wkset[block]->paramoffsets = params->paramoffsets[0];
  // }
  
}

////////////////////////////////////////////////////////////////////////////////
// The following function is not updated for multi-set
////////////////////////////////////////////////////////////////////////////////

template<class Node>
void SolverManager<Node>::finalizeMultiscale() {
#ifndef MrHyDE_NO_AD
  if (multiscale_manager->subgridModels.size() > 0 ) {
    for (size_t k=0; k<multiscale_manager->subgridModels.size(); k++) {
      multiscale_manager->subgridModels[k]->paramvals_KVAD = params->paramvals_KVAD;
    }
    
    multiscale_manager->macro_wkset = assembler->wkset_AD;
    vector<Kokkos::View<int*,AssemblyDevice>> macro_numDOF;
    for (size_t block=0; block<assembler->groupData.size(); ++block) {
      macro_numDOF.push_back(assembler->groupData[block]->set_num_dof[0]);
    }
    multiscale_manager->setMacroInfo(disc->basis_pointers, disc->basis_types,
                                     physics->var_list[0], useBasis[0], disc->offsets[0],
                                     macro_numDOF, params->paramnames, params->discretized_param_names);
    
    vector<vector<int> > sgmodels = assembler->identifySubgridModels();
    ScalarT my_cost = multiscale_manager->initialize(sgmodels);
    ScalarT gmin = 0.0;
    Teuchos::reduceAll(*Comm,Teuchos::REDUCE_MIN,1,&my_cost,&gmin);
    ScalarT gmax = 0.0;
    Teuchos::reduceAll(*Comm,Teuchos::REDUCE_MAX,1,&my_cost,&gmin);
    
    assembler->multiscale_manager = multiscale_manager;
    if (Comm->getRank() == 0 && verbosity>0) {
      cout << "***** Load Balancing Factor " << gmax/gmin <<  endl;
    }
    
  }
#endif  
}
