/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
************************************************************************/

// ========================================================================================
// ========================================================================================

template<class Node>
void AssemblyManager<Node>::setInitial(const size_t & set, vector_RCP & rhs, matrix_RCP & mass, const bool & useadjoint,
                                       const bool & lumpmass, const ScalarT & scale) {
  
  Teuchos::TimeMonitor localtimer(*set_init_timer);
  
  debugger->print("**** Starting AssemblyManager::setInitial ...");
  
  for (size_t block=0; block<groups.size(); block++) {
    if (wkset[block]->isInitialized) {
      this->setInitial(set,rhs,mass,useadjoint,lumpmass,scale,block,block);
    }
  }
  
  mass->fillComplete();
  
  debugger->print("**** Finished AssemblyManager::setInitial ...");
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
void AssemblyManager<Node>::setInitial(const size_t & set, vector_RCP & rhs, matrix_RCP & mass, const bool & useadjoint,
                                       const bool & lumpmass, const ScalarT & scale,
                                       const size_t & block, const size_t & groupblock) {
  
  typedef typename Node::execution_space LA_exec;
  using namespace std;
  
  bool use_atomics_ = false;
  if (LA_exec().concurrency() > 1) {
    use_atomics_ = true;
  }
  
  bool fix_zero_rows = true;
  
  auto localMatrix = mass->getLocalMatrixHost();
  auto rhs_view = rhs->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  bool lump_mass_ = lump_mass;
  
  wkset[block]->updatePhysicsSet(set);
  groupData[block]->updatePhysicsSet(set);
  
  auto offsets = wkset[block]->offsets;
  auto numDOF = groupData[block]->num_dof;
  
  for (size_t grp=0; grp<groups[groupblock].size(); ++grp) {
    
    auto LIDs = groups[groupblock][grp]->LIDs[set];
    
    auto localrhs = this->getInitial(groupblock, grp, true, useadjoint);
    auto localmass = this->getMass(groupblock, grp);
    
    parallel_for("assembly insert Jac",
                 RangePolicy<LA_exec>(0,LIDs.extent(0)),
                 MRHYDE_LAMBDA (const int elem ) {
      
      int row = 0;
      LO rowIndex = 0;
      
      for (size_type n=0; n<numDOF.extent(0); ++n) {
        for (int j=0; j<numDOF(n); j++) {
          row = offsets(n,j);
          rowIndex = LIDs(elem,row);
          ScalarT val = localrhs(elem,row);
          if (use_atomics_) {
            Kokkos::atomic_add(&(rhs_view(rowIndex,0)), val);
          }
          else {
            rhs_view(rowIndex,0) += val;
          }
        }
      }
      
      const size_type numVals = LIDs.extent(1);
      int col = 0;
      LO cols[MAXDERIVS];
      ScalarT vals[MAXDERIVS];
      for (size_type n=0; n<numDOF.extent(0); ++n) {
        for (int j=0; j<numDOF(n); j++) {
          row = offsets(n,j);
          rowIndex = LIDs(elem,row);
          for (size_type m=0; m<numDOF.extent(0); m++) {
            for (int k=0; k<numDOF(m); k++) {
              col = offsets(m,k);
              vals[col] = localmass(elem,row,col);
              if (lump_mass_) {
                cols[col] = rowIndex;
              }
              else {
                cols[col] = LIDs(elem,col);
              }
            }
          }
          localMatrix.sumIntoValues(rowIndex, cols, numVals, vals, false, use_atomics_);
        }
      }
    });
  }
  
  if (fix_zero_rows) {
    size_t numrows = mass->getLocalNumRows();
    
    parallel_for("assembly insert Jac",
                 RangePolicy<LA_exec>(0,numrows),
                 MRHYDE_LAMBDA (const size_t row ) {
      auto rowdata = localMatrix.row(row);
      ScalarT abssum = 0.0;
      for (int col=0; col<rowdata.length; ++col ) {
        abssum += abs(rowdata.value(col));
      }
      ScalarT val[1];
      LO cols[1];
      if (abssum<1.0e-14) { // needs to be generalized!
        val[0] = 1.0;
        cols[0] = row;
        localMatrix.replaceValues(row,cols,1,val,false,false);
      }
    });
  }
}

// ========================================================================================
// ========================================================================================

template<class Node>
void AssemblyManager<Node>::setInitial(const size_t & set, vector_RCP & initial, const bool & useadjoint) {
  
  for (size_t block=0; block<groups.size(); ++block) {
    for (size_t grp=0; grp<groups[block].size(); ++grp) {
      LIDView_host LIDs = groups[block][grp]->LIDs_host[set];
      Kokkos::View<ScalarT**,AssemblyDevice> localinit = this->getInitial(block, grp, false, useadjoint);
      auto host_init = Kokkos::create_mirror_view(localinit);
      Kokkos::deep_copy(host_init,localinit);
      int numElem = groups[block][grp]->numElem;
      for (int c=0; c<numElem; c++) {
        for( size_t row=0; row<LIDs.extent(1); row++ ) {
          LO rowIndex = LIDs(c,row);
          ScalarT val = host_init(c,row);
          initial->replaceLocalValue(rowIndex,0, val);
        }
      }
    }
  }
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
void AssemblyManager<Node>::setInitialFace(const size_t & set, vector_RCP & rhs, matrix_RCP & mass,
                                           const bool & lumpmass) {
  
  Teuchos::TimeMonitor localtimer(*set_init_timer);
  
  using namespace std;
  debugger->print("**** Starting AssemblyManager::setInitialFace ...");
  
  auto localMatrix = mass->getLocalMatrixHost();
  
  for (size_t block=0; block<groups.size(); ++block) {
    wkset[block]->isOnSide = true;
    for (size_t grp=0; grp<groups[block].size(); ++grp) {
      int numElem = groups[block][grp]->numElem;
      auto LIDs = groups[block][grp]->LIDs_host[set];
      // Get the requested IC from the group
      auto localrhs = this->getInitialFace(block, grp, true);
      // Create the mass matrix
      auto localmass = this->getMassFace(block, grp);
      auto host_rhs = Kokkos::create_mirror_view(localrhs);
      auto host_mass = localmass;//Kokkos::create_mirror_view(localmass);
      Kokkos::deep_copy(host_rhs,localrhs);
      //Kokkos::deep_copy(host_mass,localmass);
      
      size_t numVals = LIDs.extent(1);
      // assemble into global matrix
      for (int c=0; c<numElem; c++) {
        for( size_t row=0; row<LIDs.extent(1); row++ ) {
          LO rowIndex = LIDs(c,row);
          ScalarT val = host_rhs(c,row);
          rhs->sumIntoLocalValue(rowIndex,0, val);
          if (lumpmass) {
            LO cols[1];
            ScalarT vals[1];
            
            ScalarT totalval = 0.0;
            for( size_t col=0; col<LIDs.extent(1); col++ ) {
              cols[0] = LIDs(c,col);
              totalval += host_mass(c,row,col);
            }
            vals[0] = totalval;
            localMatrix.sumIntoValues(rowIndex, cols, 1, vals, true, false);
          }
          else {
            LO cols[MAXDERIVS];
            ScalarT vals[MAXDERIVS];
            for( size_t col=0; col<LIDs.extent(1); col++ ) {
              cols[col] = LIDs(c,col);
              vals[col] = host_mass(c,row,col);
            }
            localMatrix.sumIntoValues(rowIndex, cols, numVals, vals, true, false);
            
          }
        }
      }
    }
    wkset[block]->isOnSide = false;
  }
  
  // make sure we don't have any rows of all zeroes
  // TODO I don't think this can ever happen?
  // at least globally
  
  typedef typename Node::execution_space LA_exec;
  size_t numrows = mass->getLocalNumRows();
  
  parallel_for("assembly insert Jac",
               RangePolicy<LA_exec>(0,numrows),
               MRHYDE_LAMBDA (const size_t row ) {
    auto rowdata = localMatrix.row(row);
    ScalarT abssum = 0.0;
    for (int col=0; col<rowdata.length; ++col ) {
      abssum += abs(rowdata.value(col));
    }
    ScalarT val[1];
    LO cols[1];
    if (abssum<1.0e-14) { // needs to be generalized!
      val[0] = 1.0;
      cols[0] = row;
      localMatrix.replaceValues(row,cols,1,val,false,false);
    }
  });
  
  mass->fillComplete();
  
  debugger->print("**** Finished AssemblyManager::setInitialFace ...");
  
}


///////////////////////////////////////////////////////////////////////////////////////
// Get the initial condition
///////////////////////////////////////////////////////////////////////////////////////

template<class Node>
View_Sc2 AssemblyManager<Node>::getInitial(const int & block, const size_t & grp,
                                           const bool & project, const bool & isAdjoint) {
  
  size_t set = wkset[block]->current_set;
  View_Sc2 initialvals("initial values",groups[block][grp]->numElem, groups[block][grp]->LIDs[set].extent(1));
  this->updateWorkset<ScalarT>(block, grp, 0,0);
  
  auto offsets = wkset[block]->offsets;
  auto numDOF = groupData[block]->num_dof;
  auto cwts = groups[block][grp]->wts;
  
  if (project) { // works for any basis
    vector<View_Sc2> ip = groups[block][grp]->getIntegrationPts();
    auto initialip = groupData[block]->physics->getInitial(ip, set,
                                                        groupData[block]->my_block,
                                                        project, wkset[block]);

    for (size_type n=0; n<numDOF.extent(0); n++) {
      auto cbasis = groups[block][grp]->basis[wkset[block]->usebasis[n]];
      auto off = subview(offsets, n, ALL());
      string btype = wkset[block]->basis_types[wkset[block]->usebasis[n]];
      if (btype.substr(0,5) == "HGRAD" || btype.substr(0,4) == "HVOL") {
        auto initvar = subview(initialip, ALL(), n, ALL(), 0);
        parallel_for("Group init project",
                     TeamPolicy<AssemblyExec>(initvar.extent(0), Kokkos::AUTO),
                     MRHYDE_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
          int elem = team.league_rank();
          for (size_type dof=team.team_rank(); dof<cbasis.extent(1); dof+=team.team_size() ) {
            for(size_type pt=0; pt<cwts.extent(1); pt++ ) {
              initialvals(elem,off(dof)) += initvar(elem,pt)*cbasis(elem,dof,pt,0)*cwts(elem,pt);
            }
          }
        });
      }
      else if (btype.substr(0,5) == "HCURL" || btype.substr(0,4) == "HDIV") {
        auto initvar = subview(initialip, ALL(), n, ALL(), ALL());
        parallel_for("Group init project",
                     TeamPolicy<AssemblyExec>(initvar.extent(0), Kokkos::AUTO),
                     MRHYDE_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
          int elem = team.league_rank();
          for (size_type dof=team.team_rank(); dof<cbasis.extent(1); dof+=team.team_size() ) {
            for (size_type pt=0; pt<cwts.extent(1); pt++ ) {
              for (size_type dim=0; dim<cbasis.extent(3); dim++ ) {
                initialvals(elem,off(dof)) += initvar(elem,pt,dim)*cbasis(elem,dof,pt,dim)*cwts(elem,pt);
              }
            }
          }
        });
      }
      
    }
  }
  else { // only works if using HGRAD linear basis
    vector<View_Sc2> vnodes;
    View_Sc2 vx,vy,vz;
    DRV nodes = disc->getMyNodes(block, groups[block][grp]->localElemID);
    vx = View_Sc2("view of nodes", nodes.extent(0), nodes.extent(1));
    auto n_x = subview(nodes,ALL(),ALL(),0);
    deep_copy(vx,n_x);
    vnodes.push_back(vx);
    if (nodes.extent(2) > 1) {
      vy = View_Sc2("view of nodes", nodes.extent(0), nodes.extent(1));
      auto n_y = subview(nodes,ALL(),ALL(),1);
      deep_copy(vy,n_y);
      vnodes.push_back(vy);
    }
    if (nodes.extent(2) > 2) {
      vz = View_Sc2("view of nodes", nodes.extent(0), nodes.extent(1));
      auto n_z = subview(nodes, ALL(), ALL(), 2);
      deep_copy(vz,n_z);
      vnodes.push_back(vz);
    }
    
    auto initialnodes = groupData[block]->physics->getInitial(vnodes, set,
                                                          groupData[block]->my_block,
                                                          project,
                                                          wkset[block]);
    for (size_type n=0; n<numDOF.extent(0); n++) {
      auto off = subview( offsets, n, ALL());
      auto initvar = subview(initialnodes, ALL(), n, ALL(), 0);
      parallel_for("Group init project",
                   TeamPolicy<AssemblyExec>(initvar.extent(0), Kokkos::AUTO),
                   MRHYDE_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        for (size_type dof=team.team_rank(); dof<initvar.extent(1); dof+=team.team_size() ) {
          initialvals(elem,off(dof)) = initvar(elem,dof);
        }
      });
    }
  }
  return initialvals;
}

///////////////////////////////////////////////////////////////////////////////////////
// Get the initial condition on the faces
///////////////////////////////////////////////////////////////////////////////////////

template<class Node>
View_Sc2 AssemblyManager<Node>::getInitialFace(const int & block, const size_t & grp, const bool & project) {
  
  size_t set = wkset[block]->current_set;
  View_Sc2 initialvals("initial values",groups[block][grp]->numElem, groups[block][grp]->LIDs[set].extent(1)); // TODO is this too big?
  this->updateWorkset<ScalarT>(block, grp, 0, 0); // TODO not sure if this is necessary

  auto offsets = wkset[block]->offsets;
  auto numDOF = groupData[block]->num_dof;

  // loop over faces of the reference element
  for (size_t face=0; face<groupData[block]->num_sides; face++) {

    // get basis functions, weights, etc. for that face
    this->updateWorksetFace<ScalarT>(block, grp, face);
    auto cwts = wkset[block]->wts_side; // face weights get put into wts_side after update
    // get data from IC
    auto initialip = groupData[block]->physics->getInitialFace(groups[block][grp]->ip_face[face], set,
                                                           groupData[block]->my_block,
                                                           project,
                                                           wkset[block]);
    for (size_type n=0; n<numDOF.extent(0); n++) {
      auto cbasis = wkset[block]->basis_side[wkset[block]->usebasis[n]]; // face basis gets put here after update
      auto off = subview(offsets, n, ALL());
      auto initvar = subview(initialip, ALL(), n, ALL());
      // loop over mesh elements
      parallel_for("Group init project",
                   TeamPolicy<AssemblyExec>(initvar.extent(0), Kokkos::AUTO),
                   MRHYDE_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        for (size_type dof=team.team_rank(); dof<cbasis.extent(1); dof+=team.team_size() ) {
          for(size_type pt=0; pt<cwts.extent(1); pt++ ) {
            initialvals(elem,off(dof)) += initvar(elem,pt)*cbasis(elem,dof,pt,0)*cwts(elem,pt);
          }
        }
      });
    }
  }
  
  return initialvals;
}

