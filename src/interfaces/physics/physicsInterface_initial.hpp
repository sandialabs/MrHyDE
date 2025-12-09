/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
************************************************************************/

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

ScalarT PhysicsInterface::getInitialValue(const int & block, const ScalarT & x, const ScalarT & y,
                                const ScalarT & z, const string & var, const bool & useadjoint) {
  
  /*
  // update point in wkset
  wkset->point_KV(0,0,0) = x;
  wkset->point_KV(0,0,1) = y;
  wkset->point_KV(0,0,2) = z;
  
  // evaluate the response
  View_AD2_sv idata = function_manager->evaluate("initial " + var,"point",block);
  return idata(0,0).val();
  */
  return 0.0;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

View_Sc4 PhysicsInterface::getInitial(vector<View_Sc2> & pts, const int & set, const int & block,
                                      const bool & project, Teuchos::RCP<Workset<ScalarT> > & wkset) {
  
  
  size_t currnum_vars = var_list[set][block].size();
  
  View_Sc4 ivals;
  
  if (project) {
    
    ivals = View_Sc4("tmp ivals",pts[0].extent(0), currnum_vars, pts[0].extent(1),dimension);
    
    // ip in wkset are set in cell::getInitial
    for (size_t n=0; n<var_list[set][block].size(); n++) {
      if (types[set][block][n].substr(0,5) == "HGRAD" || types[set][block][n].substr(0,4) == "HVOL") {
        auto tivals = function_managers[block]->evaluate("initial " + var_list[set][block][n],"ip");
        auto cvals = subview( ivals, ALL(), n, ALL(), 0);
        //copy
        parallel_for("physics fill initial values",
                     RangePolicy<AssemblyExec>(0,cvals.extent(0)),
                     KOKKOS_LAMBDA (const int e ) {
          for (size_t i=0; i<cvals.extent(1); i++) {
            cvals(e,i) = tivals(e,i);
          }
        });
      }
      else if (types[set][block][n].substr(0,5) == "HCURL" || types[set][block][n].substr(0,4) == "HDIV") {
        auto tivals = function_managers[block]->evaluate("initial " + var_list[set][block][n] + "[x]","ip");
        auto cvals = subview( ivals, ALL(), n, ALL(), 0);
        //copy
        parallel_for("physics fill initial values",
                     RangePolicy<AssemblyExec>(0,cvals.extent(0)),
                     KOKKOS_LAMBDA (const int e ) {
          for (size_t i=0; i<cvals.extent(1); i++) {
            cvals(e,i) = tivals(e,i);
          }
        });
        if (dimension > 1) {
          auto tivals = function_managers[block]->evaluate("initial " + var_list[set][block][n] + "[y]","ip");
          auto cvals = subview( ivals, ALL(), n, ALL(), 1);
          //copy
          parallel_for("physics fill initial values",
                       RangePolicy<AssemblyExec>(0,cvals.extent(0)),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_t i=0; i<cvals.extent(1); i++) {
              cvals(e,i) = tivals(e,i);
            }
          });
        }
        if (dimension>2) {
          auto tivals = function_managers[block]->evaluate("initial " + var_list[set][block][n] + "[z]","ip");
          auto cvals = subview( ivals, ALL(), n, ALL(), 2);
          //copy
          parallel_for("physics fill initial values",
                       RangePolicy<AssemblyExec>(0,cvals.extent(0)),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_t i=0; i<cvals.extent(1); i++) {
              cvals(e,i) = tivals(e,i);
            }
          });
        }
      }
    }
  }
  else {
    // TMW: will not work on device yet
    
    size_type dim = wkset->dimension;
    size_type Nelem = pts[0].extent(0);
    size_type Npts = pts[0].extent(1);
    
    View_Sc2 ptx("ptx",Nelem,Npts), pty("pty",Nelem,Npts), ptz("ptz",Nelem,Npts);
    ptx = pts[0];
    
    wkset->isOnPoint = true;

    View_Sc2 x,y,z;
    x = wkset->getScalarField("x");
    if (dim > 1) {
      pty = pts[1];
      y = wkset->getScalarField("y");
    }
    if (dim > 2) {
      ptz = pts[2];
      z = wkset->getScalarField("z");
    }
    
    
    ivals = View_Sc4("tmp ivals",Nelem,currnum_vars,Npts,dimension);
    for (size_t e=0; e<ptx.extent(0); e++) {
      for (size_t i=0; i<ptx.extent(1); i++) {
        // set the node in wkset
        int dim_ = dimension;
        parallel_for("physics initial set point",
                     RangePolicy<AssemblyExec>(0,1),
                     KOKKOS_LAMBDA (const int s ) {
          x(0,0) = ptx(e,i); // TMW: this might be ok
          if (dim_ > 1) {
            y(0,0) = pty(e,i);
          }
          if (dim_ > 2) {
            z(0,0) = ptz(e,i);
          }
          
        });
        
        for (size_t n=0; n<var_list[set][block].size(); n++) {
          // evaluate
          auto tivals = function_managers[block]->evaluate("initial " + var_list[set][block][n],"point");
        
          // Also might be ok (terribly inefficient though)
          parallel_for("physics initial set point",
                       RangePolicy<AssemblyExec>(0,1),
                       KOKKOS_LAMBDA (const int s ) {
            ivals(e,n,i,0) = tivals(0,0);
          });
          
        }
      }
    }
    wkset->isOnPoint = false;
  }
   
  return ivals;
}


/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

View_Sc3 PhysicsInterface::getInitialFace(vector<View_Sc2> & pts, const int & set,
                                          const int & block, const bool & project, Teuchos::RCP<Workset<ScalarT> > & wkset) {
  
  size_t currnum_vars = var_list[set][block].size();
  
  View_Sc3 ivals;
  
  if (project) {
    
    ivals = View_Sc3("tmp ivals",pts[0].extent(0), currnum_vars, pts[0].extent(1));
    
    // ip in wkset are set in cell::getInitial
    for (size_t n=0; n<var_list[set][block].size(); n++) {

      auto tivals = function_managers[block]->evaluate("initial " + var_list[set][block][n],"side ip");
      auto cvals = subview( ivals, ALL(), n, ALL());
      //copy
      parallel_for("physics fill initial values",
                   RangePolicy<AssemblyExec>(0,cvals.extent(0)),
                   KOKKOS_LAMBDA (const int e ) {
        for (size_t i=0; i<cvals.extent(1); i++) {
          cvals(e,i) = tivals(e,i);
        }
      });
    }
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(!project,std::runtime_error,"MyHyDE Error: HFACE variables need to use an L2-projection for the initial conditions");
  }
   
  return ivals;
}

