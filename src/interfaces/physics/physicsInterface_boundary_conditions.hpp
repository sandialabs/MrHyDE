/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
************************************************************************/

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

// TMW: this function is probably never used

AD PhysicsInterface::getDirichletValue(const int & block, const ScalarT & x, const ScalarT & y,
                                       const ScalarT & z, const ScalarT & t, const string & var,
                                       const string & gside, const bool & useadjoint,
                                       Teuchos::RCP<Workset<AD> > & wkset) {
  
  // update point in wkset
  auto xpt = wkset->getScalarField("x point");
  Kokkos::deep_copy(xpt,x);
  
  auto ypt = wkset->getScalarField("y point");
  Kokkos::deep_copy(ypt,y);
  
  if (dimension == 3) {
    auto zpt = wkset->getScalarField("z point");
    Kokkos::deep_copy(zpt,z);
  }
  
  //wkset->setTime(t);
  
  // evaluate the response
#ifndef MrHyDE_NO_AD
  auto ddata = function_managers_AD[block]->evaluate("Dirichlet " + var + " " + gside,"point");
  return ddata(0,0);
#else
  return 0.0;
#endif

}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

View_Sc2 PhysicsInterface::getDirichlet(const int & var,
                                        const int & set,
                                        const int & block,
                                        const std::string & sidename) {
  
  // evaluate
  
  auto tdvals = function_managers[block]->evaluate("Dirichlet " + var_list[set][block][var] + " " + sidename,"side ip");
  
  View_Sc2 dvals("temp dnvals", function_managers[block]->num_elem_, function_managers[block]->num_ip_side_);
  
  // copy values
  parallel_for("physics fill Dirichlet values",
               RangePolicy<AssemblyExec>(0,dvals.extent(0)),
               KOKKOS_LAMBDA (const int e ) {
    for (size_t i=0; i<dvals.extent(1); i++) {
      dvals(e,i) = tdvals(e,i);
    }
  });
  return dvals;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

std::vector<View_Sc2> PhysicsInterface::getDirichletVector(const int & var,
                                                           const int & set,
                                                           const int & block,
                                                           const std::string & sidename) {
  
  std::vector<View_Sc2> dvals_vec(3);
  string varname = var_list[set][block][var];
  std::vector<string> components = {"x", "y", "z"};
  
  // check if vector component functions exist (e.g., "Dirichlet Ex bottom")
  bool has_components = function_managers[block]->hasFunction("Dirichlet " + varname + "x " + sidename);
  
  if (has_components) {
    // use component-wise Dirichlet data
    for (size_t d=0; d<3; d++) {
      string label = "Dirichlet " + varname + components[d] + " " + sidename;
      if (function_managers[block]->hasFunction(label)) {
        auto tdvals = function_managers[block]->evaluate(label, "side ip");
        View_Sc2 dvals("dirichlet component", function_managers[block]->num_elem_, function_managers[block]->num_ip_side_);
        parallel_for("physics fill Dirichlet vector component",
                     RangePolicy<AssemblyExec>(0,dvals.extent(0)),
                     KOKKOS_LAMBDA (const int e) {
          for (size_t i=0; i<dvals.extent(1); i++) {
            dvals(e,i) = tdvals(e,i);
          }
        });
        dvals_vec[d] = dvals;
      }
      else {
        // use zero, if component not specified
        View_Sc2 dvals("dirichlet component zero", function_managers[block]->num_elem_, function_managers[block]->num_ip_side_);
        Kokkos::deep_copy(dvals, 0.0);
        dvals_vec[d] = dvals;
      }
    }
  }
  else {
    // fall back to scalar Dirichlet data broadcast to all components
    auto tdvals = function_managers[block]->evaluate("Dirichlet " + varname + " " + sidename, "side ip");
    for (size_t d=0; d<3; d++) {
      View_Sc2 dvals("dirichlet component broadcast", function_managers[block]->num_elem_, function_managers[block]->num_ip_side_);
      parallel_for("physics fill Dirichlet broadcast",
                   RangePolicy<AssemblyExec>(0,dvals.extent(0)),
                   KOKKOS_LAMBDA (const int e) {
        for (size_t i=0; i<dvals.extent(1); i++) {
          dvals(e,i) = tdvals(e,i);
        }
      });
      dvals_vec[d] = dvals;
    }
  }
  
  return dvals_vec;
}


/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

template<class EvalT>
void PhysicsInterface::computeFlux(const size_t & set, const size_t block) {
  debugger->print(1, "**** Starting PhysicsInterface compute flux ...");
  
  if (std::is_same<EvalT, ScalarT>::value) {
    for (size_t i=0; i<modules[set][block].size(); i++) {
      modules[set][block][i]->computeFlux();
    }
  }
#ifndef MrHyDE_NO_AD
  else if (std::is_same<EvalT, AD>::value) {
    for (size_t i=0; i<modules_AD[set][block].size(); i++) {
      modules_AD[set][block][i]->computeFlux();
    }
  }
  else if (std::is_same<EvalT, AD2>::value) {
    for (size_t i=0; i<modules_AD2[set][block].size(); i++) {
      modules_AD2[set][block][i]->computeFlux();
    }
  }
  else if (std::is_same<EvalT, AD4>::value) {
    for (size_t i=0; i<modules_AD4[set][block].size(); i++) {
      modules_AD4[set][block][i]->computeFlux();
    }
  }
  else if (std::is_same<EvalT, AD8>::value) {
    for (size_t i=0; i<modules_AD8[set][block].size(); i++) {
      modules_AD8[set][block][i]->computeFlux();
    }
  }
  else if (std::is_same<EvalT, AD16>::value) {
    for (size_t i=0; i<modules_AD16[set][block].size(); i++) {
      modules_AD16[set][block][i]->computeFlux();
    }
  }
  else if (std::is_same<EvalT, AD18>::value) {
    for (size_t i=0; i<modules_AD18[set][block].size(); i++) {
      modules_AD18[set][block][i]->computeFlux();
    }
  }
  else if (std::is_same<EvalT, AD24>::value) {
    for (size_t i=0; i<modules_AD24[set][block].size(); i++) {
      modules_AD24[set][block][i]->computeFlux();
    }
  }
  else if (std::is_same<EvalT, AD32>::value) {
    for (size_t i=0; i<modules_AD32[set][block].size(); i++) {
      modules_AD32[set][block][i]->computeFlux();
    }
  }
#endif
  debugger->print(1, "**** Finished PhysicsInterface compute flux");
  
}

template void PhysicsInterface::computeFlux<ScalarT>(const size_t & set, const size_t block);
#ifndef MrHyDE_NO_AD
template void PhysicsInterface::computeFlux<AD>(const size_t & set, const size_t block);
template void PhysicsInterface::computeFlux<AD2>(const size_t & set, const size_t block);
template void PhysicsInterface::computeFlux<AD4>(const size_t & set, const size_t block);
template void PhysicsInterface::computeFlux<AD8>(const size_t & set, const size_t block);
template void PhysicsInterface::computeFlux<AD16>(const size_t & set, const size_t block);
template void PhysicsInterface::computeFlux<AD18>(const size_t & set, const size_t block);
template void PhysicsInterface::computeFlux<AD24>(const size_t & set, const size_t block);
template void PhysicsInterface::computeFlux<AD32>(const size_t & set, const size_t block);
#endif


/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

template<class EvalT>
void PhysicsInterface::fluxConditions(const size_t & set, const size_t block) {
  if (std::is_same<EvalT, ScalarT>::value) {
    for (size_t var=0; var<var_list[set][block].size(); ++var) {
      int cside = function_managers[block]->wkset->currentside;
      string bctype = function_managers[block]->wkset->var_bcs(var,cside);
      if (bctype == "Flux") {
        string varname = var_list[set][block][var];
        string sidename = function_managers[block]->wkset->sidename;
        string label = "Flux " + varname + " " + sidename;
        auto fluxvals = function_managers[block]->evaluate(label,"side ip");
      
        auto basis = function_managers[block]->wkset->getBasisSide(varname);
        auto wts = function_managers[block]->wkset->wts_side;
        auto res = function_managers[block]->wkset->res;
        auto off = function_managers[block]->wkset->getOffsets(varname);
      
        parallel_for("physics flux condition",
                     TeamPolicy<AssemblyExec>(wts.extent(0), Kokkos::AUTO, VECTORSIZE),
                     KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
          int elem = team.league_rank();
          for (size_type dof=team.team_rank(); dof<basis.extent(1); dof+=team.team_size() ) {
            for (size_type pt=0; pt<basis.extent(2); ++pt ) {
              res(elem,off(dof)) += -fluxvals(elem,pt)*wts(elem,pt)*basis(elem,dof,pt,0);
            }
          }
        });
      }
    }
  }
#ifndef MrHyDE_NO_AD
  else if (std::is_same<EvalT, AD>::value) {
    for (size_t var=0; var<var_list[set][block].size(); ++var) {
      int cside = function_managers_AD[block]->wkset->currentside;
      string bctype = function_managers_AD[block]->wkset->var_bcs(var,cside);
      if (bctype == "Flux") {
        string varname = var_list[set][block][var];
        string sidename = function_managers_AD[block]->wkset->sidename;
        string label = "Flux " + varname + " " + sidename;
        auto fluxvals = function_managers_AD[block]->evaluate(label,"side ip");
      
        auto basis = function_managers_AD[block]->wkset->getBasisSide(varname);
        auto wts = function_managers_AD[block]->wkset->wts_side;
        auto res = function_managers_AD[block]->wkset->res;
        auto off = function_managers_AD[block]->wkset->getOffsets(varname);
      
        parallel_for("physics flux condition",
                     TeamPolicy<AssemblyExec>(wts.extent(0), Kokkos::AUTO, VECTORSIZE),
                     KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
          int elem = team.league_rank();
          for (size_type dof=team.team_rank(); dof<basis.extent(1); dof+=team.team_size() ) {
            for (size_type pt=0; pt<basis.extent(2); ++pt ) {
              res(elem,off(dof)) += -fluxvals(elem,pt)*wts(elem,pt)*basis(elem,dof,pt,0);
            }
          }
        });
      }
    }
  }
#endif
}

template void PhysicsInterface::fluxConditions<ScalarT>(const size_t & set, const size_t block);
#ifndef MrHyDE_NO_AD
template void PhysicsInterface::fluxConditions<AD>(const size_t & set, const size_t block);
template void PhysicsInterface::fluxConditions<AD2>(const size_t & set, const size_t block);
template void PhysicsInterface::fluxConditions<AD4>(const size_t & set, const size_t block);
template void PhysicsInterface::fluxConditions<AD8>(const size_t & set, const size_t block);
template void PhysicsInterface::fluxConditions<AD16>(const size_t & set, const size_t block);
template void PhysicsInterface::fluxConditions<AD18>(const size_t & set, const size_t block);
template void PhysicsInterface::fluxConditions<AD24>(const size_t & set, const size_t block);
template void PhysicsInterface::fluxConditions<AD32>(const size_t & set, const size_t block);
#endif
