/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::computeSolAvg(const int & block, const size_t & grp) {
  
  // THIS FUNCTION ASSUMES THAT THE WORKSET BASIS HAS BEEN UPDATED
  
  //Teuchos::TimeMonitor localtimer(*computeSolAvgTimer);
  /*
  // Compute the average weight, i.e., the size of each elem
  // May consider storing this
  auto cwts = wkset[block]->wts;
  View_Sc1 avgwts("elem size",cwts.extent(0));
  parallel_for("Group sol avg",
               RangePolicy<AssemblyExec>(0,cwts.extent(0)),
               KOKKOS_CLASS_LAMBDA (const size_type elem ) {
    ScalarT avgwt = 0.0;
    for (size_type pt=0; pt<cwts.extent(1); pt++) {
      avgwt += cwts(elem,pt);
    }
    avgwts(elem) = avgwt;
  });
  
  for (size_t set=0; set<groups[block][grp]->sol_avg.size(); ++set) {
    
    // HGRAD vars
    vector<int> vars_HGRAD = wkset[block]->vars_HGRAD[set];
    vector<string> varlist_HGRAD = wkset[block]->varlist_HGRAD[set];
    for (size_t i=0; i<vars_HGRAD.size(); ++i) {
      auto sol = wkset[block]->getSolutionField(varlist_HGRAD[i]);
      auto savg = subview(groups[block][grp]->sol_avg[set],ALL(),vars_HGRAD[i],0);
      parallel_for("Group sol avg",
                   RangePolicy<AssemblyExec>(0,savg.extent(0)),
                   KOKKOS_CLASS_LAMBDA (const size_type elem ) {
        ScalarT solavg = 0.0;
        for (size_type pt=0; pt<sol.extent(2); pt++) {
          solavg += sol(elem,pt)*cwts(elem,pt);
        }
        savg(elem) = solavg/avgwts(elem);
      });
    }
    
    // HVOL vars
    vector<int> vars_HVOL = wkset[block]->vars_HVOL[set];
    vector<string> varlist_HVOL = wkset[block]->varlist_HVOL[set];
    for (size_t i=0; i<vars_HVOL.size(); ++i) {
      auto sol = wkset[block]->getSolutionField(varlist_HVOL[i]);
      auto savg = subview(groups[block][grp]->sol_avg[set],ALL(),vars_HVOL[i],0);
      parallel_for("Group sol avg",
                   RangePolicy<AssemblyExec>(0,savg.extent(0)),
                   KOKKOS_CLASS_LAMBDA (const size_type elem ) {
        ScalarT solavg = 0.0;
        for (size_type pt=0; pt<sol.extent(2); pt++) {
          solavg += sol(elem,pt)*cwts(elem,pt);
        }
        savg(elem) = solavg/avgwts(elem);
      });
    }
    
    // Compute the postfix options for vector vars
    vector<string> postfix = {"[x]"};
    if (groups[block][grp]->sol_avg[set].extent(2) > 1) { // 2D or 3D
      postfix.push_back("[y]");
    }
    if (groups[block][grp]->sol_avg[set].extent(2) > 2) { // 3D
      postfix.push_back("[z]");
    }
    
    // HDIV vars
    vector<int> vars_HDIV = wkset[block]->vars_HDIV[set];
    vector<string> varlist_HDIV = wkset[block]->varlist_HDIV[set];
    for (size_t i=0; i<vars_HDIV.size(); ++i) {
      for (size_t j=0; j<postfix.size(); ++j) {
        auto sol = wkset[block]->getSolutionField(varlist_HDIV[i]+postfix[j]);
        auto savg = subview(groups[block][grp]->sol_avg[set],ALL(),vars_HDIV[i],j);
        parallel_for("Group sol avg",
                     RangePolicy<AssemblyExec>(0,savg.extent(0)),
                     KOKKOS_CLASS_LAMBDA (const size_type elem ) {
          ScalarT solavg = 0.0;
          for (size_type pt=0; pt<sol.extent(2); pt++) {
            solavg += sol(elem,pt)*cwts(elem,pt);
          }
          savg(elem) = solavg/avgwts(elem);
        });
      }
    }
    
    // HCURL vars
    vector<int> vars_HCURL = wkset[block]->vars_HCURL[set];
    vector<string> varlist_HCURL = wkset[block]->varlist_HCURL[set];
    for (size_t i=0; i<vars_HCURL.size(); ++i) {
      for (size_t j=0; j<postfix.size(); ++j) {
        auto sol = wkset[block]->getSolutionField(varlist_HCURL[i]+postfix[j]);
        auto savg = subview(groups[block][grp]->sol_avg[set],ALL(),vars_HCURL[i],j);
        parallel_for("Group sol avg",
                     RangePolicy<AssemblyExec>(0,savg.extent(0)),
                     KOKKOS_CLASS_LAMBDA (const size_type elem ) {
          ScalarT solavg = 0.0;
          for (size_type pt=0; pt<sol.extent(2); pt++) {
            solavg += sol(elem,pt)*cwts(elem,pt);
          }
          savg(elem) = solavg/avgwts(elem);
        });
      }
    }
  }
  */

  /*
  if (param_avg.extent(1) > 0) {
    View_AD4 psol = wkset[block]->local_param;
    auto pavg = param_avg;

    parallel_for("Group param avg",
                 RangePolicy<AssemblyExec>(0,pavg.extent(0)),
                 KOKKOS_CLASS_LAMBDA (const size_type elem ) {
      ScalarT avgwt = 0.0;
      for (size_type pt=0; pt<cwts.extent(1); pt++) {
        avgwt += cwts(elem,pt);
      }
      for (size_type dof=0; dof<psol.extent(1); dof++) {
        for (size_type dim=0; dim<psol.extent(3); dim++) {
          ScalarT solavg = 0.0;
          for (size_type pt=0; pt<psol.extent(2); pt++) {
            solavg += psol(elem,dof,pt,dim).val()*cwts(elem,pt);
          }
          pavg(elem,dof,dim) = solavg/avgwt;
        }
      }
    });
  }
   */
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::computeSolutionAverage(const int & block, const size_t & grp,
                                                  const string & var, View_Sc2 csol) {
  // Figure out which basis we need
  int index;
  wkset[block]->isVar(var,index);
  
  CompressedView<View_Sc4> cbasis;
  auto cwts = groups[block][grp]->wts;

  if (groups[block][grp]->storeAll || groups[block][grp]->group_data->use_basis_database) {
    cbasis = groups[block][grp]->basis[wkset[block]->usebasis[index]];
  }
  else {
    vector<View_Sc4> tbasis, tbasis_grad, tbasis_curl, tbasis_nodes;
    vector<View_Sc3> tbasis_div;
    disc->getPhysicalVolumetricBasis(groupData[block], groups[block][grp]->nodes, groups[block][grp]->orientation,
                                     tbasis, tbasis_grad, tbasis_curl,
                                     tbasis_div, tbasis_nodes);
    cbasis = CompressedView<View_Sc4>(tbasis[wkset[block]->usebasis[index]]);
  }
  
  // Compute the average weight, i.e., the size of each elem
  // May consider storing this
  View_Sc1 avgwts("elem size",cwts.extent(0));
  parallel_for("Group sol avg",
               RangePolicy<AssemblyExec>(0,cwts.extent(0)),
               KOKKOS_CLASS_LAMBDA (const size_type elem ) {
    ScalarT avgwt = 0.0;
    for (size_type pt=0; pt<cwts.extent(1); pt++) {
      avgwt += cwts(elem,pt);
    }
    avgwts(elem) = avgwt;
  });
  
  size_t set = wkset[block]->current_set;
  
  auto scsol = subview(groupData[block]->sol[set],ALL(),index,ALL());
  parallel_for("wkset[block] soln ip HGRAD",
               RangePolicy<AssemblyExec>(0,cwts.extent(0)),
               KOKKOS_CLASS_LAMBDA (const size_type elem ) {
    for (size_type dim=0; dim<cbasis.extent(3); ++dim) {
      ScalarT avgval = 0.0;
      for (size_type dof=0; dof<cbasis.extent(1); ++dof ) {
        for (size_type pt=0; pt<cbasis.extent(2); ++pt) {
          avgval += scsol(elem,dof)*cbasis(elem,dof,pt,dim)*cwts(elem,pt);
        }
      }
      csol(elem,dim) = avgval/avgwts(elem);
    }
  });
  
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::computeParameterAverage(const int & block, const size_t & grp,
                                                    const string & var, View_Sc2 sol) {
  
  //Teuchos::TimeMonitor localtimer(*computeSolAvgTimer);
  
  // Figure out which basis we need
  int index;
  wkset[block]->isParameter(var,index);
  
  CompressedView<View_Sc4> cbasis;
  auto cwts = groups[block][grp]->wts;

  if (groups[block][grp]->storeAll || groupData[block]->use_basis_database) {
    cbasis = groups[block][grp]->basis[wkset[block]->paramusebasis[index]];
  }
  else {
    vector<View_Sc4> tbasis, tbasis_grad, tbasis_curl, tbasis_nodes;
    vector<View_Sc3> tbasis_div;
    disc->getPhysicalVolumetricBasis(groupData[block], groups[block][grp]->nodes, groups[block][grp]->orientation,
                                     tbasis, tbasis_grad, tbasis_curl,
                                     tbasis_div, tbasis_nodes);
    cbasis = CompressedView<View_Sc4>(tbasis[wkset[block]->paramusebasis[index]]);
  }
  
  // Compute the average weight, i.e., the size of each elem
  // May consider storing this
  View_Sc1 avgwts("elem size",cwts.extent(0));
  parallel_for("Group sol avg",
               RangePolicy<AssemblyExec>(0,cwts.extent(0)),
               KOKKOS_CLASS_LAMBDA (const size_type elem ) {
    ScalarT avgwt = 0.0;
    for (size_type pt=0; pt<cwts.extent(1); pt++) {
      avgwt += cwts(elem,pt);
    }
    avgwts(elem) = avgwt;
  });
  
  auto csol = subview(groupData[block]->param,ALL(),index,ALL());
  parallel_for("wkset[block] soln ip HGRAD",
               RangePolicy<AssemblyExec>(0,cwts.extent(0)),
               KOKKOS_CLASS_LAMBDA (const size_type elem ) {
    for (size_type dim=0; dim<cbasis.extent(3); ++dim) {
      ScalarT avgval = 0.0;
      for (size_type dof=0; dof<cbasis.extent(1); ++dof ) {
        for (size_type pt=0; pt<cbasis.extent(2); ++pt) {
          avgval += csol(elem,dof)*cbasis(elem,dof,pt,dim)*cwts(elem,pt);
        }
      }
      sol(elem,dim) = avgval/avgwts(elem);
    }
  });
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Get the solution at the nodes
///////////////////////////////////////////////////////////////////////////////////////

template<class Node>
Kokkos::View<ScalarT***,AssemblyDevice> AssemblyManager<Node>::getSolutionAtNodes(const int & block, const size_t & grp, const int & var) {
  
  size_t set = wkset[block]->current_set;
  
  int bnum = wkset[block]->usebasis[var];
  auto cbasis = groups[block][grp]->basis_nodes[bnum];
    
  Kokkos::View<ScalarT***,AssemblyDevice> nodesol("solution at nodes",
                                                  cbasis.extent(0), cbasis.extent(2), groupData[block]->dimension);
  auto uvals = subview(groupData[block]->sol[set], ALL(), var, ALL());
  parallel_for("Group node sol",
               RangePolicy<AssemblyExec>(0,cbasis.extent(0)),
               KOKKOS_CLASS_LAMBDA (const size_type elem ) {
    for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
      ScalarT uval = uvals(elem,dof);
      for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
        for (size_type s=0; s<cbasis.extent(3); s++ ) {
          nodesol(elem,pt,s) += uval*cbasis(elem,dof,pt,s);
        }
      }
    }
  });
  
  return nodesol;
  
}


template<class Node>
vector<vector<int> > AssemblyManager<Node>::identifySubgridModels() {

  vector<vector<int> > sgmodels;

#ifndef MrHyDE_NO_AD

  for (size_t block=0; block<groups.size(); ++block) {
    
    vector<int> block_sgmodels;
    bool uses_subgrid = false;
    for (size_t s=0; s<multiscale_manager->subgridModels.size(); s++) {
      if (multiscale_manager->subgridModels[s]->macro_block == block) {
        uses_subgrid = true;
      }
    }
    
    if (uses_subgrid) {
      for (size_t grp=0; grp<groups[block].size(); ++grp) {
        
        this->updateWorkset<AD>(block, grp, 0, 0);
        
        vector<int> sgvotes(multiscale_manager->subgridModels.size(),0);
        
        for (size_t s=0; s<multiscale_manager->subgridModels.size(); s++) {
          if (multiscale_manager->subgridModels[s]->macro_block == block) {
            std::stringstream ss;
            ss << s;
            auto usagecheck = function_managers_AD[block]->evaluate(multiscale_manager->subgridModels[s]->name + " usage","ip");
            
            Kokkos::View<ScalarT**,AssemblyDevice> usagecheck_tmp("temp usage check",
                                                                  function_managers[block]->num_elem_,
                                                                  function_managers[block]->num_ip_);
                                                                  
            parallel_for("assembly copy LIDs",
                         RangePolicy<AssemblyExec>(0,usagecheck_tmp.extent(0)),
                         KOKKOS_CLASS_LAMBDA (const int i ) {
              for (size_type j=0; j<usagecheck_tmp.extent(1); j++) {
                usagecheck_tmp(i,j) = usagecheck(i,j).val();
              }
            });
            
            auto host_usagecheck = Kokkos::create_mirror_view(usagecheck_tmp);
            Kokkos::deep_copy(host_usagecheck, usagecheck_tmp);
            for (size_t p=0; p<groups[block][grp]->numElem; p++) {
              for (size_t j=0; j<host_usagecheck.extent(1); j++) {
                if (host_usagecheck(p,j) >= 1.0) {
                  sgvotes[s] += 1;
                }
              }
            }
          }
        }
        
        int maxvotes = -1;
        int sgwinner = 0;
        for (size_t i=0; i<sgvotes.size(); i++) {
          if (sgvotes[i] >= maxvotes) {
            maxvotes = sgvotes[i];
            sgwinner = i;
          }
        }
        block_sgmodels.push_back(sgwinner);
      }
    }
    sgmodels.push_back(block_sgmodels); 
  }
#endif
  return sgmodels;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::setMeshData() {
  
  debugger->print("**** Starting assembly manager setMeshData");
  
  if (mesh->have_quadrature_data) {
    this->importQuadratureData();
  }
  else if (mesh->have_mesh_data) {
    this->importMeshData();
  }
  else if (mesh->compute_mesh_data) {
    int randSeed = settings->sublist("Mesh").get<int>("random seed", 1234);
    auto seeds = mesh->generateNewMicrostructure(randSeed);
    this->importNewMicrostructure(randSeed, seeds);
  }
  
  debugger->print("**** Finished mesh interface setMeshData");
  
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::importMeshData() {
  
  debugger->print("**** Starting AssemblyManager::importMeshData ...");
  
  Teuchos::Time meshimporttimer("mesh import", false);
  meshimporttimer.start();
  
  int numdata = 1;
  if (mesh->have_rotations) {
    numdata = 9;
  }
  else if (mesh->have_rotation_phi) {
    numdata = 3;
  }
  
  for (size_t block=0; block<groups.size(); ++block) {
    for (size_t grp=0; grp<groups[block].size(); ++grp) {
      int numElem = groups[block][grp]->numElem;
      Kokkos::View<ScalarT**,AssemblyDevice> cell_data("cell_data",numElem,numdata);
      groups[block][grp]->data = cell_data;
      groups[block][grp]->data_distance = vector<ScalarT>(numElem);
      groups[block][grp]->data_seed = vector<size_t>(numElem);
      groups[block][grp]->data_seedindex = vector<size_t>(numElem);
    }
  }
  for (size_t block=0; block<boundary_groups.size(); ++block) {
    for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
      int numElem = boundary_groups[block][grp]->numElem;
      Kokkos::View<ScalarT**,AssemblyDevice> cell_data("cell_data",numElem,numdata);
      boundary_groups[block][grp]->data = cell_data;
      boundary_groups[block][grp]->data_distance = vector<ScalarT>(numElem);
      boundary_groups[block][grp]->data_seed = vector<size_t>(numElem);
      boundary_groups[block][grp]->data_seedindex = vector<size_t>(numElem);
    }
  }
  
  Teuchos::RCP<Data> mesh_data;
  
  string mesh_data_pts_file = mesh->mesh_data_pts_tag + ".dat";
  string mesh_data_file = mesh->mesh_data_tag + ".dat";
  
  bool have_grid_data = settings->sublist("Mesh").get<bool>("data on grid",false);
  if (have_grid_data) {
    int Nx = settings->sublist("Mesh").get<int>("data grid Nx",0);
    int Ny = settings->sublist("Mesh").get<int>("data grid Ny",0);
    int Nz = settings->sublist("Mesh").get<int>("data grid Nz",0);
    mesh_data = Teuchos::rcp(new Data("mesh data", mesh->dimension, mesh_data_pts_file,
                                      mesh_data_file, false, Nx, Ny, Nz));
    
    for (size_t block=0; block<groups.size(); ++block) {
      for (size_t grp=0; grp<groups[block].size(); ++grp) {
        DRV nodes = groups[block][grp]->nodes;
        int numElem = groups[block][grp]->numElem;
        
        auto centers = mesh->getElementCenters(nodes, groups[block][grp]->group_data->cell_topo);
        auto centers_host = create_mirror_view(centers);
        deep_copy(centers_host,centers);
        
        for (int c=0; c<numElem; c++) {
          ScalarT distance = 0.0;
          
          // Doesn't use the Compadre interface
          int cnode = mesh_data->findClosestGridPoint(centers_host(c,0), centers_host(c,1),
                                                      centers_host(c,2), distance);
          
          Kokkos::View<ScalarT**,HostDevice> cdata = mesh_data->getData(cnode);
          for (size_type i=0; i<cdata.extent(1); i++) {
            groups[block][grp]->data(c,i) = cdata(0,i);
          }
          groups[block][grp]->group_data->have_extra_data = true;
          groups[block][grp]->group_data->have_rotation = mesh->have_rotations;
          groups[block][grp]->group_data->have_phi = mesh->have_rotation_phi;
          
          groups[block][grp]->data_seed[c] = cnode;
          groups[block][grp]->data_seedindex[c] = cnode % 100;
          groups[block][grp]->data_distance[c] = distance;
        }
      }
    }
    
    for (size_t block=0; block<boundary_groups.size(); ++block) {
      for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
        DRV nodes = boundary_groups[block][grp]->nodes;
        int numElem = boundary_groups[block][grp]->numElem;
        
        auto centers = mesh->getElementCenters(nodes, boundary_groups[block][grp]->group_data->cell_topo);
        auto centers_host = create_mirror_view(centers);
        deep_copy(centers_host,centers);
        
        for (int c=0; c<numElem; c++) {
          ScalarT distance = 0.0;
          
          int cnode = mesh_data->findClosestGridPoint(centers_host(c,0), centers_host(c,1),
                                                      centers_host(c,2), distance);
          Kokkos::View<ScalarT**,HostDevice> cdata = mesh_data->getData(cnode);
          for (size_type i=0; i<cdata.extent(1); i++) {
            boundary_groups[block][grp]->data(c,i) = cdata(0,i);
          }
          boundary_groups[block][grp]->group_data->have_extra_data = true;
          boundary_groups[block][grp]->group_data->have_rotation = mesh->have_rotations;
          boundary_groups[block][grp]->group_data->have_phi = mesh->have_rotation_phi;
          
          boundary_groups[block][grp]->data_seed[c] = cnode;
          boundary_groups[block][grp]->data_seedindex[c] = cnode % 100;
          boundary_groups[block][grp]->data_distance[c] = distance;
        }
      }
    }
  }
  else {
    mesh_data = Teuchos::rcp(new Data("mesh data", mesh->dimension, mesh_data_pts_file,
                                      mesh_data_file, false));
    
    for (size_t block=0; block<groups.size(); ++block) {
      for (size_t grp=0; grp<groups[block].size(); ++grp) {
        DRV nodes = groups[block][grp]->nodes;
        int numElem = groups[block][grp]->numElem;
        
        auto centers = mesh->getElementCenters(nodes, groups[block][grp]->group_data->cell_topo);
        
        Kokkos::View<ScalarT*, AssemblyDevice> distance("distance",numElem);
        Kokkos::View<int*, CompadreDevice> cnode("cnode",numElem);
        
        mesh_data->findClosestPoint(centers,cnode,distance);
        
        auto distance_mirror = Kokkos::create_mirror_view(distance);
        auto data_mirror = Kokkos::create_mirror_view(groups[block][grp]->data);

        for (int c=0; c<numElem; c++) {
          Kokkos::View<ScalarT**,HostDevice> cdata = mesh_data->getData(cnode(c));

          for (size_t i=0; i<cdata.extent(1); i++) {
            data_mirror(c,i) = cdata(0,i);
          }

          groups[block][grp]->group_data->have_extra_data = true;
          groups[block][grp]->group_data->have_rotation = mesh->have_rotations;
          groups[block][grp]->group_data->have_phi = mesh->have_rotation_phi;
          
          groups[block][grp]->data_seed[c] = cnode(c);
          groups[block][grp]->data_seedindex[c] = cnode(c) % 100;
          groups[block][grp]->data_distance[c] = distance_mirror(c);

        }
        Kokkos::deep_copy(groups[block][grp]->data, data_mirror);
      }
    }
    
    for (size_t block=0; block<boundary_groups.size(); ++block) {
      for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
        DRV nodes = boundary_groups[block][grp]->nodes;
        int numElem = boundary_groups[block][grp]->numElem;
        
        auto centers = mesh->getElementCenters(nodes, boundary_groups[block][grp]->group_data->cell_topo);
        
        Kokkos::View<ScalarT*, AssemblyDevice> distance("distance",numElem);
        Kokkos::View<int*, CompadreDevice> cnode("cnode",numElem);
        
        mesh_data->findClosestPoint(centers,cnode,distance);

        auto distance_mirror = Kokkos::create_mirror_view(distance);
        auto data_mirror = Kokkos::create_mirror_view(boundary_groups[block][grp]->data);
        
        for (int c=0; c<numElem; c++) {
          Kokkos::View<ScalarT**,HostDevice> cdata = mesh_data->getData(cnode(c));

          for (size_t i=0; i<cdata.extent(1); i++) {
            data_mirror(c,i) = cdata(0,i);
          }

          boundary_groups[block][grp]->group_data->have_extra_data = true;
          boundary_groups[block][grp]->group_data->have_rotation = mesh->have_rotations;
          boundary_groups[block][grp]->group_data->have_phi = mesh->have_rotation_phi;
          
          boundary_groups[block][grp]->data_seed[c] = cnode(c);
          boundary_groups[block][grp]->data_seedindex[c] = cnode(c) % 50;
          boundary_groups[block][grp]->data_distance[c] = distance_mirror(c);
        }
        Kokkos::deep_copy(boundary_groups[block][grp]->data, data_mirror);
      }
    }
  }
  
  
  meshimporttimer.stop();
  if (verbosity>5 && comm->getRank() == 0) {
    cout << "mesh data import time: " << meshimporttimer.totalElapsedTime(false) << endl;
  }
  
  debugger->print("**** Finished AssemblyManager::meshDataImport");
  
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::importQuadratureData() {
  
  debugger->print("**** Starting AssemblyManager::importQuadratureData ...");
  
  Teuchos::Time meshimporttimer("mesh import", false);
  meshimporttimer.start();
  
  
  for (size_t block=0; block<groups.size(); ++block) {
    int numdata = disc->numip[block];    
    for (size_t grp=0; grp<groups[block].size(); ++grp) {
      int numElem = groups[block][grp]->numElem;
      Kokkos::View<ScalarT**,AssemblyDevice> cell_data("cell_data",numElem,numdata);
      groups[block][grp]->data = cell_data;
      //groups[block][grp]->data_distance = vector<ScalarT>(numElem);
      //groups[block][grp]->data_seed = vector<size_t>(numElem);
      //groups[block][grp]->data_seedindex = vector<size_t>(numElem);
    }
  }
  /*
  for (size_t block=0; block<boundary_groups.size(); ++block) {
    int numdata = disc->numip_side[block];
    for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
      int numElem = boundary_groups[block][grp]->numElem;
      Kokkos::View<ScalarT**,AssemblyDevice> cell_data("cell_data",numElem,numdata);
      boundary_groups[block][grp]->data = cell_data;
      boundary_groups[block][grp]->data_distance = vector<ScalarT>(numElem);
      boundary_groups[block][grp]->data_seed = vector<size_t>(numElem);
      boundary_groups[block][grp]->data_seedindex = vector<size_t>(numElem);
    }
  }
  */
  
  string mesh_data_pts_file = mesh->mesh_data_pts_tag + ".dat";
  string mesh_data_file = mesh->mesh_data_tag + ".dat";
  
  Teuchos::RCP<Data> qpt_data = Teuchos::rcp(new Data("mesh data", mesh->dimension, mesh_data_pts_file,
                                                      mesh_data_file, false));
  for (size_t block=0; block<groups.size(); ++block) {
    int numip = disc->numip[block];
    for (size_t grp=0; grp<groups[block].size(); ++grp) {
      DRV nodes = groups[block][grp]->nodes;
      int numElem = groups[block][grp]->numElem;
      auto qpts_vec = groups[block][grp]->getIntegrationPts();
      View_Sc2 qpts("quadrature pts", numElem*numip, mesh->dimension);
      auto qpts_mirror = Kokkos::create_mirror_view(qpts);
      
      for (int dim=0; dim<mesh->dimension; ++dim) {
        auto pt_mirror = Kokkos::create_mirror_view(qpts_vec[dim]);
        Kokkos::deep_copy(pt_mirror, qpts_vec[dim]);
        
        int prog = 0;
        for (int elem=0; elem<numElem; elem++) {
          for (int pt=0; pt<numip; pt++) {
            qpts_mirror(prog,dim) = pt_mirror(elem, pt);
            prog++;
          }
        }
      }
      Kokkos::deep_copy(qpts, qpts_mirror);
      Kokkos::View<ScalarT*, AssemblyDevice> distance("distance", numElem*numip);
      Kokkos::View<int*, CompadreDevice> cnode("cnode", numElem*numip);
      
      qpt_data->findClosestPoint(qpts,cnode,distance);
      
      auto data_mirror = Kokkos::create_mirror_view(groups[block][grp]->data);
      
      int prog = 0;
      for (int elem=0; elem<numElem; elem++) {
        for (int pt=0; pt<numip; pt++) {
          Kokkos::View<ScalarT**,HostDevice> cdata = qpt_data->getData(cnode(prog));
          data_mirror(elem,pt) = cdata(0,0);
          prog++;
        }
      }
      groups[block][grp]->group_data->have_quadrature_data = true;
      
      Kokkos::deep_copy(groups[block][grp]->data, data_mirror);
    }
  }
  
    // Mp
    /*
  for (size_t block=0; block<boundary_groups.size(); ++block) {
    for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
      DRV nodes = boundary_groups[block][grp]->nodes;
      int numElem = boundary_groups[block][grp]->numElem;
      
      auto centers = mesh->getElementCenters(nodes, boundary_groups[block][grp]->group_data->cell_topo);
      
      Kokkos::View<ScalarT*, AssemblyDevice> distance("distance",numElem);
      Kokkos::View<int*, CompadreDevice> cnode("cnode",numElem);
      
      mesh_data->findClosestPoint(centers,cnode,distance);
      
      auto distance_mirror = Kokkos::create_mirror_view(distance);
      auto data_mirror = Kokkos::create_mirror_view(boundary_groups[block][grp]->data);
      
      for (int c=0; c<numElem; c++) {
        Kokkos::View<ScalarT**,HostDevice> cdata = mesh_data->getData(cnode(c));
        
        for (size_t i=0; i<cdata.extent(1); i++) {
          data_mirror(c,i) = cdata(0,i);
        }
        
        boundary_groups[block][grp]->group_data->have_extra_data = true;
        boundary_groups[block][grp]->group_data->have_rotation = mesh->have_rotations;
        boundary_groups[block][grp]->group_data->have_phi = mesh->have_rotation_phi;
        
        boundary_groups[block][grp]->data_seed[c] = cnode(c);
        boundary_groups[block][grp]->data_seedindex[c] = cnode(c) % 50;
        boundary_groups[block][grp]->data_distance[c] = distance_mirror(c);
      }
      Kokkos::deep_copy(boundary_groups[block][grp]->data, data_mirror);
    }
  }*/
  
  
  
  meshimporttimer.stop();
  if (verbosity>5 && comm->getRank() == 0) {
    cout << "mesh data import time: " << meshimporttimer.totalElapsedTime(false) << endl;
  }
  
  debugger->print("**** Finished AssemblyManager::meshDataImport");
  
}


// ========================================================================================
// ========================================================================================

template<class Node>
View_Sc2 AssemblyManager<Node>::getQuadratureData(string & block) {
  
  debugger->print("**** Starting AssemblyManager::getQuadratureData");
  
  int dimension = mesh->dimension;
  
  size_t blockindex = 0;
  for (size_t blk=0; blk<blocknames.size(); ++blk) {
    if (blocknames[blk] == block) {
      blockindex = blk;
    }
  }
  size_type totalip = 0;
  for (size_t grp=0; grp<groups[blockindex].size(); ++grp) {
    auto wts = groups[blockindex][grp]->getWts();
    totalip += wts.extent(0)*wts.extent(1); // stored as [elem,pt]
  }
  View_Sc2 qdata("quadrature data", totalip, dimension+1);
    
  size_t prog = 0;
  for (size_t grp=0; grp<groups[blockindex].size(); ++grp) {
    // These might be stored as compressed views, so use this functionality to get actual data
    View_Sc2 wts = groups[blockindex][grp]->getWts();
    vector<View_Sc2> ip = groups[blockindex][grp]->getIntegrationPts();
    for (size_t elem=0; elem<wts.extent(0); ++elem) {
      for (size_t pt=0; pt<wts.extent(1); ++pt) {
        for (size_t dim=0; dim<dimension; ++dim) {
          qdata(prog,dim) = ip[dim](elem,pt);
        }
        qdata(prog,dimension) = wts(elem,pt);
        ++prog;
      }
    }
  }
  
  debugger->print("**** Finished AssemblyManager::getQuadratureData");
  
  return qdata;
}

// ========================================================================================
// ========================================================================================

template<class Node>
View_Sc2 AssemblyManager<Node>::getboundaryQuadratureData(string & block, string & sidename) {
  
  int dimension = mesh->dimension;
  
  size_t blockindex = 0;
  for (size_t blk=0; blk<blocknames.size(); ++blk) {
    if (blocknames[blk] == block) {
      blockindex = blk;
    }
  }
  
  size_type totalip = 0;
  for (size_t grp=0; grp<boundary_groups[blockindex].size(); ++grp) {
    if (boundary_groups[blockindex][grp]->sidename == sidename) {
      auto wts = boundary_groups[blockindex][grp]->wts; // not compressed
      totalip += wts.extent(0)*wts.extent(1); // stored as [elem,pt]
    }
  }
  View_Sc2 qdata("quadrature data", totalip, 2*dimension+1); // [ip wts normals]
  
  size_t prog = 0;
  for (size_t grp=0; grp<boundary_groups[blockindex].size(); ++grp) {
    if (boundary_groups[blockindex][grp]->sidename == sidename) {
      // These are never stored as compressed views, so just grab the views
      View_Sc2 wts = boundary_groups[blockindex][grp]->wts;
      vector<View_Sc2> ip = boundary_groups[blockindex][grp]->ip;
      vector<View_Sc2> normals = boundary_groups[blockindex][grp]->normals;
      for (size_t elem=0; elem<wts.extent(0); ++elem) {
        for (size_t pt=0; pt<wts.extent(1); ++pt) {
          for (size_t dim=0; dim<dimension; ++dim) {
            qdata(prog,dim) = ip[dim](elem,pt);
          }
          qdata(prog,dimension) = wts(elem,pt);
          for (size_t dim=0; dim<dimension; ++dim) {
            qdata(prog,dimension+dim+1) = normals[dim](elem,pt);
          }
          ++prog;
        }
      }
    }
  }
  return qdata;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::importNewMicrostructure(int & randSeed, View_Sc2 seeds) {
  
  debugger->print("**** Starting AssemblyManager::importNewMicrostructure ...");
  
  Teuchos::Time meshimporttimer("mesh import", false);
  meshimporttimer.start();
  
  std::default_random_engine generator(randSeed);
  
  size_type num_seeds = seeds.extent(0);
  std::uniform_int_distribution<int> idistribution(0,100);
  Kokkos::View<int*,HostDevice> seedIndex("seed index",num_seeds);
  for (size_type i=0; i<num_seeds; i++) {
    int ci = idistribution(generator);
    seedIndex(i) = ci;
  }
  
  //KokkosTools::print(seedIndex);
  
  ////////////////////////////////////////////////////////////////////////////////
  // Set seed data
  ////////////////////////////////////////////////////////////////////////////////
  
  int numdata = 9;
  
  std::normal_distribution<ScalarT> ndistribution(0.0,1.0);
  Kokkos::View<ScalarT**,HostDevice> rotation_data("cell_data",num_seeds,numdata);
  for (size_type k=0; k<num_seeds; k++) {
    ScalarT x = ndistribution(generator);
    ScalarT y = ndistribution(generator);
    ScalarT z = ndistribution(generator);
    ScalarT w = ndistribution(generator);
    
    ScalarT r = sqrt(x*x + y*y + z*z + w*w);
    x *= 1.0/r;
    y *= 1.0/r;
    z *= 1.0/r;
    w *= 1.0/r;
    
    rotation_data(k,0) = w*w + x*x - y*y - z*z;
    rotation_data(k,1) = 2.0*(x*y - w*z);
    rotation_data(k,2) = 2.0*(x*z + w*y);
    
    rotation_data(k,3) = 2.0*(x*y + w*z);
    rotation_data(k,4) = w*w - x*x + y*y - z*z;
    rotation_data(k,5) = 2.0*(y*z - w*x);
    
    rotation_data(k,6) = 2.0*(x*z - w*y);
    rotation_data(k,7) = 2.0*(y*z + w*x);
    rotation_data(k,8) = w*w - x*x - y*y + z*z;
    
  }
  
  //KokkosTools::print(rotation_data);
  
  ////////////////////////////////////////////////////////////////////////////////
  // Initialize cell data
  ////////////////////////////////////////////////////////////////////////////////
  
  int totalElem = 0;
  for (size_t block=0; block<groups.size(); ++block) {
    for (size_t grp=0; grp<groups[block].size(); ++grp) {
      int numElem = groups[block][grp]->numElem;
      totalElem += numElem;
      Kokkos::View<ScalarT**,AssemblyDevice> cell_data("cell_data",numElem,numdata);
      groups[block][grp]->data = cell_data;
      groups[block][grp]->data_distance = vector<ScalarT>(numElem);
      groups[block][grp]->data_seed = vector<size_t>(numElem);
      groups[block][grp]->data_seedindex = vector<size_t>(numElem);
    }
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  // Create a list of all cell nodes
  ////////////////////////////////////////////////////////////////////////////////
  
  DRV tmpnodes = disc->getMyNodes(0, groups[0][0]->localElemID);
  DRV totalNodes("nodes from all groups",totalElem,
                 tmpnodes.extent(1),
                 tmpnodes.extent(2));
  int prog = 0;
  for (size_t block=0; block<groups.size(); ++block) {
    for (size_t grp=0; grp<groups[block].size(); ++grp) {
      DRV nodes = disc->getMyNodes(block, groups[block][grp]->localElemID);
      
      parallel_for("mesh data cell nodes",
                   RangePolicy<AssemblyExec>(0,nodes.extent(0)),
                   KOKKOS_CLASS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<nodes.extent(1); ++pt) {
          for (size_type dim=0; dim<nodes.extent(2); ++dim) {
            totalNodes(prog+elem,pt,dim) = nodes(elem,pt,dim);
          }
        }
      });
      prog += groups[block][grp]->numElem;
    }
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  // Create a list of all cell centers
  ////////////////////////////////////////////////////////////////////////////////
  
  auto centers = mesh->getElementCenters(totalNodes, groups[0][0]->group_data->cell_topo);
  
  ////////////////////////////////////////////////////////////////////////////////
  // Find the closest seeds
  ////////////////////////////////////////////////////////////////////////////////
  
  Kokkos::View<ScalarT*, AssemblyDevice> distance("distance",totalElem);
  Kokkos::View<int*, CompadreDevice> cnode("cnode",totalElem);
  
  Compadre::NeighborLists<Kokkos::View<int*, CompadreDevice> > neighborlists = CompadreInterface_constructNeighborLists(seeds, centers, distance);
  cnode = neighborlists.getNeighborLists();
  
  ////////////////////////////////////////////////////////////////////////////////
  // Set group data
  ////////////////////////////////////////////////////////////////////////////////
  
  prog = 0;
  for (size_t block=0; block<groups.size(); ++block) {
    for (size_t grp=0; grp<groups[block].size(); ++grp) {
      int numElem = groups[block][grp]->numElem;
      //auto centers = this->getElementCenters(nodes, groups[block][grp]->group_data->cellTopo);
      
      //Kokkos::View<ScalarT*, AssemblyDevice> distance("distance",numElem);
      //Kokkos::View<int*, AssemblyDevice> cnode("cnode",numElem);
      //Compadre::NeighborLists<Kokkos::View<int*> > neighborlists = CompadreTools_constructNeighborLists(seeds, centers, distance);
      //cnode = neighborlists.getNeighborLists();

      for (int c=0; c<numElem; c++) {
        
        int cpt = cnode(prog);
        prog++;
        
        for (int i=0; i<9; i++) {
          groups[block][grp]->data(c,i) = rotation_data(cpt,i);//rotation_data(cnode(c),i);
        }
        
        groups[block][grp]->group_data->have_rotation = true;
        groups[block][grp]->group_data->have_phi = false;
        
        groups[block][grp]->data_seed[c] = cpt % 100;//cnode(c) % 100;
        groups[block][grp]->data_seedindex[c] = seedIndex(cpt); //seedIndex(cnode(c));
        groups[block][grp]->data_distance[c] = distance(cpt);//distance(c);
        
      }
    }
    
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  // Initialize boundary data
  ////////////////////////////////////////////////////////////////////////////////
  
  totalElem = 0;
  for (size_t block=0; block<boundary_groups.size(); ++block) {
    for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
      int numElem = boundary_groups[block][grp]->numElem;
      totalElem += numElem;
      Kokkos::View<ScalarT**,AssemblyDevice> cell_data("cell_data",numElem,numdata);
      boundary_groups[block][grp]->data = cell_data;
      boundary_groups[block][grp]->data_distance = vector<ScalarT>(numElem);
      boundary_groups[block][grp]->data_seed = vector<size_t>(numElem);
      boundary_groups[block][grp]->data_seedindex = vector<size_t>(numElem);
    }
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  // Create a list of all cell nodes
  ////////////////////////////////////////////////////////////////////////////////
  
  if (totalElem > 0) {
    
    DRV tmpnodes = disc->getMyNodes(0, groups[0][0]->localElemID);
    totalNodes = DRV("nodes from all groups",totalElem,
                     tmpnodes.extent(1),
                     tmpnodes.extent(2));
    prog = 0;
    for (size_t block=0; block<boundary_groups.size(); ++block) {
      for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
        DRV nodes = disc->getMyNodes(block, boundary_groups[block][grp]->localElemID);
        parallel_for("mesh data cell nodes",
                     RangePolicy<AssemblyExec>(0,nodes.extent(0)),
                     KOKKOS_CLASS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<nodes.extent(1); ++pt) {
            for (size_type dim=0; dim<nodes.extent(2); ++dim) {
              totalNodes(prog+elem,pt,dim) = nodes(elem,pt,dim);
            }
          }
        });
        prog += boundary_groups[block][grp]->numElem;
      }
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    // Create a list of all cell centers
    ////////////////////////////////////////////////////////////////////////////////
    
    centers = mesh->getElementCenters(totalNodes, groups[0][0]->group_data->cell_topo);
    
    ////////////////////////////////////////////////////////////////////////////////
    // Find the closest seeds
    ////////////////////////////////////////////////////////////////////////////////
    
    distance = Kokkos::View<ScalarT*, AssemblyDevice>("distance",totalElem);
    cnode = Kokkos::View<int*, CompadreDevice>("cnode",totalElem);
    neighborlists = CompadreInterface_constructNeighborLists(seeds, centers, distance);
    cnode = neighborlists.getNeighborLists();
    
    ////////////////////////////////////////////////////////////////////////////////
    // Set data
    ////////////////////////////////////////////////////////////////////////////////
    
    prog = 0;
    for (size_t block=0; block<boundary_groups.size(); ++block) {
      for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
        DRV nodes = disc->getMyNodes(block, boundary_groups[block][grp]->localElemID);
        int numElem = boundary_groups[block][grp]->numElem;
        
        for (int c=0; c<numElem; c++) {
          
          int cpt = cnode(prog);
          prog++;
          
          for (int i=0; i<9; i++) {
            boundary_groups[block][grp]->data(c,i) = rotation_data(cpt,i);
          }
          
          boundary_groups[block][grp]->group_data->have_rotation = true;
          boundary_groups[block][grp]->group_data->have_phi = false;
          
          boundary_groups[block][grp]->data_seed[c] = cpt % 100;
          boundary_groups[block][grp]->data_seedindex[c] = seedIndex(cpt);
          boundary_groups[block][grp]->data_distance[c] = distance(cpt);
          
        }
      }
    }
    
  }
  
  meshimporttimer.stop();
  if (verbosity>5 && comm->getRank() == 0) {
    cout << "microstructure import time: " << meshimporttimer.totalElapsedTime(false) << endl;
  }
  
  debugger->print("**** Finished AssemblyManager::importNewMicrostructure");
  
}

/////////////////////////////////////////////////////////////////////////////////////////////
// After the setup phase, we can get rid of a few things
/////////////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::purgeMemory() {
  // nothing here
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::computeFlux(const int & block, const int & grp,
                                        View_Sc2 u_sub,
                                        View_Sc2 du_sub,
                                        View_Sc2 dp_sub,
                                        View_Sc3 lambda,
                                        const ScalarT & time, const int & side, const ScalarT & coarse_h,
                                        const bool & compute_sens, const ScalarT & fluxwt,
                                        bool & useTransientSol) {
  
#ifndef MrHyDE_NO_AD
  typedef Kokkos::View<AD**,ContLayout,AssemblyDevice> View_AD2;
  int wkblock = 0;
  
  wkset_AD[wkblock]->setTime(time);
  wkset_AD[wkblock]->sidename = boundary_groups[block][grp]->sidename;
  wkset_AD[wkblock]->currentside = boundary_groups[block][grp]->sidenum;
  wkset_AD[wkblock]->numElem = boundary_groups[block][grp]->numElem;
  
  // Currently hard coded to one physics sets
  int set = 0;
  
  vector<View_AD2> sol_vals = wkset_AD[wkblock]->sol_vals;
  //auto param_AD = wkset_AD->pvals;
  //auto ulocal = groupData[block]->sol[set];
  auto ulocal = boundary_groups[block][grp]->sol[set];
  auto currLIDs = boundary_groups[block][grp]->LIDs[set];

  if (useTransientSol) {
    int stage = wkset_AD[wkblock]->current_stage;
    auto b_A = wkset_AD[wkblock]->butcher_A;
    auto b_b = wkset_AD[wkblock]->butcher_b;
    auto BDF = wkset_AD[wkblock]->BDF_wts;
    
    ScalarT one = 1.0;
    
    for (size_type var=0; var<ulocal.extent(1); var++ ) {
      size_t uindex = wkset_AD[wkblock]->sol_vals_index[set][var];
      auto u_AD = sol_vals[uindex];
      auto off = subview(wkset_AD[wkblock]->set_offsets[set],var,ALL());
      auto cu = subview(ulocal,ALL(),var,ALL());
      //auto cu_prev = subview(groupData[block]->sol_prev[set],ALL(),var,ALL(),ALL());
      //auto cu_stage = subview(groupData[block]->sol_stage[set],ALL(),var,ALL(),ALL());
      
      auto cu_prev = subview(boundary_groups[block][grp]->sol_prev[set],ALL(),var,ALL(),ALL());
      auto cu_stage = subview(boundary_groups[block][grp]->sol_stage[set],ALL(),var,ALL(),ALL());
      
      parallel_for("wkset transient sol seedwhat 1",
                   TeamPolicy<AssemblyExec>(currLIDs.extent(0), Kokkos::AUTO, VECTORSIZE),
                   KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        ScalarT beta_u;//, beta_t;
        ScalarT alpha_u = b_A(stage,stage)/b_b(stage);
        //ScalarT timewt = one/dt/b_b(stage);
        //ScalarT alpha_t = BDF(0)*timewt;
        
        for (size_type dof=team.team_rank(); dof<u_AD.extent(1); dof+=team.team_size() ) {
          
          // Seed the stage solution
          AD stageval = AD(MAXDERIVS,0,cu(elem,dof));
          for( size_t p=0; p<du_sub.extent(1); p++ ) {
            stageval.fastAccessDx(p) = fluxwt*du_sub(currLIDs(elem,off(dof)),p);
          }
          // Compute the evaluating solution
          beta_u = (one-alpha_u)*cu_prev(elem,dof,0);
          for (int s=0; s<stage; s++) {
            beta_u += b_A(stage,s)/b_b(s) * (cu_stage(elem,dof,s) - cu_prev(elem,dof,0));
          }
          u_AD(elem,dof) = alpha_u*stageval+beta_u;
          
          // Compute the time derivative
          //beta_t = zero;
          //for (size_type s=1; s<BDF.extent(0); s++) {
          //  beta_t += BDF(s)*cu_prev(elem,dof,s-1);
          //}
          //beta_t *= timewt;
          //u_dot_AD(elem,dof) = alpha_t*stageval + beta_t;
        }
        
      });
      
    }
  }
  else {
    //Teuchos::TimeMonitor localtimer(*fluxGatherTimer);
    
    if (compute_sens) {
      for (size_t var=0; var<ulocal.extent(1); var++) {
        auto u_AD = sol_vals[var];
        auto offsets = subview(wkset_AD[wkblock]->offsets,var,ALL());
        parallel_for("flux gather",
                     RangePolicy<AssemblyExec>(0,ulocal.extent(0)),
                     KOKKOS_CLASS_LAMBDA (const int elem ) {
          for( size_t dof=0; dof<u_AD.extent(1); dof++ ) {
            u_AD(elem,dof) = AD(u_sub(currLIDs(elem,offsets(dof)),0));
          }
        });
      }
    }
    else {
      for (size_t var=0; var<ulocal.extent(1); var++) {
        auto u_AD = sol_vals[var];
        auto offsets = subview(wkset_AD[wkblock]->offsets,var,ALL());
        parallel_for("flux gather",
                     RangePolicy<AssemblyExec>(0,ulocal.extent(0)),
                     KOKKOS_CLASS_LAMBDA (const int elem ) {
          for( size_t dof=0; dof<u_AD.extent(1); dof++ ) {
            u_AD(elem,dof) = AD(MAXDERIVS, 0, u_sub(currLIDs(elem,offsets(dof)),0));
            for( size_t p=0; p<du_sub.extent(1); p++ ) {
              u_AD(elem,dof).fastAccessDx(p) = du_sub(currLIDs(elem,offsets(dof)),p);
            }
          }
        });
      }
    }
  }
  
  {
    //Teuchos::TimeMonitor localtimer(*fluxWksetTimer);
    wkset_AD[wkblock]->computeSolnSideIP(boundary_groups[block][grp]->sidenum);//, u_AD, param_AD);
  }
  
  if (wkset_AD[wkblock]->numAux > 0) {
    
    // Teuchos::TimeMonitor localtimer(*fluxAuxTimer);
    
    auto numAuxDOF = groupData[wkblock]->num_aux_dof;
    
    for (size_type var=0; var<numAuxDOF.extent(0); var++) {
      auto abasis = boundary_groups[block][grp]->auxside_basis[boundary_groups[block][grp]->auxusebasis[var]];
      auto off = subview(boundary_groups[block][grp]->auxoffsets,var,ALL());
      string varname = wkset_AD[wkblock]->aux_varlist[var];
      auto local_aux = wkset_AD[wkblock]->getSolutionField("aux "+varname,false);
      Kokkos::deep_copy(local_aux,0.0);
      //auto local_aux = Kokkos::subview(wkset_AD->local_aux_side,Kokkos::ALL(),var,Kokkos::ALL(),0);
      auto localID = boundary_groups[block][grp]->localElemID;
      auto varaux = subview(lambda,ALL(),var,ALL());
      parallel_for("flux aux",
                   RangePolicy<AssemblyExec>(0,localID.extent(0)),
                   KOKKOS_CLASS_LAMBDA (const size_type elem ) {
        for (size_type dof=0; dof<abasis.extent(1); ++dof) {
          AD auxval = AD(MAXDERIVS,off(dof), varaux(localID(elem),dof));
          auxval.fastAccessDx(off(dof)) *= fluxwt;
          for (size_type pt=0; pt<abasis.extent(2); ++pt) {
            local_aux(elem,pt) += auxval*abasis(elem,dof,pt);
          }
        }
      });
    }
    
  }
  
  {
    //Teuchos::TimeMonitor localtimer(*fluxEvalTimer);
    physics->computeFlux<AD>(0,groupData[block]->my_block);
  }
#endif
  //wkset_AD->isOnSide = false;
}

