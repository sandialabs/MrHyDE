/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
************************************************************************/

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::buildDatabase(const size_t & block) {
  
  vector<std::pair<size_t,size_t> > first_users, first_users_x, first_users_y, first_users_z; // stores <grpID,elemID>
  vector<std::pair<size_t,size_t> > first_boundary_users; // stores <grpID,elemID>
  
  /////////////////////////////////////////////////////////////////////////////
  // Step 1: identify the duplicate information
  /////////////////////////////////////////////////////////////////////////////
  
  this->identifyVolumetricDatabase(block, first_users);
  
  this->identifyBoundaryDatabase(block, first_boundary_users);
  
  if (groupData[block]->use_ip_database) {
    this->identifyVolumetricIPDatabase(block, first_users_x, first_users_y, first_users_z);
  }

  /////////////////////////////////////////////////////////////////////////////
  // Step 2: inform the user about the savings
  /////////////////////////////////////////////////////////////////////////////
  
  size_t totalelem = 0, boundaryelem = 0;
  for (size_t grp=0; grp<groups[block].size(); ++grp) {
    totalelem += groups[block][grp]->numElem;
  }
  for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
    boundaryelem += boundary_groups[block][grp]->numElem;
  }
  
  if (verbosity > 5) {
    cout << " - Processor " << comm->getRank() << ": Number of elements on block " << blocknames[block] << ": " << totalelem << endl;
    cout << " - Processor " << comm->getRank() << ": Number of unique elements on block " << blocknames[block] << ": " << first_users.size() << endl;
    cout << " - Processor " << comm->getRank() << ": Database memory savings on " << blocknames[block] << ": "
    << (100.0 - 100.0*((double)first_users.size()/(double)totalelem)) << "%" << endl;
    cout << " - Processor " << comm->getRank() << ": Number of boundary elements on block " << blocknames[block] << ": " << boundaryelem << endl;
    cout << " - Processor " << comm->getRank() << ": Number of unique boundary elements on block " << blocknames[block] << ": " << first_boundary_users.size() << endl;
    cout << " - Processor " << comm->getRank() << ": Database boundary memory savings on " << blocknames[block] << ": "
    << (100.0 - 100.0*((double)first_boundary_users.size()/(double)boundaryelem)) << "%" << endl;
    if (groupData[block]->use_ip_database) {
      int totalip = totalelem*groupData[block]->ref_ip.extent(0);
      cout << " - Processor " << comm->getRank() << ": Number of quadrature points on block " << blocknames[block] << ": " << totalip << endl;
      size_t total_ip_stored = first_users_x.size() + first_users_y.size() + first_users_z.size();
      cout << " - Processor " << comm->getRank() << ": Number of unique quadrature points on block " << blocknames[block] << ": " << total_ip_stored << endl;
      cout << " - Processor " << comm->getRank() << ": Database quadrature memory savings on " << blocknames[block] << ": "
      << (100.0 - 100.0*((double)total_ip_stored/(double)totalip)) << "%" << endl;
    }
  }
  
  /////////////////////////////////////////////////////////////////////////////
  // Step 3: build the database
  /////////////////////////////////////////////////////////////////////////////
  
  this->buildVolumetricDatabase(block, first_users);
  
  this->buildBoundaryDatabase(block, first_boundary_users);
  if (groupData[block]->use_ip_database) {
    this->buildVolumetricIPDatabase(block, first_users_x, first_users_y, first_users_z);
  }
  
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::identifyVolumetricDatabase(const size_t & block, vector<std::pair<size_t,size_t> > & first_users) {
  Teuchos::TimeMonitor localtimer(*group_database_create_timer);
  
  double database_TOL = settings->sublist("Solver").get<double>("database TOL",1.0e-10);
  
  int dimension = groupData[block]->dimension;
  size_type numip = groupData[block]->ref_ip.extent(0);
  bool ignore_orientations = false;//true;
  for (size_t i=0; i<groupData[block]->basis_pointers.size(); ++i) {
    if (groupData[block]->basis_pointers[i]->requireOrientation()) {
      ignore_orientations = false;
    }
  }
  
  vector<Kokkos::View<ScalarT***,HostDevice>> db_jacobians;
  vector<ScalarT> db_measures;
  
  // There are only so many unique orientation
  // Creating a short list of the unique ones and the index for each element
  
  vector<string> unique_orients;
  vector<vector<size_t> > all_orients;
  for (size_t grp=0; grp<groups[block].size(); ++grp) {
    Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation("tmp orients",groups[block][grp]->numElem);
    disc->getPhysicalOrientations(groupData[block], groups[block][grp]->localElemID, orientation, true);
    auto orient_host = create_mirror_view(orientation);
    deep_copy(orient_host, orientation);
    vector<size_t> grp_orient(groups[block][grp]->numElem);
    for (size_t e=0; e<groups[block][grp]->numElem; ++e) {
      string orient = orient_host(e).to_string();
      bool found = false;
      size_t oprog = 0;
      while (!found && oprog<unique_orients.size()) {
        if (orient == unique_orients[oprog]) {
          found = true;
        }
        else {
          ++oprog;
        }
      }
      if (found) {
        grp_orient[e] = oprog;
      }
      else {
        unique_orients.push_back(orient);
        grp_orient[e] = unique_orients.size()-1;
      }
    }
    all_orients.push_back(grp_orient);
  }
  
  // Write data to file (for clustering)
  
  bool write_volumetric_data = settings->sublist("Solver").get<bool>("write volumetric data",false);
  if (write_volumetric_data) {
    this->writeVolumetricData(block, all_orients);
  }
  
  for (size_t grp=0; grp<groups[block].size(); ++grp) {
    groups[block][grp]->storeAll = false;
    Kokkos::View<LO*,AssemblyDevice> index("basis database index",groups[block][grp]->numElem);
    auto index_host = create_mirror_view(index);
    
    // Get the Jacobian for this group
    DRV jacobian("jacobian", groups[block][grp]->numElem, numip, dimension, dimension);
    disc->getJacobian(groupData[block], groups[block][grp]->localElemID, jacobian);
    auto jacobian_host = create_mirror_view(jacobian);
    deep_copy(jacobian_host,jacobian);
    
    // Get the measures for this group
    DRV measure("measure", groups[block][grp]->numElem);
    disc->getMeasure(groupData[block], jacobian, measure);
    auto measure_host = create_mirror_view(measure);
    deep_copy(measure_host,measure);
    auto numElem = groups[block][grp]->numElem;
    bool store_anyway = false;
    if (numElem < 10) {
      store_anyway = true;
    }
    for (size_t e=0; e<groups[block][grp]->numElem; ++e) {
      bool found = false;
      size_t prog = 0;

      //ScalarT refmeas = std::pow(measure_host(e),1.0/dimension);
            
      while (!found && prog<first_users.size()) {
        size_t refgrp = first_users[prog].first;
        size_t refelem = first_users[prog].second;
        
        // Check #1: element orientations
        size_t orient = all_orients[grp][e];
        size_t reforient = all_orients[refgrp][refelem];
        if (ignore_orientations || orient == reforient) {
          
          // Check #2: element measures
          ScalarT diff = std::abs(measure_host(e)-db_measures[prog]);
          
          if (std::abs(diff/db_measures[prog])<database_TOL) { // abs(measure) is probably unnecessary here
            
            // Check #3: element Jacobians
            size_type pt = 0;
            bool ruled_out = false;
            while (pt<numip && !ruled_out) {
              ScalarT fronorm = 0.0;
              ScalarT frodiff = 0.0;
              ScalarT diff = 0.0;
              for (size_type d0=0; d0<jacobian_host.extent(2); ++d0) {
                for (size_type d1=0; d1<jacobian_host.extent(3); ++d1) {
                  diff = jacobian_host(e,pt,d0,d1)-db_jacobians[prog](pt,d0,d1);
                  frodiff += diff*diff;
                  fronorm += jacobian_host(e,pt,d0,d1)*jacobian_host(e,pt,d0,d1);
                }
              }
              if (std::sqrt(frodiff)/std::sqrt(fronorm) > database_TOL) {
                ruled_out = true;
              }
              pt++;
            }
            
            if (!ruled_out) {
              found = true;
              index_host(e) = prog;
            }
            else {
              ++prog;
            }
            
          }
          else {
            ++prog;
          }
        }
        else {
          ++prog;
        }
      }
      if (!found || store_anyway) {
        index_host(e) = first_users.size();
        std::pair<size_t,size_t> newuj{grp,e};
        first_users.push_back(newuj);
        
        Kokkos::View<ScalarT***,HostDevice> new_jac("new db jac",numip, dimension, dimension);
        for (size_type pt=0; pt<new_jac.extent(0); ++pt) {
          for (size_type d0=0; d0<new_jac.extent(1); ++d0) {
            for (size_type d1=0; d1<new_jac.extent(2); ++d1) {
              new_jac(pt,d0,d1) = jacobian(e,pt,d0,d1);
            }
          }
        }
        db_jacobians.push_back(new_jac);
        db_measures.push_back(measure_host(e));
        
      }
    }
    deep_copy(index,index_host);
    groups[block][grp]->basis_index = index;
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::identifyBoundaryDatabase(const size_t & block, vector<std::pair<size_t,size_t> > & first_boundary_users) {
  
  Teuchos::TimeMonitor localtimer(*group_database_create_timer);

  double database_TOL = settings->sublist("Solver").get<double>("database TOL",1.0e-10);
  
  int dimension = groupData[block]->dimension;
  size_type numip = groupData[block]->ref_ip.extent(0);
  
  vector<Kokkos::View<ScalarT***,HostDevice>> db_jacobians;
  vector<ScalarT> db_measures, db_jacobian_norms;
  
  // There are only so many unique orientation
  // Creating a short list of the unique ones and the index for each element
  vector<string> unique_orients;
  vector<vector<size_t> > all_orients;
  for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
    vector<size_t> grp_orient(boundary_groups[block][grp]->numElem);
    Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation("tmp orients",boundary_groups[block][grp]->numElem);
    disc->getPhysicalOrientations(groupData[block], boundary_groups[block][grp]->localElemID, orientation, true);
    auto orient_host = create_mirror_view(orientation);
    deep_copy(orient_host, orientation);
    
    //auto orient_host = create_mirror_view(boundary_groups[block][grp]->orientation);
    //deep_copy(orient_host,boundary_groups[block][grp]->orientation);
    
    for (size_t e=0; e<boundary_groups[block][grp]->numElem; ++e) {
      string orient = orient_host(e).to_string();
      bool found = false;
      size_t oprog = 0;
      while (!found && oprog<unique_orients.size()) {
        if (orient == unique_orients[oprog]) {
          found = true;
        }
        else {
          ++oprog;
        }
      }
      if (found) {
        grp_orient[e] = oprog;
      }
      else {
        unique_orients.push_back(orient);
        grp_orient[e] = unique_orients.size()-1;
      }
    }
    all_orients.push_back(grp_orient);
  }
  
  for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
    boundary_groups[block][grp]->storeAll = false;
    Kokkos::View<LO*,AssemblyDevice> index("basis database index",boundary_groups[block][grp]->numElem);
    auto index_host = create_mirror_view(index);
    
    // Get the Jacobian for this group
    DRV jacobian("jacobian", boundary_groups[block][grp]->numElem, numip, dimension, dimension);
    disc->getJacobian(groupData[block], boundary_groups[block][grp]->localElemID, jacobian);
    auto jacobian_host = create_mirror_view(jacobian);
    deep_copy(jacobian_host,jacobian);
    
    // Get the measures for this group
    DRV measure("measure", boundary_groups[block][grp]->numElem);
    disc->getMeasure(groupData[block], jacobian, measure);
    auto measure_host = create_mirror_view(measure);
    deep_copy(measure_host,measure);
    
    for (size_t e=0; e<boundary_groups[block][grp]->numElem; ++e) {
      LO localSideID = boundary_groups[block][grp]->localSideID;
      LO sidenum = boundary_groups[block][grp]->sidenum;
      bool found = false;
      size_t prog = 0;
      while (!found && prog<first_boundary_users.size()) {
        size_t refgrp = first_boundary_users[prog].first;
        size_t refelem = first_boundary_users[prog].second;
        
        // Check #0: side IDs must match
        if (localSideID == boundary_groups[block][refgrp]->localSideID &&
            sidenum == boundary_groups[block][refgrp]->sidenum) {
          
          // Check #1: element orientations
          
          size_t orient = all_orients[grp][e];
          size_t reforient = all_orients[refgrp][refelem];
          if (orient == reforient) { // if all 3 checks have passed
            
            // Check #2: element measures
            ScalarT diff = std::abs(measure_host(e)-db_measures[prog]);
            //ScalarT refmeas = std::pow(db_measures[prog],1.0/dimension);
            ScalarT refnorm = db_jacobian_norms[prog];

            if (std::abs(diff/db_measures[prog])<database_TOL) {
              
              // Check #3: element Jacobians
              
              ScalarT diff2 = 0.0;
              for (size_type pt=0; pt<jacobian_host.extent(1); ++pt) {
                for (size_type d0=0; d0<jacobian_host.extent(2); ++d0) {
                  for (size_type d1=0; d1<jacobian_host.extent(3); ++d1) {
                    diff2 += std::abs(jacobian_host(e,pt,d0,d1) - db_jacobians[prog](pt,d0,d1));
                  }
                }
              }
              
              if (std::abs(diff2/refnorm)<database_TOL) {
                found = true;
                index_host(e) = prog;
              }
              else {
                ++prog;
              }
              
            }
            else {
              ++prog;
            }
          }
          else {
            ++prog;
          }
        }
        else {
          ++prog;
        }
      }
      if (!found) {
        index_host(e) = first_boundary_users.size();
        std::pair<size_t,size_t> newuj{grp,e};
        first_boundary_users.push_back(newuj);
        
        ScalarT jnorm = 0.0;
        Kokkos::View<ScalarT***,HostDevice> new_jac("new db jac",numip, dimension, dimension);
        for (size_type pt=0; pt<new_jac.extent(0); ++pt) {
          for (size_type d0=0; d0<new_jac.extent(1); ++d0) {
            for (size_type d1=0; d1<new_jac.extent(2); ++d1) {
              new_jac(pt,d0,d1) = jacobian(e,pt,d0,d1);
              jnorm += std::abs(jacobian(e,pt,d0,d1));
            }
          }
        }
        db_jacobians.push_back(new_jac);
        db_measures.push_back(measure_host(e));
        db_jacobian_norms.push_back(jnorm);
        
      }
    }
    deep_copy(index,index_host);
    boundary_groups[block][grp]->basis_index = index;
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::identifyVolumetricIPDatabase(const size_t & block, vector<std::pair<size_t,size_t> > & first_users_x,
                                                         vector<std::pair<size_t,size_t> > & first_users_y,
                                                         vector<std::pair<size_t,size_t> > & first_users_z) {

  Teuchos::TimeMonitor localtimer(*group_database_create_timer);
  
  double database_TOL = settings->sublist("Solver").get<double>("database TOL",1.0e-10);
  
  int dimension = groupData[block]->dimension;
  size_type numip = groupData[block]->ref_ip.extent(0);
  
  vector<vector<ScalarT> > db_x, db_y, db_z;

  bool use_simple_mesh = settings->sublist("Mesh").get<bool>("use simple mesh",false);
  if (use_simple_mesh) {
    if (dimension == 2) {
      int NX = settings->sublist("Mesh").get("NX",20);
      int prog = 0;
      size_t grp = 0;
      while (prog<NX && grp<groups[block].size()) {
        auto elems = groups[block][grp]->localElemID;
        for (size_type e=0; e<elems.size(); ++e) {
          auto el = elems(e);
          if (el<NX) {
            std::pair<size_t,size_t> newuj{grp,e};
            first_users_x.push_back(newuj);
            ++prog;
          }
        }
        ++grp;
      }
      

      int NY = settings->sublist("Mesh").get("NY",20);
      prog = 0;
      grp = 0;
      while (prog<NY && grp<groups[block].size()) {
        auto elems = groups[block][grp]->localElemID;
        if (prog*NX >= elems(0) && elems(elems.extent(0)-1) >= (prog)*NX) {
          for (size_type e=0; e<elems.size(); ++e) {
            auto el = elems(e);
            if (el == prog*NX) {
              std::pair<size_t,size_t> newuj{grp,e};
              first_users_y.push_back(newuj);
              ++prog;
            }
          }
        }
        else {
          ++grp;
        }
      }
      

      for (size_t grp=0; grp<groups[block].size(); ++grp) {
        size_t numElem = groups[block][grp]->numElem;
        auto elemindex = groups[block][grp]->localElemID;
          
        {
          Kokkos::View<LO*,AssemblyDevice> index("ip x database index",numElem);
          for (size_type e=0; e<elemindex.extent(0); ++e) {
            index(e) = elemindex(e) % NX;
          }
          groups[block][grp]->ip_x_index = index;
        }
        {
          Kokkos::View<LO*,AssemblyDevice> index("ip y database index",numElem);
          for (size_type e=0; e<elemindex.extent(0); ++e) {
            index(e) = std::floor(elemindex(e)/NX);
          }
          groups[block][grp]->ip_y_index = index;
        }
      }
    }
  }
  else {
  for (size_t grp=0; grp<groups[block].size(); ++grp) {
    size_t numElem = groups[block][grp]->numElem;
    
    vector<View_Sc2> newip;
    if (groups[block][grp]->have_nodes) {
      disc->getPhysicalIntegrationPts(groupData[block], groups[block][grp]->nodes, newip);
    }
    else {
      disc->getPhysicalIntegrationPts(groupData[block], groups[block][grp]->localElemID, newip);
    }
    
    // Identify database for x
    {
      auto newip_host = create_mirror_view(newip[0]);
      deep_copy(newip_host, newip[0]);
      Kokkos::View<LO*,AssemblyDevice> index("ip x database index",numElem);
      auto index_host = create_mirror_view(index);

      for (size_t e=0; e<groups[block][grp]->numElem; ++e) {
        bool found = false;
        size_t prog = 0;
        
        while (!found && prog<first_users_x.size()) {
          
          // Check #1: x pts
          size_type pt = 0;
          bool ruled_out = false;
          while (pt<numip && !ruled_out) {
            ScalarT frodiff = 0.0;
            ScalarT diff = 0.0;
            diff = newip_host(e,pt)-db_x[prog][pt];
            frodiff += diff*diff;
            if (std::sqrt(frodiff) > database_TOL) {
              ruled_out = true;
            }
            pt++;
          }
            
          if (!ruled_out) {
            found = true;
            index_host(e) = prog;
          }
          else {
            ++prog;
          }
        }
      
        if (!found) {
          index_host(e) = first_users_x.size();
          std::pair<size_t,size_t> newuj{grp,e};
          first_users_x.push_back(newuj);
        
          vector<ScalarT> newpts(numip);
          for (size_t pt=0; pt<numip; ++pt) {
            newpts[pt] = newip_host(e,pt);
          }
          db_x.push_back(newpts);
        }
      }
    
      deep_copy(index,index_host);
      groups[block][grp]->ip_x_index = index;
    }
    
    // Identify database for y
    if (dimension > 1) {
      auto newip_host = create_mirror_view(newip[1]);
      deep_copy(newip_host, newip[1]);
      Kokkos::View<LO*,AssemblyDevice> index("ip y database index",numElem);
      auto index_host = create_mirror_view(index);

      for (size_t e=0; e<groups[block][grp]->numElem; ++e) {
        bool found = false;
        size_t prog = 0;
        
        while (!found && prog<first_users_y.size()) {
          
          // Check #1: y pts
          size_type pt = 0;
          bool ruled_out = false;
          while (pt<numip && !ruled_out) {
            ScalarT frodiff = 0.0;
            ScalarT diff = 0.0;
            diff = newip_host(e,pt)-db_y[prog][pt];
            frodiff += diff*diff;
            if (std::sqrt(frodiff) > database_TOL) {
              ruled_out = true;
            }
            pt++;
          }
            
          if (!ruled_out) {
            found = true;
            index_host(e) = prog;
          }
          else {
            ++prog;
          }
        }
      
        if (!found) {
          index_host(e) = first_users_y.size();
          std::pair<size_t,size_t> newuj{grp,e};
          first_users_y.push_back(newuj);
        
          vector<ScalarT> newpts(numip);
          for (size_t pt=0; pt<numip; ++pt) {
            newpts[pt] = newip_host(e,pt);
          }
          db_y.push_back(newpts);
        }
      }
    
      deep_copy(index,index_host);
      groups[block][grp]->ip_y_index = index;
    }

    // Identify database for z
    if (dimension > 2) {
      auto newip_host = create_mirror_view(newip[2]);
      deep_copy(newip_host, newip[2]);
      Kokkos::View<LO*,AssemblyDevice> index("ip z database index",numElem);
      auto index_host = create_mirror_view(index);

      for (size_t e=0; e<groups[block][grp]->numElem; ++e) {
        bool found = false;
        size_t prog = 0;
        
        while (!found && prog<first_users_z.size()) {
          
          // Check #1: y pts
          size_type pt = 0;
          bool ruled_out = false;
          while (pt<numip && !ruled_out) {
            ScalarT frodiff = 0.0;
            ScalarT diff = 0.0;
            diff = newip_host(e,pt)-db_z[prog][pt];
            frodiff += diff*diff;
            if (std::sqrt(frodiff) > database_TOL) {
              ruled_out = true;
            }
            pt++;
          }
            
          if (!ruled_out) {
            found = true;
            index_host(e) = prog;
          }
          else {
            ++prog;
          }
        }
      
        if (!found) {
          index_host(e) = first_users_z.size();
          std::pair<size_t,size_t> newuj{grp,e};
          first_users_z.push_back(newuj);
        
          vector<ScalarT> newpts(numip);
          for (size_t pt=0; pt<numip; ++pt) {
            newpts[pt] = newip_host(e,pt);
          }
          db_z.push_back(newpts);
        }
      }
    
      deep_copy(index,index_host);
      groups[block][grp]->ip_z_index = index;
    }


  }
  }
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::buildVolumetricDatabase(const size_t & block, vector<std::pair<size_t,size_t> > & first_users) {
  
  Teuchos::TimeMonitor localtimer(*group_database_basis_timer);
  
  using namespace std;

  int dimension = groupData[block]->dimension;
  
  size_t database_numElem = first_users.size();
  DRV database_nodes("nodes for the database",database_numElem, mesh->num_nodes_per_elem, dimension);
  //Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> database_orientation("database orientations",database_numElem);
  View_Sc2 database_wts("physical wts",database_numElem, groupData[block]->ref_ip.extent(0));
  Kokkos::View<LO*,AssemblyDevice> database_ids("database local elem ids", database_numElem);
  //auto database_nodes_host = create_mirror_view(database_nodes);
  auto database_ids_host = create_mirror_view(database_ids);
  //auto database_orientation_host = create_mirror_view(database_orientation);
  auto database_wts_host = create_mirror_view(database_wts);
  
  for (size_t e=0; e<first_users.size(); ++e) {
    size_t refgrp = first_users[e].first;
    size_t refelem = first_users[e].second;
    
    // Get the nodes on the host
    //DRV nodes = mesh->getMyNodes(block, groups[block][refgrp]->localElemID);
    //auto nodes_host = create_mirror_view(nodes);
    //deep_copy(nodes_host, nodes);
    
    //for (size_type node=0; node<database_nodes.extent(1); ++node) {
    //  for (size_type dim=0; dim<database_nodes.extent(2); ++dim) {
    //    database_nodes_host(e,node,dim) = nodes_host(refelem,node,dim);
    //  }
    //}
    
    // Get the orientations on the host
    //auto orientations_host = create_mirror_view(groups[block][refgrp]->orientation);
    //deep_copy(orientations_host, groups[block][refgrp]->orientation);
    //database_orientation_host(e) = orientations_host(refelem);
    
    database_ids_host(e) = groups[block][refgrp]->localElemID(refelem);
    
    // Get the wts on the host
    View_Sc2 twts("temp physical wts",groups[block][refgrp]->numElem, database_wts.extent(1));
    vector<View_Sc2> tmpip;
    disc->getPhysicalIntegrationData(groupData[block], groups[block][refgrp]->localElemID, tmpip, twts);
    
    //auto wts_host = groups[block][refgrp]->wts;
    auto wts_host = create_mirror_view(twts);
    deep_copy(wts_host, twts);
    
    for (size_type pt=0; pt<database_wts_host.extent(1); ++pt) {
      database_wts_host(e,pt) = wts_host(refelem,pt);
    }
    
  }
  
  //deep_copy(database_nodes, database_nodes_host);
  deep_copy(database_ids, database_ids_host);
  //deep_copy(database_orientation, database_orientation_host);
  deep_copy(database_wts, database_wts_host);
  groupData[block]->database_wts = database_wts;
  
  vector<View_Sc4> tbasis, tbasis_grad, tbasis_curl, tbasis_nodes;
  vector<View_Sc3> tbasis_div;
  
  disc->getPhysicalVolumetricBasis(groupData[block], database_ids,
                                   tbasis, tbasis_grad, tbasis_curl,
                                   tbasis_div, tbasis_nodes, true);
  groupData[block]->database_basis = tbasis;
  groupData[block]->database_basis_grad = tbasis_grad;
  groupData[block]->database_basis_div = tbasis_div;
  groupData[block]->database_basis_curl = tbasis_curl;
  
  if (groupData[block]->build_face_terms) {
    for (size_type side=0; side<groupData[block]->num_sides; side++) {
      vector<View_Sc4> face_basis, face_basis_grad;
      
      disc->getPhysicalFaceBasis(groupData[block], side, database_ids,
                                 face_basis, face_basis_grad);
      
      groupData[block]->database_face_basis.push_back(face_basis);
      groupData[block]->database_face_basis_grad.push_back(face_basis_grad);
    }
    
  }
  
  /////////////////////////////////////////////////////////////////////////////
  // Step 3b: database of mass matrices while we have basis and wts
  /////////////////////////////////////////////////////////////////////////////
  
  // Create a database of mass matrices
  if (groupData[block]->use_mass_database) {

    size_t database_numElem = first_users.size();
    for (size_t set=0; set<physics->set_names.size(); ++set) {
      View_Sc3 mass("local mass",database_numElem, groups[block][0]->LIDs[set].extent(1),
                    groups[block][0]->LIDs[set].extent(1));
      
      auto offsets = wkset[block]->set_offsets[set];
      auto numDOF = groupData[block]->set_num_dof[set];
      
      bool use_sparse_quad = settings->sublist("Solver").get<bool>("use sparsifying mass quadrature",false);
      bool include_high_order = false;

      for (size_type n=0; n<numDOF.extent(0); n++) {
        string btype = wkset[block]->basis_types[wkset[block]->set_usebasis[set][n]];
        
        vector<vector<string> > qrules;

        if (btype.substr(0,5) == "HGRAD" || btype.substr(0,4) == "HVOL") {
          // throw an error
        }
        else if (btype.substr(0,4) == "HDIV") {
          //vector<string> qrule1 = {"GAUSS","GAUSS","GAUSS"};
          //qrules.push_back(qrule1);
          //vector<string> qrule2 = {"GAUSS-LOBATTO","GAUSS","GAUSS"};
          //qrules.push_back(qrule2);
          //vector<string> qrule3 = {"GAUSS","GAUSS-LOBATTO","GAUSS"};
          //qrules.push_back(qrule3);
          //vector<string> qrule4 = {"GAUSS","GAUSS","GAUSS-LOBATTO"};
          //qrules.push_back(qrule4);
        }
        else if (btype.substr(0,5) == "HCURL") {
          vector<string> qrule1 = {"GAUSS-LOBATTO","GAUSS-LOBATTO","GAUSS-LOBATTO"};
          qrules.push_back(qrule1);
          //vector<string> qrule2 = {"GAUSS","GAUSS-LOBATTO","GAUSS-LOBATTO"};
          //qrules.push_back(qrule2);
          //vector<string> qrule3 = {"GAUSS-LOBATTO","GAUSS","GAUSS-LOBATTO"};
          //qrules.push_back(qrule3);
          //vector<string> qrule4 = {"GAUSS-LOBATTO","GAUSS-LOBATTO","GAUSS"};
          //qrules.push_back(qrule4);
        }
        else {
         // throw an error
        }

        if (use_sparse_quad && qrules.size() > 0) {
          ScalarT mwt = physics->mass_wts[set][block][n];
          View_Sc4 cbasis = tbasis[wkset[block]->set_usebasis[set][n]];
          
          View_Sc3 mass_sparse("local mass", mass.extent(0), cbasis.extent(1), cbasis.extent(1));
          
          if (include_high_order) {
            auto cwts = database_wts;
          //for (size_type n=0; n<numDOF.extent(0); n++) {
            ScalarT mwt = physics->mass_wts[set][block][n];
            View_Sc4 cbasis = tbasis[wkset[block]->set_usebasis[set][n]];
            string btype = wkset[block]->basis_types[wkset[block]->set_usebasis[set][n]];
            auto off = subview(offsets,n,ALL());
            if (btype.substr(0,5) == "HGRAD" || btype.substr(0,4) == "HVOL") {
              parallel_for("Group get mass",
                           RangePolicy<AssemblyExec>(0,mass.extent(0)),
                           KOKKOS_CLASS_LAMBDA (const size_type e ) {
                for (size_type i=0; i<cbasis.extent(1); i++ ) {
                  for (size_type j=0; j<cbasis.extent(1); j++ ) {
                    for (size_type k=0; k<cbasis.extent(2); k++ ) {
                      mass_sparse(e,off(i),off(j)) += cbasis(e,i,k,0)*cbasis(e,j,k,0)*cwts(e,k)*mwt;
                    }
                  }
                }
              });
            }
            else if (btype.substr(0,4) == "HDIV" || btype.substr(0,5) == "HCURL") {
              parallel_for("Group get mass",
                           RangePolicy<AssemblyExec>(0,mass.extent(0)),
                           KOKKOS_CLASS_LAMBDA (const size_type e ) {
                for (size_type i=0; i<cbasis.extent(1); i++ ) {
                  for (size_type j=0; j<cbasis.extent(1); j++ ) {
                    for (size_type k=0; k<cbasis.extent(2); k++ ) {
                      for (size_type dim=0; dim<cbasis.extent(3); dim++ ) {
                        mass_sparse(e,off(i),off(j)) += cbasis(e,i,k,dim)*cbasis(e,j,k,dim)*cwts(e,k)*mwt;
                      }
                    }
                  }
                }
              });
            }
          //}
          }
          for (size_t q=0; q<qrules.size(); ++q) {
            vector<string> qrule = qrules[q];
            DRV cwts;
            DRV cbasis = disc->evaluateBasisNewQuadrature(groupData[block], block, wkset[block]->set_usebasis[set][n], qrule,
                                                          database_ids, cwts);

            View_Sc3 newmass("local mass", mass.extent(0), cbasis.extent(1), cbasis.extent(1));
              
            parallel_for("Group get mass",
                         RangePolicy<AssemblyExec>(0,mass.extent(0)),
                         KOKKOS_CLASS_LAMBDA (const size_type e ) {
              for (size_type i=0; i<cbasis.extent(1); i++ ) {
                for (size_type j=0; j<cbasis.extent(1); j++ ) {
                  for (size_type k=0; k<cbasis.extent(2); k++ ) {
                    for (size_type dim=0; dim<cbasis.extent(3); dim++ ) {
                      newmass(e,i,j) += cbasis(e,i,k,dim)*cbasis(e,j,k,dim)*cwts(e,k)*mwt;
                    }
                  }
                }
              }
            });
            if (q==0 && !include_high_order) { // first qrule is the base rule
              deep_copy(mass_sparse,newmass);
            }
            else { // see if any alternative rules are better for each entry
              parallel_for("Group get mass",
                           RangePolicy<AssemblyExec>(0,mass.extent(0)),
                           KOKKOS_CLASS_LAMBDA (const size_type e ) {
            
                for (size_type i=0; i<newmass.extent(1); i++ ) {
                  for (size_type j=0; j<newmass.extent(2); j++ ) {
                    if (i==j) {
                      if (newmass(e,i,j) > mass_sparse(e,i,j)) { // not using abs since always positive
                        mass_sparse(e,i,j) = newmass(e,i,j);
                      }
                    }
                    else {
                      if (abs(newmass(e,i,j)) < abs(mass_sparse(e,i,j))) {
                        mass_sparse(e,i,j) = newmass(e,i,j);
                      }
                    }
                  }
                }
              });
            }
          }
          
          // Permute the entries in mass matrix using the offsets
          auto off = subview(offsets,n,ALL());
          parallel_for("Group get mass",
                       RangePolicy<AssemblyExec>(0,mass.extent(0)),
                       KOKKOS_CLASS_LAMBDA (const size_type e ) {
            for (size_type i=0; i<mass_sparse.extent(1); i++ ) {
              for (size_type j=0; j<mass_sparse.extent(2); j++ ) {
                mass(e,off(i),off(j)) = mass_sparse(e,i,j);
              }
            }
          });
        }
      
        else {
          auto cwts = database_wts;
          //for (size_type n=0; n<numDOF.extent(0); n++) {
            ScalarT mwt = physics->mass_wts[set][block][n];
            View_Sc4 cbasis = tbasis[wkset[block]->set_usebasis[set][n]];
            string btype = wkset[block]->basis_types[wkset[block]->set_usebasis[set][n]];
            auto off = subview(offsets,n,ALL());
            if (btype.substr(0,5) == "HGRAD" || btype.substr(0,4) == "HVOL") {
              parallel_for("Group get mass",
                           RangePolicy<AssemblyExec>(0,mass.extent(0)),
                           KOKKOS_CLASS_LAMBDA (const size_type e ) {
                for (size_type i=0; i<cbasis.extent(1); i++ ) {
                  for (size_type j=0; j<cbasis.extent(1); j++ ) {
                    for (size_type k=0; k<cbasis.extent(2); k++ ) {
                      mass(e,off(i),off(j)) += cbasis(e,i,k,0)*cbasis(e,j,k,0)*cwts(e,k)*mwt;
                    }
                  }
                }
              });
            }
            else if (btype.substr(0,4) == "HDIV" || btype.substr(0,5) == "HCURL") {
              parallel_for("Group get mass",
                           RangePolicy<AssemblyExec>(0,mass.extent(0)),
                           KOKKOS_CLASS_LAMBDA (const size_type e ) {
                for (size_type i=0; i<cbasis.extent(1); i++ ) {
                  for (size_type j=0; j<cbasis.extent(1); j++ ) {
                    for (size_type k=0; k<cbasis.extent(2); k++ ) {
                      for (size_type dim=0; dim<cbasis.extent(3); dim++ ) {
                        mass(e,off(i),off(j)) += cbasis(e,i,k,dim)*cbasis(e,j,k,dim)*cwts(e,k)*mwt;
                      }
                    }
                  }
                }
              });
            }
          //}
        }
      }
      
      bool use_sparse = settings->sublist("Solver").get<bool>("sparse mass format",false);
      if (use_sparse) {
        ScalarT tol = settings->sublist("Solver").get<double>("sparse mass TOL",1.0e-10);
        Teuchos::RCP<Sparse3DView> sparse_mass = Teuchos::rcp( new Sparse3DView(mass,tol) );
        groupData[block]->sparse_database_mass.push_back(sparse_mass);
        groupData[block]->use_sparse_mass = true;
        cout << " - Processor " << comm->getRank() << ": Sparse mass format savings on " << blocknames[block] << ": "
             << (100.0 - 100.0*((double)sparse_mass->size()/(double)mass.size())) << "%" << endl;
      }
      //else {
        groupData[block]->database_mass.push_back(mass);
      //}

      bool write_matrices_to_file = false;
      if (write_matrices_to_file) {
        std::stringstream ss;
        ss << set;
        string filename = "mass_matrices." + ss.str() + ".out";
        KokkosTools::printToFile(mass,filename);
      }
    }
  }
}


/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::buildVolumetricIPDatabase(const size_t & block, vector<std::pair<size_t,size_t> > & first_users_x,
                                                      vector<std::pair<size_t,size_t> > & first_users_y,
                                                      vector<std::pair<size_t,size_t> > & first_users_z) {
  
  Teuchos::TimeMonitor localtimer(*group_database_basis_timer);
  
  using namespace std;

  int dimension = groupData[block]->dimension;
  size_type numip = groupData[block]->ref_ip.extent(0);
  
  // Build x ip database
  {
    size_t database_numElem = first_users_x.size();
    Kokkos::View<ScalarT**,AssemblyDevice> database_x("database x ip", database_numElem, numip);
    auto database_x_host = create_mirror_view(database_x);

    for (size_t e=0; e<first_users_x.size(); ++e) {
      size_t refgrp = first_users_x[e].first;
      size_t refelem = first_users_x[e].second;
    
      View_Sc2 twts("temp physical wts",groups[block][refgrp]->numElem, numip);
      vector<View_Sc2> newip;
      disc->getPhysicalIntegrationData(groupData[block], groups[block][refgrp]->localElemID, newip, twts);
      auto newip_x_host = create_mirror_view(newip[0]);
      deep_copy(newip_x_host, newip[0]);
      for (size_type pt=0; pt<newip_x_host.extent(1); ++pt) {
        database_x_host(e,pt) = newip_x_host(refelem,pt);
      }
    }
    deep_copy(database_x, database_x_host);
    groupData[block]->database_x = database_x;
  }

  // Build y ip database
  if (dimension > 1) {
    size_t database_numElem = first_users_y.size();
    Kokkos::View<ScalarT**,AssemblyDevice> database_y("database y ip", database_numElem, numip);
    auto database_y_host = create_mirror_view(database_y);

    for (size_t e=0; e<first_users_y.size(); ++e) {
      size_t refgrp = first_users_y[e].first;
      size_t refelem = first_users_y[e].second;
    
      View_Sc2 twts("temp physical wts",groups[block][refgrp]->numElem, numip);
      vector<View_Sc2> newip;
      disc->getPhysicalIntegrationData(groupData[block], groups[block][refgrp]->localElemID, newip, twts);
      auto newip_y_host = create_mirror_view(newip[1]);
      deep_copy(newip_y_host, newip[1]);
      for (size_type pt=0; pt<newip_y_host.extent(1); ++pt) {
        database_y_host(e,pt) = newip_y_host(refelem,pt);
      }
    }
    deep_copy(database_y, database_y_host);
    groupData[block]->database_y = database_y;
  }

  // Build z ip database
  if (dimension > 2) {
    size_t database_numElem = first_users_z.size();
    Kokkos::View<ScalarT**,AssemblyDevice> database_z("database z ip", database_numElem, numip);
    auto database_z_host = create_mirror_view(database_z);

    for (size_t e=0; e<first_users_z.size(); ++e) {
      size_t refgrp = first_users_z[e].first;
      size_t refelem = first_users_z[e].second;
    
      View_Sc2 twts("temp physical wts",groups[block][refgrp]->numElem, numip);
      vector<View_Sc2> newip;
      disc->getPhysicalIntegrationData(groupData[block], groups[block][refgrp]->localElemID, newip, twts);
      auto newip_z_host = create_mirror_view(newip[2]);
      deep_copy(newip_z_host, newip[2]);
      for (size_type pt=0; pt<newip_z_host.extent(1); ++pt) {
        database_z_host(e,pt) = newip_z_host(refelem,pt);
      }
    }
    deep_copy(database_z, database_z_host);
    groupData[block]->database_z = database_z;
  }
  
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::buildBoundaryDatabase(const size_t & block, vector<std::pair<size_t,size_t> > & first_boundary_users) {
  
  //int dimension = groupData[block]->dimension;
  size_type numsideip = groupData[block]->ref_side_ip[0].extent(0);
  
  size_t database_boundary_numElem = first_boundary_users.size();
  vector<View_Sc4> bbasis, bbasis_grad;
  for (size_t i=0; i<groupData[block]->basis_pointers.size(); i++) {
    int numb = groupData[block]->basis_pointers[i]->getCardinality();
    View_Sc4 basis_vals, basis_grad_vals;
    if (groupData[block]->basis_types[i].substr(0,5) == "HGRAD"){
      basis_vals = View_Sc4("basis vals",database_boundary_numElem,numb,numsideip,1);
      basis_grad_vals = View_Sc4("basis vals",database_boundary_numElem,numb,numsideip,groupData[block]->dimension);
    }
    else if (groupData[block]->basis_types[i].substr(0,4) == "HVOL"){ // does not require orientations
      basis_vals = View_Sc4("basis vals",database_boundary_numElem,numb,numsideip,1);
    }
    else if (groupData[block]->basis_types[i].substr(0,5) == "HFACE"){
      basis_vals = View_Sc4("basis vals",database_boundary_numElem,numb,numsideip,1);
    }
    else if (groupData[block]->basis_types[i].substr(0,4) == "HDIV"){
      basis_vals = View_Sc4("basis vals",database_boundary_numElem,numb,numsideip,groupData[block]->dimension);
    }
    else if (groupData[block]->basis_types[i].substr(0,5) == "HCURL"){
      basis_vals = View_Sc4("basis vals",database_boundary_numElem,numb,numsideip,groupData[block]->dimension);
    }
    bbasis.push_back(basis_vals);
    bbasis_grad.push_back(basis_grad_vals);
  }
  for (size_t e=0; e<first_boundary_users.size(); ++e) {
    size_t refgrp = first_boundary_users[e].first;
    size_t refelem = first_boundary_users[e].second;
    
    Kokkos::View<LO*,AssemblyDevice> database_localID("tmp local elem id",1);
    database_localID(0) = boundary_groups[block][refgrp]->localElemID(refelem);
    //DRV nodes = mesh->getMyNodes(block, boundary_groups[block][refgrp]->localElemID);
    //DRV database_bnodes("nodes for the database", 1, nodes.extent(1), dimension);
    //Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> database_borientation("database orientations", 1);
    
    //auto database_bnodes_host = create_mirror_view(database_bnodes);
    //auto database_borientation_host = create_mirror_view(database_borientation);
    
    LO localSideID = boundary_groups[block][refgrp]->localSideID;
    // Get the nodes on the host
    //auto nodes_host = create_mirror_view(nodes);
    //deep_copy(nodes_host, nodes);
    
    //for (size_type node=0; node<database_bnodes.extent(1); ++node) {
    //  for (size_type dim=0; dim<database_bnodes.extent(2); ++dim) {
    //    database_bnodes_host(0,node,dim) = nodes_host(refelem,node,dim);
    //  }
    //}
    //deep_copy(database_bnodes, database_bnodes_host);
    
    // Get the orientations on the host
    //auto orientations_host = create_mirror_view(boundary_groups[block][refgrp]->orientation);
    //deep_copy(orientations_host, boundary_groups[block][refgrp]->orientation);
    //database_borientation_host(0) = orientations_host(refelem);
    //deep_copy(database_borientation, database_borientation_host);
    
    vector<View_Sc4> tbasis, tbasis_grad, tbasis_curl;
    vector<View_Sc3> tbasis_div;
    disc->getPhysicalBoundaryBasis(groupData[block], database_localID, localSideID,
                                   tbasis, tbasis_grad, tbasis_curl, tbasis_div);
    
    for (size_t i=0; i<groupData[block]->basis_pointers.size(); i++) {
      auto tbasis_slice = subview(tbasis[i],0,ALL(),ALL(),ALL());
      auto bbasis_slice = subview(bbasis[i],e,ALL(),ALL(),ALL());
      deep_copy(bbasis_slice, tbasis_slice);
    }
  }
  
  groupData[block]->database_side_basis = bbasis;
  groupData[block]->database_side_basis_grad = bbasis_grad;
  
  
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::writeVolumetricData(const size_t & block, vector<vector<size_t>> & all_orients) {
  
  vector<vector<ScalarT> > all_meas;
  vector<vector<ScalarT> > all_fros;
  int dimension = groupData[block]->dimension;
  size_type numip = groupData[block]->ref_ip.extent(0);
  
  for (size_t grp=0; grp<groups[block].size(); ++grp) {
    
    // Get the Jacobian for this group
    DRV jacobian("jacobian", groups[block][grp]->numElem, numip, dimension, dimension);
    disc->getJacobian(groupData[block], groups[block][grp]->localElemID, jacobian);
    
    // Get the measures for this group
    DRV measure("measure", groups[block][grp]->numElem);
    disc->getMeasure(groupData[block], jacobian, measure);
    
    DRV fro("fro norm of J", groups[block][grp]->numElem);
    disc->getFrobenius(groupData[block], jacobian, fro);
    vector<ScalarT> currmeas;
    for (size_type e=0; e<measure.extent(0); ++e) {
      currmeas.push_back(measure(e));
      //currmeas.push_back(jacobian(e,0,0,0));
    }
    all_meas.push_back(currmeas);
    
    
    vector<ScalarT> currfro;
    for (size_type e=0; e<fro.extent(0); ++e) {
      ScalarT val = 0.0;
      for (size_type d1=0; d1<jacobian.extent(2); ++d1) {
        for (size_type d2=0; d2<jacobian.extent(3); ++d2) {
          for (size_type pt=0; pt<jacobian.extent(1); ++pt) {
            ScalarT cval = jacobian(e,pt,d1,d2) - jacobian(e,pt,d2,d1);
            val += cval*cval;
          }
        }
      }
      currfro.push_back(val);
      //currfro.push_back(jacobian(e,0,1,1));
    }
    all_fros.push_back(currfro);
  }
  
  if (comm->getRank() == 0) {
    
    string outfile = "jacobian_data.out";
    std::ofstream respOUT(outfile.c_str());
    respOUT.precision(16);
    
    for (size_t grp=0; grp<groups[block].size(); ++grp) {
      // Get the Jacobian for this group
      DRV jac("jacobian", groups[block][grp]->numElem, numip, dimension, dimension);
      disc->getJacobian(groupData[block], groups[block][grp]->localElemID, jac);
      
      DRV wts("jacobian", groups[block][grp]->numElem, numip);
      disc->getPhysicalWts(groupData[block], groups[block][grp]->localElemID, jac, wts);
      
      for (size_t e=0; e<groups[block][grp]->numElem; ++e) {
        /*
        ScalarT j00 = 0.0, j01 = 0.0, j02 = 0.0;
        ScalarT j10 = 0.0, j11 = 0.0, j12 = 0.0;
        ScalarT j20 = 0.0, j21 = 0.0, j22 = 0.0;
        
        for (size_type pt=0; pt<jac.extent(1); ++pt) {
          j00 += jac(e,pt,0,0)*wts(e,pt);
          j01 += jac(e,pt,0,1)*wts(e,pt);
          j02 += jac(e,pt,0,2)*wts(e,pt);
          j10 += jac(e,pt,1,0)*wts(e,pt);
          j11 += jac(e,pt,1,1)*wts(e,pt);
          j12 += jac(e,pt,1,2)*wts(e,pt);
          j20 += jac(e,pt,2,0)*wts(e,pt);
          j21 += jac(e,pt,2,1)*wts(e,pt);
          j22 += jac(e,pt,2,2)*wts(e,pt);
        }
        */

        //respOUT << j00 << ", " << j01 << ", " << j02 << ", " << j10 << ", " << j11 << ", " << j12 << ", " << j20 << ", " << j21 << ", " << j22 << endl;
        respOUT << all_orients[grp][e] << ", " << all_meas[grp][e] << ", " << all_fros[grp][e] << endl;
      }
    }
    respOUT.close();
  }
}


