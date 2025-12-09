/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
************************************************************************/

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
    
void MeshInterface::setupExodusFile(const string & filename) {
  stk_mesh->setupExodusFile(filename);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void MeshInterface::setupOptimizationExodusFile(const string & filename) {
  stk_optimization_mesh->setupExodusFile(filename);
}
    
/////////////////////////////////////////////////////////////////////////////
// Read in discretized data from an exodus mesh
/////////////////////////////////////////////////////////////////////////////

void MeshInterface::readExodusData() {
  
  debugger->print("**** Starting mesh::readExodusData ...");
  
  string exofile;
  string fname;
  
  exofile = settings->sublist("Mesh").get<std::string>("mesh file","mesh.exo");
  
  if (comm->getSize() > 1) {
    std::stringstream ssProc, ssPID;
    ssProc << comm->getSize();
    ssPID << comm->getRank();
    string strProc = ssProc.str();
    string strPID = ssPID.str();
    // this section may need tweaking if the input exodus mesh is
    // spread across 10's, 100's, or 1000's (etc) of processors
    //if (Comm->MyPID() < 10)
    if (false)
      fname = exofile + "." + strProc + ".0" + strPID;
    else
      fname = exofile + "." + strProc + "." + strPID;
  }
  else {
    fname = exofile;
  }
  
  // open exodus file
  int CPU_word_size, IO_word_size, exoid, exo_error;
  int num_dim, num_nods, num_el, num_el_blk, num_ns, num_ss;
  char title[MAX_STR_LENGTH+1];
  float exo_version;
  CPU_word_size = sizeof(ScalarT);
  IO_word_size = 0;
  exoid = ex_open(fname.c_str(), EX_READ, &CPU_word_size,&IO_word_size,
                  &exo_version);
  exo_error = ex_get_init(exoid, title, &num_dim, &num_nods, &num_el,
                          &num_el_blk, &num_ns, &num_ss);
  
  if (exo_error>0) {
    // need some debug statement
  }
  int id = 1; // only one blkid
  int step = 1; // only one time step (for now)
  ex_block eblock;
  eblock.id = id;
  eblock.type = EX_ELEM_BLOCK;
  
  exo_error = ex_get_block_param(exoid, &eblock);
  
  int num_el_in_blk = eblock.num_entry;
  //int num_node_per_el = eblock.num_nodes_per_entry;
  
  
  // get elem vars
  if (settings->sublist("Mesh").get<bool>("have element data", false)) {
    int num_elem_vars = 0;
    int var_ind;
    numResponses = 1;
    //exo_error = ex_get_var_param(exoid, "e", &num_elem_vars); // TMW: this is depracated
    exo_error = ex_get_variable_param(exoid, EX_ELEM_BLOCK, &num_elem_vars); // TMW: this is depracated
    // This turns off this feature
    for (int i=0; i<num_elem_vars; i++) {
      char varname[MAX_STR_LENGTH+1];
      ScalarT *var_vals = new ScalarT[num_el_in_blk];
      var_ind = i+1;
      exo_error = ex_get_variable_name(exoid, EX_ELEM_BLOCK, var_ind, varname);
      string vname(varname);
      efield_names.push_back(vname);
      size_t found = vname.find("Val");
      if (found != std::string::npos) {
        vector<string> results;
        std::stringstream sns, snr;
        int nr;
        results = this->breakupList(vname,"_");
        //boost::split(results, vname, [](char u){return u == '_';});
        snr << results[3];
        snr >> nr;
        numResponses = std::max(numResponses,nr);
      }
      efield_vals.push_back(vector<ScalarT>(num_el_in_blk));
      exo_error = ex_get_var(exoid,step,EX_ELEM_BLOCK,var_ind,id,num_el_in_blk,var_vals);
      for (int j=0; j<num_el_in_blk; j++) {
        efield_vals[i][j] = var_vals[j];
      }
      delete [] var_vals;
    }
  }
  
  /*
  // assign nodal vars to meas multivector
  if (settings->sublist("Mesh").get<bool>("have nodal data", false)) {
    int *connect = new int[num_el_in_blk*num_node_per_el];
    int edgeconn, faceconn;
    //exo_error = ex_get_elem_conn(exoid, id, connect);
    exo_error = ex_get_conn(exoid, EX_ELEM_BLOCK, id, connect, &edgeconn, &faceconn);
    
    // get nodal vars
    int num_node_vars = 0;
    int var_ind;
    //exo_error = ex_get_variable_param(exoid, EX_NODAL, &num_node_vars);
    // This turns off this feature
    for (int i=0; i<num_node_vars; i++) {
      char varname[MAX_STR_LENGTH+1];
      ScalarT *var_vals = new ScalarT[num_nods];
      var_ind = i+1;
      exo_error = ex_get_variable_name(exoid, EX_NODAL, var_ind, varname);
      string vname(varname);
      nfield_names.push_back(vname);
      nfield_vals.push_back(vector<ScalarT>(num_nods));
      exo_error = ex_get_var(exoid,step,EX_NODAL,var_ind,0,num_nods,var_vals);
      for (int j=0; j<num_nods; j++) {
        nfield_vals[i][j] = var_vals[j];
      }
      delete [] var_vals;
    }
    
    
    meas = Teuchos::rcp(new Tpetra::MultiVector<ScalarT,LO,GO,SolverNode>(LA_overlapped_map,1)); // empty solution
    size_t b = 0;
    //meas->sync<HostDevice>();
    auto meas_kv = meas->getLocalView<HostDevice>();
    
    //meas.modify_host();
    int index, dindex;
    
    auto dev_offsets = groups[b][0]->wkset->offsets;
    auto offsets = Kokkos::create_mirror_view(dev_offsets);
    Kokkos::deep_copy(offsets,dev_offsets);
    
    for (size_t grp=0; grp<groups[block].size(); ++grp) {
      //cindex = groups[block][grp]->index;
      auto LIDs = groups[block][grp]->LIDs_host;
      auto nDOF = groups[block][grp]->group_data->numDOF_host;
      
      for (int n=0; n<nDOF(0); n++) {
        //Kokkos::View<GO**,HostDevice> GIDs = assembler->groups[block][grp]->GIDs;
        for (size_t p=0; p<groups[block][grp]->numElem; p++) {
          for( int i=0; i<nDOF(n); i++ ) {
            index = LIDs(p,offsets(n,i));//cindex(p,n,i);//LA_overlapped_map->getLocalElement(GIDs(p,curroffsets[n][i]));
            dindex = connect[e*num_node_per_el + i] - 1;
            meas_kv(index,0) = nfield_vals[n][dindex];
            //(*meas)[0][index] = nfield_vals[n][dindex];
          }
        }
      }
    }
    //meas.sync<>();
    delete [] connect;
    
  }
   */
  exo_error = ex_close(exoid);
  
  debugger->print("**** Finished mesh::readExodusData");
  
}

////////////////////////////////////////////////////////////////////////////////
// Access function (mostly) for the stk mesh
////////////////////////////////////////////////////////////////////////////////
    
void MeshInterface::setSolutionFieldData(string var, string & blockID, vector<size_t> & myElements, Kokkos::View<ScalarT**,HostDevice> soln) {
  if (use_stk_mesh) {
    stk_mesh->setSolutionFieldData(var, blockID, myElements, soln);
  }
}

// ============================================================
// ============================================================

void MeshInterface::setCellFieldData(string var, string & blockID, vector<size_t> & myElements, Kokkos::View<ScalarT*,HostDevice> soln) {
  if (use_stk_mesh) {
    stk_mesh->setCellFieldData(var, blockID, myElements, soln);
  }
}

// ============================================================
// ============================================================

void MeshInterface::setOptimizationSolutionFieldData(string & var, string & blockID, vector<size_t> & myElements, Kokkos::View<ScalarT**,HostDevice> soln) {
  if (use_stk_mesh) {
    stk_optimization_mesh->setSolutionFieldData(var, blockID, myElements, soln);
  }
}

// ============================================================
// ============================================================

void MeshInterface::setOptimizationCellFieldData(string & var, string & blockID, vector<size_t> & myElements, Kokkos::View<ScalarT*,HostDevice> soln) {
  if (use_stk_mesh) {
    stk_optimization_mesh->setCellFieldData(var, blockID, myElements, soln);
  }
}

// ============================================================
// ============================================================

void MeshInterface::writeToExodus(const double & currenttime) {
  if (use_stk_mesh) {
    stk_mesh->writeToExodus(currenttime);
  }
}

// ============================================================
// ============================================================

void MeshInterface::writeToExodus(const string & filename) {
  if (use_stk_mesh) {
    stk_mesh->writeToExodus(filename);
  }
}
 
// ============================================================
// ============================================================

void MeshInterface::writeToOptimizationExodus(const double & currenttime) {
  if (use_stk_mesh) {
    stk_optimization_mesh->writeToExodus(currenttime);
  }
}

// ============================================================
// ============================================================

void MeshInterface::writeToOptimizationExodus(const string & filename) {
  if (use_stk_mesh) {
    stk_optimization_mesh->writeToExodus(filename);
  }
}
