/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

template<class Node>
AssemblyManager<Node>::AssemblyManager(const Teuchos::RCP<MpiComm> & comm_,
                                       Teuchos::RCP<Teuchos::ParameterList> & settings_,
                                       Teuchos::RCP<MeshInterface> & mesh_,
                                       Teuchos::RCP<DiscretizationInterface> & disc_,
                                       Teuchos::RCP<PhysicsInterface> & physics_,
                                       Teuchos::RCP<ParameterManager<Node>> & params_) :
comm(comm_), settings(settings_), mesh(mesh_), disc(disc_), physics(physics_), params(params_) {
  
  RCP<Teuchos::Time> constructor_time = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager - constructor");
  Teuchos::TimeMonitor constructor_timer(*constructor_time);
  
  // Get the required information from the settings
  debugger = Teuchos::rcp(new MrHyDE_Debugger(settings->get<int>("debug level",0), comm));
  debugger->print("**** Starting assembly manager constructor ...");
  
  verbosity = settings->get<int>("verbosity",0);
  usestrongDBCs = settings->sublist("Solver").get<bool>("use strong DBCs",true);
  
  // TMW: the following flag should only be used if there are extra variables, but no corresponding equation/constraint
  fix_zero_rows = settings->sublist("Solver").get<bool>("fix zero rows",false);
  
  // Really, this lumps the Jacobian and should only be used in explicit time integration
  lump_mass = settings->sublist("Solver").get<bool>("lump mass",false);
  matrix_free = settings->sublist("Solver").get<bool>("matrix free",false);
  
  use_meas_as_dbcs = settings->sublist("Mesh").get<bool>("use measurements as DBCs", false);
  
  assembly_partitioning = settings->sublist("Solver").get<string>("assembly partitioning","sequential");
  allow_autotune = settings->sublist("Solver").get<bool>("enable autotune",true);
  store_nodes = settings->sublist("Solver").get<bool>("store nodes",true);
  
  //if (settings->isSublist("Subgrid")) {
  //assembly_partitioning = "subgrid-preserving";
  //}
  
  string solver_type = settings->sublist("Solver").get<string>("solver","none"); // or "transient"
  isTransient = false;
  if (solver_type == "transient") {
    isTransient = true;
  }
  
  // needed information from the mesh
  //blocknames = mesh->block_names;
  blocknames = mesh->getBlockNames();
  
  // check if we need to assembly volumetric, boundary and face terms
  for (size_t set=0; set<physics->set_names.size(); ++set) {
    vector<bool> set_assemble_vol, set_assemble_bndry, set_assemble_face;
    for (size_t block=0; block<blocknames.size(); ++block) {
      set_assemble_vol.push_back(physics->physics_settings[set][block].template get<bool>("assemble volume terms",true));
      set_assemble_bndry.push_back(physics->physics_settings[set][block].template get<bool>("assemble boundary terms",true));
      set_assemble_face.push_back(physics->physics_settings[set][block].template get<bool>("assemble face terms",false));
    }
    assemble_volume_terms.push_back(set_assemble_vol);
    assemble_boundary_terms.push_back(set_assemble_bndry);
    assemble_face_terms.push_back(set_assemble_face);
  }
  // overwrite assemble_face_terms if HFACE vars are used
  for (size_t set=0; set<assemble_face_terms.size(); ++set) {
    for (size_t block=0; block<blocknames.size(); ++block) {
      vector<string> ctypes = physics->unique_types[block];
      for (size_t n=0; n<ctypes.size(); n++) {
        if (ctypes[n] == "HFACE") {
          assemble_face_terms[set][block] = true;
        }
      }
    }
  }
  
  // determine if we need to build basis functions
  for (size_t block=0; block<blocknames.size(); ++block) {
    bool build_volume = false, build_bndry = false, build_face = false;
    
    for (size_t set=0; set<physics->set_names.size(); ++set) {
      
      if (assemble_volume_terms[set][block]) {
        build_volume = true;
      }
      else if (physics->physics_settings[set][block].template get<bool>("build volume terms",true) ) {
        build_volume = true;
      }
      
      if (assemble_boundary_terms[set][block]) {
        build_bndry = true;
      }
      else if (physics->physics_settings[set][block].template get<bool>("build boundary terms",true)) {
        build_bndry = true;
      }
      
      if (assemble_face_terms[set][block]) {
        build_face = true;
      }
      else if (physics->physics_settings[set][block].template get<bool>("build face terms",false)) {
        build_face = true;
      }
    }
    
    build_volume_terms.push_back(build_volume);
    build_boundary_terms.push_back(build_bndry);
    build_face_terms.push_back(build_face);
  }
  
  // needed information from the physics interface
  varlist = physics->var_list;
  
  // Create groups/boundary groups
  this->createGroups();
  
  params->setupDiscretizedParameters(groups, boundary_groups);

  num_derivs_required = disc->num_derivs_required;
  physics->num_derivs_required = disc->num_derivs_required;
  
  // Set the initial AD type to none, will get redefined below
  type_AD = 0;
  int max_ndr = 0;
  for (size_t i=0; i<num_derivs_required.size(); ++i) {
    max_ndr = std::max(max_ndr,num_derivs_required[i]);
  }
  
  if (max_ndr > 0 && max_ndr <= 2 ) {
    type_AD = 2;
  }
  else if (max_ndr > 2 && max_ndr <= 4 ) {
    type_AD = 4;
  }
  else if (max_ndr > 4 && max_ndr <= 8 ) {
    type_AD = 8;
  }
  else if (max_ndr>8 && max_ndr <= 16 ) {
    type_AD = 16;
  }
  else if (max_ndr>16 && max_ndr <= 18 ) {
    type_AD = 18;
  }
  else if (max_ndr>18 && max_ndr <= 24 ) {
    type_AD = 24;
  }
  else if (max_ndr>24 && max_ndr <= 32 ) {
    type_AD = 32;
  }
  else {
    type_AD = -1;
  }
  
  for (size_t block=0; block<blocknames.size(); ++block) {
    if (groupData[block]->multiscale) {
      type_AD = -1;
    }
  }
  if (!allow_autotune) {
    type_AD = -1;
  }
  bool fully_explicit = settings->sublist("Solver").get<bool>("fully explicit",false);
  if (fully_explicit) {
    type_AD = 0;
  }
  physics->importPhysicsAD(type_AD);
  
  this->createConstraints();
  
  this->createFunctions();

  debugger->print("**** Finished assembly manager constructor");
  
  
}

