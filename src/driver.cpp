/***********************************************************************
Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)

Copyright 2018 National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
U.S. Government retains certain rights in this software.”

Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
Bart van Bloemen Waanders (bartv@sandia.gov)
***********************************************************************/

#include "userInterface.hpp"
#include "meshInterface.hpp"
#include "physicsInterface.hpp"
#include "discretizationInterface.hpp"
#include "assemblyManager.hpp"
#include "parameterManager.hpp"
#include "sensorManager.hpp"
#include "multiscaleManager.hpp"
#include "solverInterface.hpp"
#include "postprocessManager.hpp"
#include "analysisInterface.hpp"
#include "trilinos.hpp"
#include "Panzer_DOFManager.hpp"

#include "preferences.hpp"
#include "subgridGenerator.hpp"
#include "milo_help.hpp"
#include "functionManager.hpp"
#include "split_mpi_communicators.hpp"

int main(int argc,char * argv[]) {
  
#ifdef HAVE_MPI
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  Teuchos::RCP<MpiComm> Comm = Teuchos::rcp( new MpiComm(MPI_COMM_WORLD) );
#else
  EPIC_FAIL // MILO requires MPI for HostDevice
#endif
  
  int verbosity = 0;
  bool profile = false;
  
  Kokkos::initialize();
  
  Teuchos::RCP<Teuchos::Time> totalTimer = Teuchos::TimeMonitor::getNewCounter("driver::total setup and execution time");
  Teuchos::RCP<Teuchos::Time> runTimer = Teuchos::TimeMonitor::getNewCounter("driver::total run time");
  
  string input_file_name = "input.yaml";
  if (argc > 1) {
    input_file_name = argv[1];
  }
  if (input_file_name == "--version") {
    cout << endl << "MILO - Multiscale/Multiphysics Interfaces for Large-scale Optimization -- Version " << MILO_VERSION << endl << endl;
    cout << "Copyright 2018 National Technology & Engineering Solutions of Sandia, LLC (NTESS)." << endl;
    cout << "Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software." << endl << endl;
    cout << "Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or Bart van Bloemen Waanders (bartv@sandia.gov)" << endl << endl;
  }
  else if (input_file_name == "--help") {
    if (argc > 2) {
      string helpwhat = argv[2];
      string details = "none";
      if (argc > 3) {
        details = argv[3];
      }
      MILOHelp::printHelp(helpwhat, details);
    }
    else {
      cout << endl << "MILO - Multiscale/Multiphysics Interfaces for Large-scale Optimization" << endl << endl;
      cout << "Suggested help topics: user, physics, solver, analysis, postprocess" << endl;
      cout << "Example: mpiexec -n 1 milo --help solver" << endl << endl;
    }
  }
  else {
    
    Teuchos::TimeMonitor ttimer(*totalTimer);
    
    ////////////////////////////////////////////////////////////////////////////////
    // Import default and user-defined settings into a parameter list
    ////////////////////////////////////////////////////////////////////////////////
    
    Teuchos::RCP<Teuchos::ParameterList> settings = userInterface(input_file_name);
    
    verbosity = settings->get<int>("verbosity",0);
    profile = settings->get<bool>("profile",false);
    int numElemPerCell = settings->sublist("Solver").get<int>("Workset size",1);
    
    ////////////////////////////////////////////////////////////////////////////////
    // split comm for SOL or multiscale runs (deprecated)
    ////////////////////////////////////////////////////////////////////////////////
    
    Teuchos::RCP<MpiComm> subgridComm, unusedComm;
    SplitComm(settings, *Comm, unusedComm, subgridComm);
    
    ////////////////////////////////////////////////////////////////////////////////
    // Create the mesh
    ////////////////////////////////////////////////////////////////////////////////
    
    Teuchos::RCP<meshInterface> mesh = Teuchos::rcp(new meshInterface(settings, Comm) );
    
    ////////////////////////////////////////////////////////////////////////////////
    // Set up the physics
    ////////////////////////////////////////////////////////////////////////////////
    
    Teuchos::RCP<physics> phys = Teuchos::rcp( new physics(settings, Comm,
                                                           mesh->cellTopo,
                                                           mesh->sideTopo,
                                                           mesh->mesh) );
    
    ////////////////////////////////////////////////////////////////////////////////
    // Mesh only needs the variable names and types to finalize
    ////////////////////////////////////////////////////////////////////////////////
    
    mesh->finalize(phys);
    
    ////////////////////////////////////////////////////////////////////////////////
    // Define the discretization(s)
    ////////////////////////////////////////////////////////////////////////////////
        
    Teuchos::RCP<discretization> disc = Teuchos::rcp( new discretization(settings, Comm,
                                                                         mesh->mesh,
                                                                         phys->unique_orders,
                                                                         phys->unique_types) );
    
    ////////////////////////////////////////////////////////////////////////////////
    // Create the function managers
    ////////////////////////////////////////////////////////////////////////////////
    
    vector<string> eBlocks;
    mesh->mesh->getElementBlockNames(eBlocks);
    vector<Teuchos::RCP<FunctionManager> > functionManagers;
    for (size_t b=0; b<eBlocks.size(); b++) {
      functionManagers.push_back(Teuchos::rcp(new FunctionManager(eBlocks[b],
                                                                  numElemPerCell,
                                                                  disc->numip[b],
                                                                  disc->numip_side[b])));
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    // Define the functions on each block
    ////////////////////////////////////////////////////////////////////////////////
    
    phys->defineFunctions(functionManagers);
    
    ////////////////////////////////////////////////////////////////////////////////
    // The DOF-manager needs to be aware of the physics and the discretization(s)
    ////////////////////////////////////////////////////////////////////////////////
    
    Teuchos::RCP<panzer::DOFManager> DOF = disc->buildDOF(mesh->mesh, phys->varlist,
                                                          phys->types, phys->orders,
                                                          phys->useDG);
    
    phys->setBCData(settings, mesh->mesh, DOF, disc->cards);
    
    ////////////////////////////////////////////////////////////////////////////////
    // Create the solver object
    ////////////////////////////////////////////////////////////////////////////////
    
    Teuchos::RCP<ParameterManager> params = Teuchos::rcp( new ParameterManager(Comm, settings,
                                                                               mesh->mesh, phys, disc));
                                                         
    Teuchos::RCP<AssemblyManager> assembler = Teuchos::rcp( new AssemblyManager(Comm, settings, mesh->mesh,
                                                                                disc, phys, DOF, params,
                                                                                numElemPerCell));
    
    mesh->setMeshData(assembler->cells);
    
    ////////////////////////////////////////////////////////////////////////////////
    // Set up the subgrid discretizations/models if using multiscale method
    ////////////////////////////////////////////////////////////////////////////////
    
    vector<Teuchos::RCP<SubGridModel> > subgridModels = subgridGenerator(subgridComm, settings, mesh->mesh);
    
    Teuchos::RCP<MultiScale> multiscale_manager = Teuchos::rcp( new MultiScale(Comm, subgridComm, settings,
                                                                               assembler->cells, subgridModels,
                                                                               functionManagers) );
    
    ////////////////////////////////////////////////////////////////////////////////
    // Set up the solver and finalize some objects
    ////////////////////////////////////////////////////////////////////////////////
    
    Teuchos::RCP<solver> solve = Teuchos::rcp( new solver(Comm, settings, mesh,
                                                          disc, phys, DOF, assembler, params) );
    
    solve->multiscale_manager = multiscale_manager;
    solve->setBatchID(Comm->getRank());
    
    ////////////////////////////////////////////////////////////////////////////////
    // Finalize the functions
    ////////////////////////////////////////////////////////////////////////////////
    
    for (size_t b=0; b<eBlocks.size(); b++) {
      functionManagers[b]->setupLists(phys->varlist[b], params->paramnames,
                                      params->discretized_param_names);
      
      functionManagers[b]->wkset = assembler->wkset[b];
      
      functionManagers[b]->validateFunctions();
      functionManagers[b]->decomposeFunctions();
    }
    
    solve->finalizeMultiscale();
    
    Teuchos::RCP<SensorManager> sensors = Teuchos::rcp( new SensorManager(settings, mesh, disc,
                                                                          assembler, params) );
    
    ////////////////////////////////////////////////////////////////////////////////
    // Create the postprocessing object
    ////////////////////////////////////////////////////////////////////////////////
    
    Teuchos::RCP<PostprocessManager>
    postproc = Teuchos::rcp( new PostprocessManager(Comm, settings, mesh->mesh, disc, phys,
                                                    solve, DOF, assembler->cells, functionManagers,
                                                    assembler, params, sensors) );
    
    ////////////////////////////////////////////////////////////////////////////////
    // Perform the requested analysis (fwd solve, adj solve, dakota run, etc.)
    // stored in settings->get<string>("analysis_type")
    ////////////////////////////////////////////////////////////////////////////////
    
    Teuchos::RCP<analysis> analys = Teuchos::rcp( new analysis(Comm, settings,
                                                               solve, postproc, params) );
    
    {
      Teuchos::TimeMonitor rtimer(*runTimer);
      analys->run();
    }
    
  }
  
  if (verbosity >= 10) {
    Teuchos::TimeMonitor::summarize();
  }
  else if (profile) {
    std::filebuf fb;
    fb.open ("MILO.profile",std::ios::out);
    std::ostream os(&fb);
    Teuchos::RCP<Teuchos::ParameterList> outlist = rcp(new Teuchos::ParameterList("options"));
    outlist->set("Report format","YAML");
    std::string filter = "";
    Teuchos::TimeMonitor::report(os, filter, outlist);
    fb.close();
  }
  
  Kokkos::finalize();
  
}
