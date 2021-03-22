/***********************************************************************
Multiscale/Multiphysics Interfaces for Large-scale Optimization (MrHyDE)

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
#include "linearAlgebraInterface.hpp"
#include "solverManager.hpp"
#include "postprocessManager.hpp"
#include "analysisManager.hpp"
#include "trilinos.hpp"
#include "Panzer_DOFManager.hpp"

#include "preferences.hpp"
#include "subgridGenerator.hpp"
#include "MrHyDE_help.hpp"
#include "functionManager.hpp"
#include "split_mpi_communicators.hpp"

int main(int argc,char * argv[]) {
  
#ifdef HAVE_MPI
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  Teuchos::RCP<MpiComm> Comm = Teuchos::rcp( new MpiComm(MPI_COMM_WORLD) );
#else
  EPIC_FAIL // MRHYDE requires MPI for HostDevice
#endif
  
  using namespace MrHyDE;
  
  int verbosity = 0;
  bool profile = false;
  
  Kokkos::initialize();
  
  Teuchos::RCP<Teuchos::Time> totalTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::driver::total setup and execution time");
  Teuchos::RCP<Teuchos::Time> runTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::driver::total run time");
  
  std::string input_file_name = "input.yaml";
  if (argc > 1) {
    input_file_name = argv[1];
  }
  if (input_file_name == "--version") {
    std::cout << std::endl << "MrHyDE - A framework for Multi-resolution Hybridized Differential Equations -- Version " << MRHYDE_VERSION << std::endl << std::endl;
    std::cout << "Copyright 2018 National Technology & Engineering Solutions of Sandia, LLC (NTESS)." << std::endl;
    std::cout << "Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software." << std::endl << std::endl;
    std::cout << "Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or Bart van Bloemen Waanders (bartv@sandia.gov)" << std::endl << std::endl;
  }
  else if (input_file_name == "--help") {
    if (argc > 2) {
      std::string helpwhat = argv[2];
      std::string details = "none";
      if (argc > 3) {
        details = argv[3];
      }
      MrHyDEHelp::printHelp(helpwhat, details);
    }
    else {
      std::cout << std::endl << "MrHyDE - Multiscale/Multiphysics Interfaces for Large-scale Optimization" << std::endl << std::endl;
      std::cout << "Suggested help topics: user, physics, solver, analysis, postprocess" << std::endl;
      std::cout << "Example: mpiexec -n 1 MrHyDE --help solver" << std::endl << std::endl;
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
    
    Teuchos::RCP<physics> phys = Teuchos::rcp( new physics(settings, Comm, mesh->stk_mesh) );
    
    ////////////////////////////////////////////////////////////////////////////////
    // Mesh only needs the variable names and types to finalize
    ////////////////////////////////////////////////////////////////////////////////
    
    mesh->finalize(phys);
    
    ////////////////////////////////////////////////////////////////////////////////
    // Define the discretization(s)
    ////////////////////////////////////////////////////////////////////////////////
        
    Teuchos::RCP<discretization> disc = Teuchos::rcp( new discretization(settings, Comm,
                                                                         mesh->stk_mesh,
                                                                         phys) );
            
    ////////////////////////////////////////////////////////////////////////////////
    // Create the solver object
    ////////////////////////////////////////////////////////////////////////////////
    
    Teuchos::RCP<ParameterManager<SolverNode> > params = Teuchos::rcp( new ParameterManager<SolverNode>(Comm, settings,
                                                                                                        mesh->stk_mesh, phys, disc));
    
    Teuchos::RCP<AssemblyManager<SolverNode> > assembler = Teuchos::rcp( new AssemblyManager<SolverNode>(Comm, settings, mesh->stk_mesh,
                                                                                                         disc, phys, params));
    
    mesh->setMeshData(assembler->cells);
    
    ////////////////////////////////////////////////////////////////////////////////
    // Create the function managers
    ////////////////////////////////////////////////////////////////////////////////
    
    std::vector<Teuchos::RCP<FunctionManager> > functionManagers;
    for (size_t b=0; b<mesh->block_names.size(); b++) {
      functionManagers.push_back(Teuchos::rcp(new FunctionManager(mesh->block_names[b],
                                                                  assembler->cellData[b]->numElem,
                                                                  disc->numip[b],
                                                                  disc->numip_side[b])));
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    // Define the functions on each block
    ////////////////////////////////////////////////////////////////////////////////
    
    phys->defineFunctions(functionManagers);
    
    ////////////////////////////////////////////////////////////////////////////////
    // Set up the subgrid discretizations/models if using multiscale method
    ////////////////////////////////////////////////////////////////////////////////
    
    std::vector<Teuchos::RCP<SubGridModel> > subgridModels = subgridGenerator(subgridComm, settings, mesh);
    
    Teuchos::RCP<MultiScale> multiscale_manager = Teuchos::rcp( new MultiScale(Comm, subgridComm, settings,
                                                                               assembler->cells, subgridModels,
                                                                               functionManagers) );
    
    ///////////////////////////////////////////////////////////////////////////////
    // Create the postprocessing object
    ////////////////////////////////////////////////////////////////////////////////
    
    Teuchos::RCP<PostprocessManager<SolverNode> >
    postproc = Teuchos::rcp( new PostprocessManager<SolverNode>(Comm, settings, mesh,
                                                                disc, phys, functionManagers, multiscale_manager,
                                                                assembler, params) );
    
    ////////////////////////////////////////////////////////////////////////////////
    // Set up the solver and finalize some objects
    ////////////////////////////////////////////////////////////////////////////////
    
    Teuchos::RCP<solver<SolverNode> > solve = Teuchos::rcp( new solver<SolverNode>(Comm, settings, mesh,
                                                                                   disc, phys, assembler, params) );
    
    
    solve->multiscale_manager = multiscale_manager;
    solve->postproc = postproc;
    
    Teuchos::RCP<SensorManager<SolverNode> > sensors = Teuchos::rcp( new SensorManager<SolverNode>(settings, mesh,
                                                                                                   assembler) );
    
    postproc->sensors = sensors;
    
    ////////////////////////////////////////////////////////////////////////////////
    // Finalize the functions
    ////////////////////////////////////////////////////////////////////////////////
    
    for (size_t b=0; b<functionManagers.size(); b++) {
      functionManagers[b]->setupLists(phys->varlist[b], phys->aux_varlist[b],
                                      params->paramnames, params->discretized_param_names);
      
      functionManagers[b]->wkset = assembler->wkset[b];
      
      //functionManagers[b]->validateFunctions();
      functionManagers[b]->decomposeFunctions();
    }
    Kokkos::fence();
    
    solve->finalizeMultiscale();
    
    ////////////////////////////////////////////////////////////////////////////////
    // Purge Panzer memory before solving
    ////////////////////////////////////////////////////////////////////////////////
    
    if (settings->get<bool>("enable memory purge",true)) {
      disc->purgeMemory();
      mesh->purgeMemory();
      params->purgeMemory();
      assembler->purgeMemory();
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    // Perform the requested analysis (fwd solve, adj solve, dakota run, etc.)
    ////////////////////////////////////////////////////////////////////////////////
    
    Teuchos::RCP<analysis> analys = Teuchos::rcp( new analysis(Comm, settings,
                                                               solve, postproc, params) );
        
    {
      Teuchos::TimeMonitor rtimer(*runTimer);
      analys->run();
    }
    
    if (verbosity >= 20) {
      for (size_t b=0; b<assembler->wkset.size(); ++b) {
        assembler->wkset[b]->printMetaData();
      }
    }
  }
  
  if (verbosity >= 10) {
    Teuchos::TimeMonitor::summarize();
  }
  else if (profile) {
    std::filebuf fb;
    fb.open ("MrHyDE.profile",std::ios::out);
    std::ostream os(&fb);
    Teuchos::RCP<Teuchos::ParameterList> outlist = rcp(new Teuchos::ParameterList("options"));
    outlist->set("Report format","YAML");
    std::string filter = "";
    Teuchos::TimeMonitor::report(os, filter, outlist);
    fb.close();
  }
  
  Kokkos::finalize();
  
}
