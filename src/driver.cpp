/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "trilinos.hpp"
#include "preferences.hpp"

#include "userInterface.hpp"
#include "meshInterface.hpp"
#include "physicsInterface.hpp"
#include "discretizationInterface.hpp"
#include "assemblyManager.hpp"
#include "parameterManager.hpp"
#include "multiscaleManager.hpp"
#include "solverManager.hpp"
#include "postprocessManager.hpp"
#include "functionManager.hpp"
#include "analysisManager.hpp"

int main(int argc,char * argv[]) {
  
#ifdef HAVE_MPI
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  Teuchos::RCP<MpiComm> Comm = Teuchos::rcp( new MpiComm(MPI_COMM_WORLD) );
#else
  EPIC_FAIL // MRHYDE requires MPI for HostDevice
#endif
  
  using namespace MrHyDE;
  
  int verbosity = 0;
  int debug_level = 0;
  bool profile = false;
  bool print_timers = false;
  
  Kokkos::initialize();
  
  {

    std::string input_file_name = "input.yaml";
    if (argc > 1) {
      input_file_name = argv[1];
    }
    if (input_file_name == "--version") {
      std::cout << std::endl << "MrHyDE - A framework for Multi-resolution Hybridized Differential Equations -- Version " << MRHYDE_VERSION << std::endl << std::endl;
      std::cout << "Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC (NTESS)." << std::endl;
      std::cout << "Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software." << std::endl << std::endl;
      std::cout << "Questions? Contact Tim Wildey (tmwilde@sandia.gov)" << std::endl << std::endl;
    }
    else {
      
      Teuchos::RCP<Teuchos::Time> totalTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::driver::total setup and execution time");
      Teuchos::RCP<Teuchos::Time> runTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::driver::total run time");
      
      Teuchos::TimeMonitor ttimer(*totalTimer);
      
      ////////////////////////////////////////////////////////////////////////////////
      // Import default and user-defined settings into a parameter list
      ////////////////////////////////////////////////////////////////////////////////
      
      Teuchos::RCP<userInterface> UI = Teuchos::rcp(new userInterface() );
      Teuchos::RCP<Teuchos::ParameterList> settings = UI->UserInterface(input_file_name);
      
      verbosity = settings->get<int>("verbosity",0);
      debug_level = settings->get<int>("debug level",0);
      profile = settings->get<bool>("profile",false);
      print_timers = settings->get<bool>("print timers",false);
      if (verbosity >= 10 && !settings->get<bool>("disable timers",false)) {
        print_timers = true;
      }
      
#if defined(MrHyDE_ENABLE_HDSA)
      bool is_hdsa_analysis = (settings->sublist("Analysis").get("analysis type", "forward") == "HDSA");
      if(is_hdsa_analysis)
      {
        settings->sublist("Postprocess").set("write solution", true);
        settings->sublist("Postprocess").set("create optimization movie", true);
      }
#endif
      
      ////////////////////////////////////////////////////////////////////////////////
      // Create the mesh
      ////////////////////////////////////////////////////////////////////////////////
      
      Teuchos::RCP<MeshInterface> mesh = Teuchos::rcp(new MeshInterface(settings, Comm) );
      
      ////////////////////////////////////////////////////////////////////////////////
      // Set up the physics
      ////////////////////////////////////////////////////////////////////////////////
      
      Teuchos::RCP<PhysicsInterface> physics = Teuchos::rcp( new PhysicsInterface(settings, Comm,
                                                                                  mesh->getBlockNames(),
                                                                                  mesh->getSideNames(),
                                                                                  mesh->getDimension()) );
      
      ////////////////////////////////////////////////////////////////////////////////
      // Mesh only needs the variable names and types to finalize
      ////////////////////////////////////////////////////////////////////////////////
      
      mesh->finalize(physics->getVarList(), physics->getVarTypes(), physics->getDerivedList());
      
      ////////////////////////////////////////////////////////////////////////////////
      // Define the discretization(s)
      ////////////////////////////////////////////////////////////////////////////////
      
      Teuchos::RCP<DiscretizationInterface> disc = Teuchos::rcp( new DiscretizationInterface(settings, Comm,
                                                                                             mesh, physics) );
      
      ////////////////////////////////////////////////////////////////////////////////
      // Create the solver object
      ////////////////////////////////////////////////////////////////////////////////
      
      Teuchos::RCP<ParameterManager<SolverNode> > params = Teuchos::rcp( new ParameterManager<SolverNode>(Comm, settings,
                                                                                                          mesh, physics, disc));
      
      Teuchos::RCP<AssemblyManager<SolverNode> > assembler = Teuchos::rcp( new AssemblyManager<SolverNode>(Comm, settings, mesh,
                                                                                                           disc, physics, params));
      
      assembler->setMeshData();
      
      ////////////////////////////////////////////////////////////////////////////////
      // Set up the subgrid discretizations/models if using multiscale method
      ////////////////////////////////////////////////////////////////////////////////
      
      Teuchos::RCP<MultiscaleManager> multiscale_manager = Teuchos::rcp( new MultiscaleManager(Comm, mesh, settings,
                                                                                               assembler->groups,
#ifndef MrHyDE_NO_AD
                                                                                               assembler->function_managers_AD) );
#else
      assembler->function_managers) );
#endif
      
      ///////////////////////////////////////////////////////////////////////////////
      // Create the postprocessing object
      ////////////////////////////////////////////////////////////////////////////////
      
      Teuchos::RCP<PostprocessManager<SolverNode> >
      postproc = Teuchos::rcp( new PostprocessManager<SolverNode>(Comm, settings, mesh,
                                                                  disc, physics, //assembler->function_managers_AD,
                                                                  multiscale_manager,
                                                                  assembler, params) );
      
      ////////////////////////////////////////////////////////////////////////////////
      // Set up the solver and finalize some objects
      ////////////////////////////////////////////////////////////////////////////////
      
      Teuchos::RCP<SolverManager<SolverNode> > solve = Teuchos::rcp( new SolverManager<SolverNode>(Comm, settings, mesh,
                                                                                                   disc, physics, assembler, params) );
      
      
      solve->multiscale_manager = multiscale_manager;
      assembler->multiscale_manager = multiscale_manager;
      solve->postproc = postproc;
      
      ////////////////////////////////////////////////////////////////////////////////
      // Allocate most of the per-element memory usage
      ////////////////////////////////////////////////////////////////////////////////
      
      mesh->allocateMeshDataStructures();
      assembler->allocateGroupStorage();
      
      ////////////////////////////////////////////////////////////////////////////////
      // Purge Panzer memory before solving
      ////////////////////////////////////////////////////////////////////////////////
      
      if (settings->get<bool>("enable memory purge",false)) {
        if (debug_level > 0 && Comm->getRank() == 0) {
          std::cout << "******** Starting driver memory purge ..." << std::endl;
        }
        if (!settings->sublist("Postprocess").get("write solution",false) &&
            !settings->sublist("Postprocess").get("create optimization movie",false)) {
          mesh->purgeMesh();
          disc->mesh = Teuchos::null;
          params->mesh = Teuchos::null;
        }
        disc->purgeOrientations();
        disc->purgeLIDs();
        mesh->purgeMemory();
        assembler->purgeMemory();
        params->purgeMemory();
        physics->purgeMemory();
        if (debug_level > 0 && Comm->getRank() == 0) {
          std::cout << "******** Finished driver memory purge ..." << std::endl;
        }
        
      }
      
      solve->completeSetup();
      postproc->linalg = solve->linalg;
      
      if (settings->get<bool>("enable memory purge",false)) {
        disc->purgeMemory();
      }
      
      ////////////////////////////////////////////////////////////////////////////////
      // Finalize the function and multiscale managers
      ////////////////////////////////////////////////////////////////////////////////
      
      assembler->finalizeFunctions();
      
      solve->finalizeMultiscale();
      
      ////////////////////////////////////////////////////////////////////////////////
      // Perform the requested analysis (fwd solve, adj solve, dakota run, etc.)
      ////////////////////////////////////////////////////////////////////////////////
      
      Teuchos::RCP<AnalysisManager> analysis = Teuchos::rcp( new AnalysisManager(Comm, settings,
                                                                                 solve, postproc, params) );
      
      // Make sure all processes are caught up at this point
      Kokkos::fence();
      Comm->barrier();
      
      if (verbosity >= 10) {
        if (Comm->getRank() == 0) {
          std::cout << std::endl << "*********************************************" << std::endl;
          std::cout << "Printing settings used by MrHyDE (ignore unused keyword):" << std::endl;
          settings->print(); // only prints on rank 0 anyway
          std::cout << "*********************************************" << std::endl;
        }
      }
      
      {
        Teuchos::TimeMonitor rtimer(*runTimer);
        analysis->run();
      }
      
    }
    
    
    if (print_timers) {
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
     
  }
  
  Kokkos::finalize();
  
}
