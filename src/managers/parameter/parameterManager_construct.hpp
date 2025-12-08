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
ParameterManager<Node>::ParameterManager(const Teuchos::RCP<MpiComm> & Comm_,
                                   Teuchos::RCP<Teuchos::ParameterList> & settings_,
                                   Teuchos::RCP<MeshInterface> & mesh_,
                                   Teuchos::RCP<PhysicsInterface> & phys_,
                                   Teuchos::RCP<DiscretizationInterface> & disc_) :
Comm(Comm_), disc(disc_), phys(phys_), settings(settings_) {
  
  Teuchos::TimeMonitor localtimer(*constructortimer);

  RCP<Teuchos::Time> constructortime = Teuchos::TimeMonitor::getNewCounter("MrHyDE::ParameterManager - constructor");
  Teuchos::TimeMonitor constructortimer(*constructortime);
  mesh = mesh_;
  
  debugger = Teuchos::rcp(new MrHyDE_Debugger(settings->get<int>("debug level",0), Comm));
  
  debugger->print("**** Starting ParameterManager constructor ... ");
  
  /////////////////////////////////////////////////////////////////////////////
  // Parameters
  /////////////////////////////////////////////////////////////////////////////
  
  blocknames = mesh->getBlockNames();
  spaceDim = mesh->getDimension();
  verbosity = settings->get<int>("verbosity",0);
  debug_level = settings->get<int>("debug level",0);
  
  num_inactive_params = 0;
  num_active_params = 0;
  num_stochastic_params = 0;
  num_discrete_params = 0;
  num_discretized_params = 0;
  globalParamUnknowns = 0;
  numParamUnknowns = 0;
  numParamUnknownsOS = 0;
  discretized_stochastic = false;
  have_dynamic_scalar = false;
  have_dynamic_discretized = false;
  
  use_custom_initial_param_guess = settings->sublist("Physics").get<bool>("use custom initial param guess",false);
  
  if (settings->sublist("Solver").isParameter("delta t")) {
    dynamic_dt = settings->sublist("Solver").get<double>("delta t");
  }
  else {
    dynamic_dt = 1.0;
  }
  dynamic_timeindex = 0; // starting point
  
  // Need number of time steps
  if (settings->sublist("Solver").isParameter("number of steps")) {
    numTimeSteps = settings->sublist("Solver").get<int>("number of steps",1);
  }
  else {
    double initial_time = settings->sublist("Solver").get<double>("initial time",0.0);
    double final_time = settings->sublist("Solver").get<double>("final time",1.0);
    double deltat = settings->sublist("Solver").get<double>("delta t",1.0);
    
    numTimeSteps = 0;
    double ctime = initial_time;
    while (ctime < final_time) {
      numTimeSteps++;
      ctime += deltat;
    }
    
  }
  
  this->setupParameters();
  
  debugger->print("**** Finished ParameterManager constructor");
  
}
