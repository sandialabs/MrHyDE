/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.

 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
 ************************************************************************/

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

AnalysisManager::AnalysisManager(const Teuchos::RCP<MpiComm> &comm,
                                 Teuchos::RCP<Teuchos::ParameterList> &settings,
                                 Teuchos::RCP<SolverManager<SolverNode>> &solver,
                                 Teuchos::RCP<PostprocessManager<SolverNode>> &postproc,
                                 Teuchos::RCP<ParameterManager<SolverNode>> &params) : comm_(comm), settings_(settings), solver_(solver),
                                                                                       postproc_(postproc), params_(params)
{

  RCP<Teuchos::Time> constructortime = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AnalysisManager - constructor");
  Teuchos::TimeMonitor constructortimer(*constructortime);

  verbosity_ = settings_->get<int>("verbosity", 0);
  debugger_ = Teuchos::rcp(new MrHyDE_Debugger(settings_->get<int>("debug level", 0), comm));
  // No debug output on this constructor
}
