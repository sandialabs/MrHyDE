/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.

 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
 ************************************************************************/

// ========================================================================================
// ========================================================================================

void AnalysisManager::run()
{

  debugger_->print("**** Starting AnalysisManager::run ...");

  std::string analysis_type = settings_->sublist("Analysis").get<string>("analysis type", "forward");
  this->run(analysis_type);

  debugger_->print("**** Finished analysis::run");
}

// ========================================================================================
// ========================================================================================

void AnalysisManager::run(std::string &analysis_type) {

  // There are several cases where one may want to modify the parameters from file
  bool params_from_file = settings_->sublist("Analysis").get<bool>("read parameters from file", false);
  if (params_from_file) {
    MrHyDE_OptVector fileparams = this->recoverParametersFromFile();
    params_->updateParams(fileparams);
  }
  
  if (analysis_type == "forward") {
    this->forwardSolve();
  }
  else if (analysis_type == "adjoint") {
    
    // Just running an adjoint means we need to read in, recover, or compute the forward state and parameters (see above)
        
    string fwdrecovery = settings_->sublist("Analysis").get<string>("forward state recovery type", "file");
    if (fwdrecovery == "file") {
      string statefilebase = settings_->sublist("Analysis").get<string>("state recovery file", "state");
      solver_->recoverForwardStateFromFile(statefilebase);
    }
    else if (fwdrecovery == "checkpointing") {
      // not actually implemented yet
      // in addition, it would need to be tightly intertwined with the time integrator to avoid just recomputing and storing the full forward state
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error: checkointing is not implemented yet.");
    }
    else if (fwdrecovery == "recompute") {
      this->forwardSolve();
    }
    else {
      // throw an error
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error: unrecognized forward state recovery type: " + fwdrecovery);
    }
    
    
    MrHyDE_OptVector sens = this->adjointSolve();
    bool writesens = settings_->sublist("Analysis").get<bool>("write gradient to file", false);
    if (writesens) {
      string filebase = settings_->sublist("Analysis").get<string>("gradient storage file", "grad");
      this->writeOptVectorToFile(sens, filebase);
    }
  }
  else if (analysis_type == "forward+adjoint") {
    this->forwardSolve();
    MrHyDE_OptVector sens = this->adjointSolve();
    bool writesens = settings_->sublist("Analysis").get<bool>("write gradient to file", false);
    if (writesens) {
      string filebase = settings_->sublist("Analysis").get<string>("gradient storage file", "grad");
      this->writeOptVectorToFile(sens, filebase);
    }
  }
  else if (analysis_type == "dry run") {
    cout << " **** MrHyDE has completed the dry run with verbosity: " << verbosity_ << endl;
  }
  else if (analysis_type == "UQ") {
    vector<Teuchos::Array<ScalarT>> response_values = this->UQSolve();
  }
  else if (analysis_type == "ROL") {
    this->ROLSolve();
  }
  else if (analysis_type == "ROL2") {
    this->ROL2Solve();
  }
  else if (analysis_type == "ROLStoch") {
    this->ROLStochSolve();
  }
#if defined(MrHyDE_ENABLE_HDSA)
  else if (analysis_type == "HDSA") {
    this->HDSASolve();
  }
  else if (analysis_type == "readExo+forward") {
    this->readExoForwardSolve();
  }
#endif
  else if (analysis_type == "DCI") {
    this->DCISolve();
  }
  else if (analysis_type == "Scalable DCI") {
    this->ScalableDCISolve();
  }
  else if (analysis_type == "Scalable Bayes") {
    this->ScalableBayesSolve();
  }
  else if (analysis_type == "restart") {
    this->restartSolve();
  }
  else {
    std::cout << "Unknown analysis option: " << analysis_type << std::endl;
    std::cout << "Valid and tested options: dry run, forward, adjoint, forward+adjoint, UQ, ROL, ROL2, DCI" << std::endl;
  }
  
  // There are several cases where one may want to write the parameters to file
  // Writing the state to file is handled by the postprocess manager
  bool params_to_file = settings_->sublist("Analysis").get<bool>("write parameters to file", false);
  if (params_to_file) {
    string filebase = settings_->sublist("Analysis").get<string>("parameter storage file", "params");
    MrHyDE_OptVector current_params = params_->getCurrentVector();
    this->writeOptVectorToFile(current_params, filebase);
  }
  
}

// ========================================================================================
// ========================================================================================

void AnalysisManager::forwardSolve() {
  ScalarT objfun = 0.0;
  solver_->forwardModel(objfun);
  postproc_->report();
}

// ========================================================================================
// ========================================================================================

MrHyDE_OptVector AnalysisManager::adjointSolve()
{

  MrHyDE_OptVector xtmp = params_->getCurrentVector();
  auto grad = xtmp.clone();
  MrHyDE_OptVector sens =
      Teuchos::dyn_cast<MrHyDE_OptVector>(const_cast<ROL::Vector<ScalarT> &>(*grad));
  sens.zero();
  solver_->adjointModel(sens);
  return sens;
}

// ========================================================================================
// ========================================================================================

vector<Teuchos::Array<ScalarT>> AnalysisManager::UQSolve()
{

  vector<Teuchos::Array<ScalarT>> response_values;

  // Build the uq manager
  Teuchos::ParameterList uqsettings_ = settings_->sublist("Analysis").sublist("UQ");
  vector<string> param_types = params_->stochastic_distribution;
  vector<ScalarT> param_means = params_->getStochasticParams("mean");
  vector<ScalarT> param_vars = params_->getStochasticParams("variance");
  vector<ScalarT> param_mins = params_->getStochasticParams("min");
  vector<ScalarT> param_maxs = params_->getStochasticParams("max");
  UQManager uq(comm_, uqsettings_, param_types, param_means, param_vars, param_mins, param_maxs);

  // Collect some settings_
  int numstochparams_ = param_types.size();
  int numsamples = uqsettings_.get<int>("samples", 100);
  int maxsamples = uqsettings_.get<int>("max samples", numsamples); // needed for generating subsets of samples
  int seed = uqsettings_.get<int>("seed", 1234);
  bool regenerate_rotations = uqsettings_.get<bool>("regenerate grain rotations", false);
  bool regenerate_grains = uqsettings_.get<bool>("regenerate grains", false);
  bool write_sol_text = uqsettings_.get<bool>("write solutions to text file", false);
  bool write_samples = uqsettings_.get<bool>("write samples", false);
  bool only_write_final = uqsettings_.get<bool>("only write final time", false);
  bool compute_adjoint = uqsettings_.get<bool>("compute adjoint", false);
  bool write_adjoint_text = uqsettings_.get<bool>("write adjoint to text file", false);
  int output_freq = uqsettings_.get<int>("output frequency", 1);

  bool finite_difference = uqsettings_.get<bool>("compute finite difference", false);
  int fd_component = uqsettings_.get<int>("finite difference component", 0);
  double fd_delta = uqsettings_.get<double>("finite difference delta", 1.0e-5);

  // Generate the samples (wastes memory if requires large number of samples in high-dim space)
  Kokkos::View<ScalarT **, HostDevice> samplepts = uq.generateSamples(maxsamples, seed);
  // Adjust the number of samples (if necessary)
  numsamples = std::min(numsamples, static_cast<int>(samplepts.extent(0)));
  Kokkos::View<int *, HostDevice> sampleints = uq.generateIntegerSamples(maxsamples, seed);

  // Write the samples to file (if requested)
  if (write_samples && comm_->getRank() == 0)
  {
    string sample_file = uqsettings_.get<string>("samples input file", "sample_inputs.dat");
    std::ofstream sampOUT(sample_file.c_str());
    sampOUT.precision(12);
    for (size_type i = 0; i < samplepts.extent(0); ++i)
    {
      for (size_type v = 0; v < samplepts.extent(1); ++v)
      {
        sampOUT << samplepts(i, v) << "  ";
      }
      sampOUT << endl;
    }
    sampOUT.close();
  }

  if (write_sol_text)
  {
    postproc_->save_solution = true;
  }

  if (comm_->getRank() == 0)
  {
    cout << "Running Monte Carlo sampling ..." << endl;
  }
  for (int j = 0; j < numsamples; j++)
  {

    ////////////////////////////////////////////////////////
    // Generate a new realization
    // Update stochastic parameters
    if (numstochparams_ > 0)
    {
      vector<ScalarT> currparams_;
      if (comm_->getRank() == 0)
      {
        cout << "New params: ";
      }
      for (int i = 0; i < numstochparams_; i++)
      {
        if (finite_difference && fd_component == i)
        {
          currparams_.push_back(samplepts(j, i) + fd_delta);
        }
        else
        {
          currparams_.push_back(samplepts(j, i));
        }
        if (comm_->getRank() == 0)
        {
          cout << samplepts(j, i) << " ";
        }
      }
      if (comm_->getRank() == 0)
      {
        cout << endl;
      }
      params_->updateParams(currparams_, 2);
    }
    // Update random microstructure
    if (regenerate_grains)
    {
      auto seeds = solver_->mesh->generateNewMicrostructure(sampleints(j));
      solver_->assembler->importNewMicrostructure(sampleints(j), seeds);
    }
    else if (regenerate_rotations)
    {
      this->updateRotationData(sampleints(j));
    }

    // Update the append string in postproc_essor for labelling
    std::stringstream ss;
    ss << "_" << j;
    postproc_->append = ss.str();

    ////////////////////////////////////////////////////////
    // Evaluate the new realization

    this->forwardSolve();
    Teuchos::Array<ScalarT> newresp = postproc_->collectResponses();

    response_values.push_back(newresp);

    ////////////////////////////////////////////////////////

    // The following output is for a specific use case.
    // Should not be used in general (unless you want TB of data)
    if (write_sol_text)
    {
      std::stringstream sfile;
      sfile << "solution." << j << "." << comm_->getRank() << ".dat";
      string filename = sfile.str();
      vector<vector<vector_RCP>> soln = postproc_->soln[0]->extractAllData();
      this->writeSolutionToText(filename, soln, only_write_final);
    }
    if (compute_adjoint)
    {

      postproc_->save_adjoint_solution = true;
      MrHyDE_OptVector sens = this->adjointSolve();

      if (write_adjoint_text)
      {
        std::stringstream sfile;
        sfile << "adjoint." << j << "." << comm_->getRank() << ".dat";
        string filename = sfile.str();
        vector<vector<vector_RCP>> soln = postproc_->adj_soln[0]->extractAllData();
        this->writeSolutionToText(filename, soln);
      }
    }

    // Update the user on the progress
    if (comm_->getRank() == 0 && j % output_freq == 0)
    {
      cout << "Finished evaluating sample number: " << j + 1 << " out of " << numsamples << endl;
    }
  } // end sample loop

  if (comm_->getRank() == 0)
  {

    string sname = uqsettings_.get<string>("samples output file", "sample_outputs.dat");
    std::ofstream respOUT(sname.c_str());
    respOUT.precision(12);
    for (size_t r = 0; r < response_values.size(); r++)
    {
      for (long int s = 0; s < response_values[r].size(); s++)
      {
        respOUT << response_values[r][s] << "  ";
      }
      respOUT << endl;
    }
    respOUT.close();
  }

  if (settings_->sublist("postprocess").get("write solution", true))
  {
    // postproc_->writeSolution(avgsoln, "output_avg");
  }
  // Compute the statistics (mean, variance, probability levels, etc.)
  // uq.computeStatistics(response_values);

  return response_values;
}

// ========================================================================================
// ========================================================================================

void AnalysisManager::ROLSolve()
{

  typedef ScalarT RealT;
  Teuchos::TimeMonitor localtimer(*roltimer);

  Teuchos::RCP<ROL::Objective_MILO<RealT>> obj;
  Teuchos::ParameterList ROLsettings;

  if (settings_->sublist("Analysis").isSublist("ROL"))
    ROLsettings = settings_->sublist("Analysis").sublist("ROL");
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error: MrHyDE could not find the ROL sublist in the input file!  Abort!");

  // New ROL input syntax
  bool use_linesearch = settings_->sublist("Analysis").get("Use Line Search", false);

  ROLsettings.sublist("General").sublist("Secant").set("Type", "Limited-Memory BFGS");
  ROLsettings.sublist("Step").sublist("Descent Method").set("Type", "Newton Krylov");
  ROLsettings.sublist("Step").sublist("Trust Region").set("Subproblem solve", "Truncated CG");

  RealT gtol = ROLsettings.sublist("Status Test").get("Gradient Tolerance", 1e-6);
  RealT stol = ROLsettings.sublist("Status Test").get("Step Tolerance", 1.e-12);
  int maxit = ROLsettings.sublist("Status Test").get("Iteration Limit", 100);

  // Turn off visualization while optimizing
  bool postproc_plot = postproc_->write_solution;
  postproc_->write_solution = false;

  Teuchos::RCP<std::ostream> outStream;
  outStream = Teuchos::rcp(&std::cout, false);
  // Generate data and get objective
  obj = Teuchos::rcp(new ROL::Objective_MILO<RealT>(solver_, postproc_, params_));

  Teuchos::RCP<ROL::Step<RealT>> step;

  if (use_linesearch)
  {
    step = Teuchos::rcp(new ROL::LineSearchStep<RealT>(ROLsettings));
  }
  else
  {
    step = Teuchos::rcp(new ROL::TrustRegionStep<RealT>(ROLsettings));
  }

  Teuchos::RCP<ROL::StatusTest<RealT>> status = Teuchos::rcp(new ROL::StatusTest<RealT>(gtol, stol, maxit));

  ROL::Algorithm<RealT> algo(step, status, false);

  MrHyDE_OptVector xtmp = params_->getCurrentVector();

  Teuchos::RCP<ROL::Vector<ScalarT>> x = xtmp.clone();
  x->set(xtmp);

  // ScalarT roltol = 1e-8;
  //*outStream << "\nTesting objective!!\n";
  // obj->value(*x, roltol);
  //*outStream << "\nObjective evaluation works!!\n";

  // bound contraint
  Teuchos::RCP<ROL::Bounds<RealT>> con;
  bool bound_vars = ROLsettings.sublist("General").get("Bound Optimization Variables", false);

  if (bound_vars)
  {

    // read in bounds for parameters...
    vector<Teuchos::RCP<vector<ScalarT>>> activeBnds = params_->getActiveParamBounds();
    vector<vector_RCP> discBnds = params_->getDiscretizedParamBounds();

    Teuchos::RCP<ROL::Vector<ScalarT>> lo = Teuchos::rcp(new MrHyDE_OptVector(discBnds[0], activeBnds[0], comm_->getRank()));
    Teuchos::RCP<ROL::Vector<ScalarT>> up = Teuchos::rcp(new MrHyDE_OptVector(discBnds[1], activeBnds[1], comm_->getRank()));

    con = Teuchos::rcp(new ROL::Bounds<RealT>(lo, up));

    // create bound constraint
  }

  //////////////////////////////////////////////////////
  // Verification tests
  //////////////////////////////////////////////////////

  // Recovering a data-generating solution
  if (ROLsettings.sublist("General").get("Generate data", false))
  {
    // std::cout << "Generating data ... " << std::endl;
    ScalarT objfun = 0.0;
    if (params_->isParameter("datagen"))
    {
      vector<ScalarT> pval = {1.0};
      params_->setParam(pval, "datagen");
    }
    postproc_->response_type = "none";
    postproc_->compute_objective = false;
    solver_->forwardModel(objfun);
    // std::cout << "Storing data ... " << std::endl;

    for (size_t set = 0; set < postproc_->soln.size(); ++set)
    {
      vector<vector<ScalarT>> times = postproc_->soln[set]->extractAllTimes();
      vector<vector<Teuchos::RCP<LA_MultiVector>>> data = postproc_->soln[set]->extractAllData();

      for (size_t i = 0; i < times.size(); i++)
      {
        for (size_t j = 0; j < times[i].size(); j++)
        {
          postproc_->datagen_soln[set]->store(data[i][j], times[i][j], i);
        }
      }
    }

    // std::cout << "Finished storing data" << std::endl;
    if (params_->isParameter("datagen"))
    {
      vector<ScalarT> pval = {0.0};
      params_->setParam(pval, "datagen");
    }
    postproc_->response_type = "discrete";
    postproc_->compute_objective = true;
    // std::cout << "Finished generating data for inversion " << std::endl;
  }

  // Comparing a gradient/Hessian with finite difference approximation
  if (ROLsettings.sublist("General").get("Do grad+hessvec check", true))
  {
    // Gradient and Hessian check
    // direction for gradient check

    Teuchos::RCP<ROL::Vector<ScalarT>> d = x->clone();

    if (ROLsettings.sublist("General").get("FD Check Use Ones Vector", false))
    {
      d->setScalar(1.0);
    }
    else
    {
      if (ROLsettings.sublist("General").isParameter("FD Check Seed"))
      {
        int seed = ROLsettings.get("FD Check Seed", 1);
        srand(seed);
      }
      else
      {
        srand(time(NULL)); // initialize random seed
      }
      d->randomize();
      if (ROLsettings.sublist("General").isParameter("FD Scale"))
      {
        ScalarT scale = ROLsettings.sublist("General").get<double>("FD Scale", 1.0);
        d->scale(scale);
      }
    }

    // check gradient and Hessian-vector computation using finite differences
    obj->checkGradient(*x, *d, (comm_->getRank() == 0));
  }

  // Teuchos::Time timer("Optimization Time", true);

  // Run algorithm.
  vector<std::string> output;
  if (bound_vars)
  {
    output = algo.run(*x, *obj, *con, (comm_->getRank() == 0)); // only processor of rank 0 print outs
  }
  else
  {
    output = algo.run(*x, *obj, (comm_->getRank() == 0)); // only processor of rank 0 prints out
  }

  ScalarT optTime = 0.0; // timer.stop();

  if (ROLsettings.sublist("General").get("Write Final Parameters", false))
  {
    string outname = ROLsettings.get("Output File Name", "ROL_out.txt");
    std::ofstream respOUT(outname);
    respOUT.precision(16);
    if (comm_->getRank() == 0)
    {

      for (unsigned i = 0; i < output.size(); i++)
      {
        std::cout << output[i];
        respOUT << output[i];
      }
      x->print(std::cout);
    }
    Kokkos::fence();
    x->print(respOUT);

    if (comm_->getRank() == 0)
    {
      if (verbosity_ > 5)
      {
        cout << "Optimization time: " << optTime << " seconds" << endl;
        respOUT << "\nOptimization time: " << optTime << " seconds" << endl;
      }
    }
    respOUT.close();
    string outname2 = "final_params_.dat";
    std::ofstream respOUT2(outname2);
    respOUT2.precision(16);
    x->print(respOUT2);
    respOUT2.close();
  }

  /*
   if (settings_->sublist("postproc_ess").get("write Hessian",false)){
   obj->printHess(settings_->sublist("postproc_ess").get("Hessian output file","hess.dat"),x,comm_->getRank());
   }
   if (settings_->sublist("Analysis").get("write output",false)) {
   ScalarT val = 0.0;
   solver_->forwardModel(val);
   //postproc_->writeSolution(settings_->sublist("postproc_ess").get<string>("Output File","output"));
   }
   */

  if (postproc_plot)
  {
    postproc_->write_solution = true;
    string outfile = "output_after_optimization.exo";
    postproc_->setNewExodusFile(outfile);
    ScalarT objfun = 0.0;
    solver_->forwardModel(objfun);
    if (ROLsettings.sublist("General").get("Disable source on final output", false))
    {
      vector<bool> newflags(1, false);
      solver_->physics->updateFlags(newflags);
      string outfile = "output_only_control.exo";
      postproc_->setNewExodusFile(outfile);
      solver_->forwardModel(objfun);
    }
  }
  
  if (settings_->sublist("Analysis").get("save parameters to file",false) ) {
    string filebase = settings_->sublist("Analysis").get("parameters file","params");
    MrHyDE_OptVector xtmp = params_->getCurrentVector();
    xtmp.print(filebase);
  }
}

// ========================================================================================
// ========================================================================================

void AnalysisManager::ROL2Solve()
{

  typedef ScalarT RealT;

  Teuchos::TimeMonitor localtimer(*rol2timer);

  Teuchos::RCP<ROL::Objective_MILO<RealT>> obj;
  Teuchos::ParameterList ROLsettings;

  if (settings_->sublist("Analysis").isSublist("ROL2"))
    ROLsettings = settings_->sublist("Analysis").sublist("ROL2");
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error: MrHyDE could not find the ROL2 sublist in the input file!  Abort!");

  // Turn off visualization while optimizing
  bool postproc_plot = postproc_->write_solution;
  postproc_->write_solution = false;

  // Output stream.
  ROL::Ptr<std::ostream> outStream;
  ROL::nullstream bhs; // outputs nothing
  if (comm_->getRank() == 0)
  {
    outStream = ROL::makePtrFromRef(std::cout);
  }
  else
  {
    outStream = ROL::makePtrFromRef(bhs);
  }

  // Generate data and get objective
  obj = Teuchos::rcp(new ROL::Objective_MILO<RealT>(solver_, postproc_, params_));

  MrHyDE_OptVector xtmp = params_->getCurrentVector();

  Teuchos::RCP<ROL::Vector<ScalarT>> x = xtmp.clone();
  x->set(xtmp);

  // bound constraint
  Teuchos::RCP<ROL::BoundConstraint<RealT>> con;
  bool bound_vars = ROLsettings.sublist("General").get("Bound Optimization Variables", false);

  if (bound_vars)
  {

    // read in bounds for parameters...
    vector<Teuchos::RCP<vector<ScalarT>>> activeBnds = params_->getActiveParamBounds();
    vector<vector_RCP> discBnds = params_->getDiscretizedParamBounds();
    Teuchos::RCP<ROL::Vector<ScalarT>> lo = Teuchos::rcp(new MrHyDE_OptVector(discBnds[0], activeBnds[0], comm_->getRank()));
    Teuchos::RCP<ROL::Vector<ScalarT>> up = Teuchos::rcp(new MrHyDE_OptVector(discBnds[1], activeBnds[1], comm_->getRank()));

    // create bound constraint
    con = Teuchos::rcp(new ROL::Bounds<RealT>(lo, up));
  }

  //////////////////////////////////////////////////////
  // Verification tests
  //////////////////////////////////////////////////////

  // Recovering a data-generating solution
  if (ROLsettings.sublist("General").get("Generate data", false))
  {
    // std::cout << "Generating data ... " << std::endl;
    ScalarT objfun = 0.0;
    if (params_->isParameter("datagen"))
    {
      vector<ScalarT> pval = {1.0};
      params_->setParam(pval, "datagen");
    }
    postproc_->response_type = "none";
    postproc_->compute_objective = false;
    solver_->forwardModel(objfun);
    // std::cout << "Storing data ... " << std::endl;

    for (size_t set = 0; set < postproc_->soln.size(); ++set)
    {
      vector<vector<ScalarT>> times = postproc_->soln[set]->extractAllTimes();
      vector<vector<Teuchos::RCP<LA_MultiVector>>> data = postproc_->soln[set]->extractAllData();
      for (size_t i = 0; i < times.size(); i++)
      {
        for (size_t j = 0; j < times[i].size(); j++)
        {
          postproc_->datagen_soln[set]->store(data[i][j], times[i][j], i);
        }
      }
    }

    // std::cout << "Finished storing data" << std::endl;
    if (params_->isParameter("datagen"))
    {
      vector<ScalarT> pval = {0.0};
      params_->setParam(pval, "datagen");
    }
    postproc_->response_type = "discrete";
    postproc_->compute_objective = true;
    // std::cout << "Finished generating data for inversion " << std::endl;
  }

  // Comparing a gradient/Hessian with finite difference approximation
  if (ROLsettings.sublist("General").get("Do grad+hessvec check", true))
  {
    // Gradient and Hessian check
    // direction for gradient check

    Teuchos::RCP<ROL::Vector<ScalarT>> d = x->clone();

    if (ROLsettings.sublist("General").get("FD Check Use Ones Vector", false))
    {
      d->setScalar(1.0);
    }
    else
    {
      if (ROLsettings.sublist("General").isParameter("FD Check Seed"))
      {
        int seed = ROLsettings.get("FD Check Seed", 1);
        srand(seed);
      }
      else
      {
        srand(time(NULL)); // initialize random seed
      }
      d->randomize();
      if (ROLsettings.sublist("General").isParameter("FD Scale"))
      {
        ScalarT scale = ROLsettings.sublist("General").get<double>("FD Scale", 1.0);
        d->scale(scale);
      }
    }

    // check gradient and Hessian-vector computation using finite differences
    obj->checkGradient(*x, *d, (comm_->getRank() == 0));
  }

  // Teuchos::Time timer("Optimization Time", true);

  // Construct ROL problem.
  ROL::Ptr<ROL::Problem<RealT>> rolProblem = ROL::makePtr<ROL::Problem<RealT>>(obj, x);
  // rolProblem->check(true, *outStream);
  if (bound_vars)
  {
    rolProblem->addBoundConstraint(con);
  }

  // Construct ROL solver.
  ROL::Solver<ScalarT> rolSolver(rolProblem, ROLsettings);

  // Run algorithm.
  rolSolver.solve(*outStream);

  // ScalarT optTime = timer.stop();

  /*
   if (settings_->sublist("postproc_ess").get("write Hessian",false)){
   obj->printHess(settings_->sublist("postproc_ess").get("Hessian output file","hess.dat"),x,comm_->getRank());
   }
   if (settings_->sublist("Analysis").get("write output",false)) {
   ScalarT val = 0.0;
   solver_->forwardModel(val);
   //postproc_->writeSolution(settings_->sublist("postproc_ess").get<string>("Output File","output"));
   }
   */

  if (postproc_plot)
  {
    postproc_->write_solution = true;
    string outfile = "output_after_optimization.exo";
    postproc_->setNewExodusFile(outfile);
    ScalarT objfun = 0.0;
    solver_->forwardModel(objfun);
    if (ROLsettings.sublist("General").get("Disable source on final output", false))
    {
      vector<bool> newflags(1, false);
      solver_->physics->updateFlags(newflags);
      string outfile = "output_only_control.exo";
      postproc_->setNewExodusFile(outfile);
      solver_->forwardModel(objfun);
    }
  }
  
  if (settings_->sublist("Analysis").get("save parameters to file",false) ) {
    string filebase = settings_->sublist("Analysis").get("parameters file","params");
    MrHyDE_OptVector xtmp = params_->getCurrentVector();
    xtmp.print(filebase);
  }
}

// ========================================================================================
// ========================================================================================

void AnalysisManager::ROLStochSolve()
{

  typedef ScalarT RealT;

  Teuchos::TimeMonitor localtimer(*rol2timer);

  Teuchos::ParameterList ROLsettings;
  if (settings_->sublist("Analysis").isSublist("ROL"))
    ROLsettings = settings_->sublist("Analysis").sublist("ROL");
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error: MrHyDE could not find the ROL sublist in the input file!  Abort!");

  // Turn off visualization while optimizing
  bool postproc_plot = postproc_->write_solution;
  postproc_->write_solution = false;

  // Output stream.
  ROL::Ptr<std::ostream> outStream;
  ROL::nullstream bhs; // outputs nothing
  if (comm_->getRank() == 0)
  {
    outStream = ROL::makePtrFromRef(std::cout);
  }
  else
  {
    outStream = ROL::makePtrFromRef(bhs);
  }

  MrHyDE_OptVector xtmp = params_->getCurrentVector();

  Teuchos::RCP<ROL::Vector<ScalarT>> x = xtmp.clone();
  x->set(xtmp);

  /*************************************************************************/
  /***************** BUILD SAMPLER *****************************************/
  /*************************************************************************/

  std::vector<ROL::Ptr<ROL::Distribution<RealT>>> dist;
  Teuchos::ParameterList parameters = settings_->sublist("Parameters");
  Teuchos::ParameterList::ConstIterator pl_itr = parameters.begin();
  while (pl_itr != parameters.end())
  {
    Teuchos::ParameterList newparam = parameters.sublist(pl_itr->first);
    if (newparam.get<string>("usage") == "stochastic")
    {
      if (newparam.get<string>("type") != "scalar")
      {
        std::cout << "Error: the current stochastic optimization implementation only permits scalar inactive parameters to be sampled" << std::endl;
      }
      ROL::Ptr<ROL::Distribution<RealT>> tmp = ROL::DistributionFactory<RealT>(newparam);
      dist.push_back(tmp);
    }
    pl_itr++;
  }
  ROL::Ptr<ROL::BatchManager<RealT>> bman = ROL::makePtr<ROL::MrHyDETeuchosBatchManager<RealT, int>>(comm_);
  int nsamp = ROLsettings.sublist("SOL").get("Number of Samples", 100);
  ROL::Ptr<ROL::SampleGenerator<RealT>> sampler;
  if (ROLsettings.sublist("SOL").get("Sample Set File", "error") == "error")
  {
    std::cout << "I was unable to read the sample set or weights, so a new sampler is being created" << std::endl;
    sampler = ROL::makePtr<ROL::MonteCarloGenerator<RealT>>(nsamp, dist, bman);
  }
  else
  {
    int dim = dist.size();
    std::string sample_pt_file = ROLsettings.sublist("SOL").get("Sample Set File", "error");
    std::string sample_wt_file = ROLsettings.sublist("SOL").get("Sample Weight File", "error");
    sampler = ROL::makePtr<ROL::Sample_Set_Reader<RealT>>(nsamp, dim, bman, sample_pt_file, sample_wt_file);
  }

  Teuchos::RCP<ROL::Stochastic_Objective_MILO<RealT>> obj = Teuchos::rcp(new ROL::Stochastic_Objective_MILO<RealT>(solver_, postproc_, params_, sampler));
  if (ROLsettings.sublist("General").get("Do grad check", true))
  {
    Teuchos::RCP<ROL::Vector<ScalarT>> dx = x->clone();
    dx->randomize();
    obj->checkGradient(*x, *dx, true, *outStream);
  }

  std::string init_iter_file = ROLsettings.sublist("General").get("Initial Iterate File", "error");
  if (init_iter_file != "error")
  {
    // This functionality is currently only supported for parameters stored as std::vectors

    RealT val = 0.0;
    // read in data
    std::ifstream in(init_iter_file);
    MrHyDE_OptVector &xs = dynamic_cast<MrHyDE_OptVector &>(*x);

    if (xs.haveScalar())
    {
      std::vector<ROL::Ptr<ROL::StdVector<ScalarT>>> x_data = xs.getParameter();
      ROL::Ptr<std::vector<ScalarT>> x_std = x_data[0]->getVector();
      // read the elements in the file into a vector
      if (in)
      {
        for (int i = 0; i < x->dimension(); i++)
        {
          in >> val;
          (*x_std)[i] = val;
        }
      }
      else
      {
        std::cout << "Error loading the data from " << init_iter_file << std::endl;
      }
    }
    else if (xs.haveField())
    {
      std::vector<ROL::Ptr<ROL::TpetraMultiVector<ScalarT>>> x_data = xs.getField();
      ROL::Ptr<Tpetra::MultiVector<ScalarT,LO,GO,SolverNode> > x_tpetra = x_data[0]->getVector();
      if (in)
      {
        for (int i = 0; i < x->dimension(); i++)
        {
          in >> val;
          x_tpetra->replaceGlobalValue(i,0,val);
        }
      }
      else
      {
        std::cout << "Error loading the data from " << init_iter_file << std::endl;
      }
    }
  }

  if (ROLsettings.sublist("General").get("Write Iteration History", false) && (comm_->getRank() == 0))
  {
    string outname = ROLsettings.get("Output File Name", "ROL_out.txt");
    freopen(outname.c_str(), "w", stdout);
  }
  Kokkos::fence();

  int N = sampler->numMySamples();
  std::vector<RealT> sample_weights = std::vector<RealT>(N, 0.0);
  if (!ROLsettings.sublist("SOL").get("Use Primal Dual", false))
  {
    ROL::Ptr<ROL::StochasticProblem<RealT>> rolProblem = ROL::makePtr<ROL::StochasticProblem<RealT>>(obj, x);
    rolProblem->makeObjectiveStochastic(ROLsettings, sampler);
    rolProblem->finalize(false, false, *outStream);
    ROL::Solver<ScalarT> rolSolver(rolProblem, ROLsettings);
    rolSolver.solve(*outStream);
    for (int i = 0; i < N; i++)
    {
      sample_weights[i] = sampler->getMyWeight(i);
    }
  }
  else
  {
#if defined(MrHyDE_ENABLE_HDSA)
    ROL::Ptr<ROL::Problem<RealT>> rolProblem = ROL::makePtr<ROL::Problem<RealT>>(obj, x);
    ROL::Ptr<ROL::PrimalDualRisk<RealT>> pd_risk = ROL::makePtr<ROL::PrimalDualRisk<RealT>>(rolProblem, sampler, ROLsettings);
    pd_risk->run(*outStream);

    sample_weights = pd_risk->getMultipliers();
#endif
  }

  if (ROLsettings.sublist("General").get("Write Iteration History", false) && (comm_->getRank() == 0))
  {
    fclose(stdout);
  }
  Kokkos::fence();

  if (ROLsettings.sublist("General").get("Write Final Parameters", false))
  {
    string outname2 = "final_params_.dat";
    std::ofstream respOUT2(outname2);
    respOUT2.precision(16);
    x->print(respOUT2);
    respOUT2.close();
  }

  if (ROLsettings.sublist("SOL").get("Write Samples", false))
  {
    string outname3 = "sample_set.dat";
    std::ofstream respOUT3(outname3);
    respOUT3.precision(16);
    for (int i = 0; i < sampler->numMySamples(); i++)
    {
      std::vector<ScalarT> pt_i = sampler->getMyPoint(i);
      for (int k = 0; k < pt_i.size(); k++)
      {
        respOUT3 << pt_i[k] << " ";
      }
      respOUT3 << std::endl;
    }
    respOUT3.close();

    string outname4 = "sample_weights.dat";
    std::ofstream respOUT4(outname4);
    respOUT4.precision(16);
    for (int i = 0; i < sampler->numMySamples(); i++)
    {
      respOUT4 << sample_weights[i] << " ";
      respOUT4 << std::endl;
    }
    respOUT4.close();
  }

  if (postproc_plot)
  {
    postproc_->write_solution = true;
    std::vector<ScalarT> obj_vals = std::vector<ScalarT>(sampler->numMySamples(), 0.0);
    for (int i = 0; i < sampler->numMySamples(); i++)
    {
      std::vector<ScalarT> pt_i = sampler->getMyPoint(i);
      params_->updateParams(pt_i, "stochastic");
      string outfile = "output_after_optimization_sample_" + std::to_string(i) + ".exo";
      postproc_->setNewExodusFile(outfile);
      ScalarT objfun = 0.0;
      solver_->forwardModel(objfun);
      obj_vals[i] = objfun;
    }
    string outname5 = "sample_obj_vals.dat";
    std::ofstream respOUT5(outname5);
    respOUT5.precision(16);
    for (int i = 0; i < sampler->numMySamples(); i++)
    {
      respOUT5 << obj_vals[i] << " ";
      respOUT5 << std::endl;
    }
    respOUT5.close();
  }
}

// ========================================================================================
// ========================================================================================

#if defined(MrHyDE_ENABLE_HDSA)
void AnalysisManager::HDSASolve()
{
  postproc_->write_solution = false;
  postproc_->write_optimization_solution = false;

  ///////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////// Parameter parsing ////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////

  Teuchos::ParameterList HDSAsettings;

  if (settings_->sublist("Analysis").isSublist("HDSA"))
    HDSAsettings = settings_->sublist("Analysis").sublist("HDSA");
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error: MrHyDE could not find the HDSA sublist in the input file!  Abort!");

  int stoch_dim = params_->getNumParams("stochastic");
  bool is_stoch = false;
  if (stoch_dim > 0)
  {
    is_stoch = true;
  }

  int num_prior_samples = HDSAsettings.sublist("Configuration").get<int>("num_prior_samples", 0);
  int num_posterior_samples = HDSAsettings.sublist("Configuration").get<int>("num_posterior_samples", 0);
  int prior_num_state_solves = HDSAsettings.sublist("Configuration").get<int>("prior_num_state_solves", 0);
  int hdsa_verbosity = HDSAsettings.sublist("Configuration").get<int>("verbosity", 0);
  bool use_lumped_mass_prior = HDSAsettings.sublist("Configuration").get<bool>("use_lumped_mass_prior", false);
  bool execute_prior_discrepancy_sampling = HDSAsettings.sublist("Configuration").get<bool>("execute_prior_discrepancy_sampling", false);
  bool execute_posterior_discrepancy_sampling = HDSAsettings.sublist("Configuration").get<bool>("execute_posterior_discrepancy_sampling", false);
  bool execute_optimal_solution_update = HDSAsettings.sublist("Configuration").get<bool>("execute_optimal_solution_update", false);

  int num_states = solver_->varlist[0][0].size();

  std::vector<ScalarT> alpha_u = std::vector<ScalarT>(num_states, 0.0);
  std::vector<ScalarT> beta_u = std::vector<ScalarT>(num_states, 0.0);
  std::vector<ScalarT> beta_t = std::vector<ScalarT>(num_states, 0.0);
  std::vector<ScalarT> alpha_d = std::vector<ScalarT>(num_states, 0.0);
  std::vector<int> prior_num_sing_vals = std::vector<int>(num_states, 0);
  std::vector<int> prior_oversampling = std::vector<int>(num_states, 0);
  std::vector<int> prior_num_subspace_iter = std::vector<int>(num_states, 0);
  for (int k = 0; k < num_states; k++)
  {
    std::string state_var_name = solver_->varlist[0][0][k];
    alpha_u[k] = HDSAsettings.sublist("HyperParameters").sublist(state_var_name).get<ScalarT>("alpha_u", 0.0);
    beta_u[k] = HDSAsettings.sublist("HyperParameters").sublist(state_var_name).get<ScalarT>("beta_u", 0.0);
    beta_t[k] = HDSAsettings.sublist("HyperParameters").sublist(state_var_name).get<ScalarT>("beta_t", 0.0);
    alpha_d[k] = HDSAsettings.sublist("HyperParameters").sublist(state_var_name).get<ScalarT>("alpha_d", 0.0);
    prior_num_sing_vals[k] = HDSAsettings.sublist("HyperParameters").sublist(state_var_name).get<int>("prior_num_sing_vals", 200);
    prior_oversampling[k] = HDSAsettings.sublist("HyperParameters").sublist(state_var_name).get<int>("prior_oversampling", 20);
    prior_num_subspace_iter[k] = HDSAsettings.sublist("HyperParameters").sublist(state_var_name).get<int>("prior_num_subspace_iter", 1);
  }

  ScalarT alpha_z = HDSAsettings.sublist("HyperParameters").sublist("z").get<ScalarT>("alpha_z", 0.0);
  ScalarT beta_z = HDSAsettings.sublist("HyperParameters").sublist("z").get<ScalarT>("beta_z", 0.0);
  ScalarT beta_t_z_prior = HDSAsettings.sublist("HyperParameters").sublist("z").get<ScalarT>("beta_t", 0.0);

  ScalarT max_marginal_var_percent = HDSAsettings.sublist("HyperParameters").sublist("OUU").get<ScalarT>("max_marginal_var_percent", 1.0);
  ScalarT min_cond_variance_percent = HDSAsettings.sublist("HyperParameters").sublist("OUU").get<ScalarT>("min_cond_variance_percent", 0.1);
  bool assume_independent_ensembles = HDSAsettings.sublist("HyperParameters").sublist("OUU").get<bool>("assume_independent_ensembles", false);

  int hessian_num_eig_vals = HDSAsettings.sublist("HyperParameters").get<int>("hessian_num_eig_vals", 5);
  int hessian_oversampling = HDSAsettings.sublist("HyperParameters").get<int>("hessian_oversampling", 3);

  bool center_data = HDSAsettings.sublist("HyperParameters").get<bool>("center_data", false);
  bool adapt_time_variance = HDSAsettings.sublist("HyperParameters").get<bool>("adapt_time_variance", false);

  Teuchos::ParameterList data_load_list = HDSAsettings.sublist("DataLoadParameters");
  std::string random_number_file = data_load_list.get<std::string>("random_number_file", "error");
  int num_random_numbers = data_load_list.get<int>("num_random_numbers", 0);

  ///////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////// Random number and sampler instantiation //////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////

  HDSA::Ptr<HDSA::Random_Number_Generator<ScalarT>> random_number_generator;
  HDSA::Ptr<HDSA::Comm<int>> hdsa_comm = HDSA::makePtr<HDSA::Comm<int>>(comm_);
  if (random_number_file == "error")
  {
    random_number_generator = HDSA::makePtr<HDSA::Random_Number_Generator<ScalarT>>(hdsa_comm);
  }
  else
  {
    if (num_random_numbers == 0)
    {
      std::cout << " Error: number of random numbers not specified" << std::endl;
    }
    random_number_generator = HDSA::makePtr<HDSA::Random_Number_Generator<ScalarT>>(num_random_numbers, random_number_file);
  }

  HDSA::Ptr<ROL::SampleGenerator<ScalarT>> sampler;
  int ens_size = data_load_list.get<int>("Ensemble Size", 0);
  if (is_stoch)
  {
    if (ens_size == 0)
    {
      std::cout << "Error: the ensemble size was not specified" << std::endl;
    }
    HDSA::Ptr<ROL::BatchManager<ScalarT>> bman = HDSA::makePtr<ROL::MrHyDETeuchosBatchManager<ScalarT, int>>(comm_);
    std::string sample_pt_file = data_load_list.get("Sample Set File", "error");
    std::string sample_wt_file = data_load_list.get("Sample Weight File", "error");
    sampler = HDSA::makePtr<ROL::Sample_Set_Reader<ScalarT>>(ens_size, stoch_dim, bman, sample_pt_file, sample_wt_file);
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////// Data_Interface ///////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////
  if (hdsa_verbosity > 1)
  {
    std::cout << "Beginning Data_Interface instantiation" << std::endl;
  }

  HDSA::Ptr<HDSA::MD_Data_Interface<ScalarT>> data_interface;
  if (is_stoch)
  {
    std::vector<HDSA::Ptr<MD_Data_Interface_MrHyDE<ScalarT>>> data_interface_ens;
    data_interface_ens.resize(ens_size);
    for (int s = 0; s < ens_size; s++)
    {
      data_interface_ens[s] = HDSA::makePtr<MD_Data_Interface_MrHyDE<ScalarT>>(comm_, solver_, params_, random_number_generator, data_load_list);

      std::string exo_file_base = data_interface_ens[s]->Get_Opt_Solution_Exo_File();
      std::string exo_file = exo_file_base + std::to_string(s) + ".exo";
      data_interface_ens[s]->Overwrite_Opt_Solution_Exo_File(exo_file);

      std::vector<std::string> exo_files_base = data_interface_ens[s]->Get_HiFi_Exo_Files();
      std::vector<std::string> exo_files;
      exo_files.resize(exo_files_base.size());
      for (int k = 0; k < exo_files.size(); k++)
      {
        exo_files[k] = exo_files_base[k] + std::to_string(s) + ".exo";
      }
      data_interface_ens[s]->Overwrite_HiFi_Exo_Files(exo_files);
    }
    data_interface = HDSA::makePtr<MD_OUU_Data_Interface_MrHyDE<ScalarT>>(data_interface_ens, sampler, ens_size);
  }
  else
  {
    data_interface = HDSA::makePtr<MD_Data_Interface_MrHyDE<ScalarT>>(comm_, solver_, params_, random_number_generator, data_load_list);
  }

  if (hdsa_verbosity > 0)
  {
    HDSA::Ptr<const HDSA::Vector<ScalarT>> u_opt = data_interface->Get_u_opt();
    HDSA::Ptr<const HDSA::Vector<ScalarT>> z_opt = data_interface->Get_z_opt();
    HDSA::Ptr<const HDSA::MultiVector<ScalarT>> Z = data_interface->Get_Z();
    HDSA::Ptr<const HDSA::MultiVector<ScalarT>> D = data_interface->Get_D();
    std::cout << "u_opt->norm() = " << u_opt->Norm() << std::endl;
    std::cout << "z_opt->norm() = " << z_opt->Norm() << std::endl;
    int N = Z->Number_of_Vectors();
    for (int k = 0; k < N; k++)
    {
      std::cout << "Z[" << k << "]->norm() = " << (*Z)[k]->Norm() << std::endl;
      std::cout << "D[" << k << "]->norm() = " << (*D)[k]->Norm() << std::endl;
    }
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////// Opt_Prob_Interface ///////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////

  if (hdsa_verbosity > 1)
  {
    std::cout << "Beginning Opt_Prob_Interface instantiation" << std::endl;
  }

  HDSA::Ptr<HDSA::MD_Opt_Prob_Interface<ScalarT>> opt_prob_interface;
  if (is_stoch)
  {
    std::vector<ScalarT> ens_weights = std::vector<ScalarT>(ens_size, 0.0);
    for (int s = 0; s < ens_size; s++)
    {
      ens_weights[s] = sampler->getMyWeight(s);
    }
    opt_prob_interface = HDSA::makePtr<MD_OUU_Opt_Prob_Interface_MrHyDE<ScalarT>>(solver_, postproc_, params_, data_interface, sampler, ens_weights);
  }
  else
  {
    opt_prob_interface = HDSA::makePtr<MD_Opt_Prob_Interface_MrHyDE<ScalarT>>(solver_, postproc_, params_, data_interface);
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////// u_Prior_Interface ////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////

  if (hdsa_verbosity > 1)
  {
    std::cout << "Beginning u_Prior_Interface instantiation" << std::endl;
  }

  vector<string> blockNames = solver_->mesh->getBlockNames();
  HDSA::Ptr<Prior_Operators_Interface_MrHyDE<ScalarT>> prior_operator_interface = HDSA::makePtr<Prior_Operators_Interface_MrHyDE<ScalarT>>(comm_, settings_, blockNames);
  HDSA::Ptr<HDSA::Sparse_Matrix<ScalarT>> M = HDSA::makePtr<HDSA::Sparse_Matrix<ScalarT>>(prior_operator_interface->M);
  HDSA::Ptr<HDSA::Sparse_Matrix<ScalarT>> S = HDSA::makePtr<HDSA::Sparse_Matrix<ScalarT>>(prior_operator_interface->S);

  HDSA::Ptr<HDSA::MD_u_Prior_Interface<ScalarT>> u_prior_interface;
  HDSA::Ptr<HDSA::MD_u_Hyperparameter_Interface<ScalarT>> u_hyperparam_interface;

  std::vector<HDSA::Ptr<HDSA::MD_u_Hyperparameter_Interface<ScalarT>>> u_hyperparam_interface_std;
  std::vector<HDSA::Ptr<HDSA::MD_u_Prior_Interface<ScalarT>>> u_prior_interface_std;
  u_hyperparam_interface_std.resize(num_states);
  u_prior_interface_std.resize(num_states);

  bool is_transient = solver_->isTransient;
  for (int k = 0; k < num_states; k++)
  {
    u_hyperparam_interface_std[k] = HDSA::makePtr<MD_u_Hyperparameter_Interface_MrHyDE<ScalarT>>(comm_, data_interface, is_transient, center_data, adapt_time_variance);
    u_hyperparam_interface_std[k]->Set_alpha_d(alpha_d[k]);
    u_hyperparam_interface_std[k]->Set_alpha_u(alpha_u[k]);
    u_hyperparam_interface_std[k]->Set_beta_u(beta_u[k]);
    u_hyperparam_interface_std[k]->Set_beta_t(beta_t[k]);
    u_hyperparam_interface_std[k]->Set_GSVD_Hyperparameters(prior_num_sing_vals[k], prior_oversampling[k], prior_num_subspace_iter[k]);

    if (is_transient)
    {
      HDSA::Ptr<HDSA::MD_u_Prior_Interface<ScalarT>> spatial_u_prior_interface_k;
      HDSA::Ptr<HDSA::MD_Transient_Prior_Covariance<ScalarT>> transient_prior_cov_k;
      ScalarT T = solver_->final_time;
      int n_t = solver_->settings->sublist("Solver").get<int>("number of steps", 0) + 1;

      if (is_stoch)
      {
        HDSA::Ptr<HDSA::MD_OUU_Data_Interface<ScalarT>> ouu_data_interface = Teuchos::rcp_dynamic_cast<HDSA::MD_OUU_Data_Interface<ScalarT>>(data_interface);
        HDSA::Ptr<HDSA::MD_OUU_Hyperparameter_Data_Interface<ScalarT>> data_interface_hyperparam = HDSA::makePtr<HDSA::MD_OUU_Hyperparameter_Data_Interface<ScalarT>>(ouu_data_interface);
        if (use_lumped_mass_prior)
        {
          spatial_u_prior_interface_k = HDSA::makePtr<HDSA::MD_Lumped_Mass_u_Prior_Interface<ScalarT>>(S, M, data_interface_hyperparam, u_hyperparam_interface_std[k], hdsa_comm, random_number_generator);
        }
        else
        {
          spatial_u_prior_interface_k = HDSA::makePtr<HDSA::MD_Numeric_Laplacian_u_Prior_Interface<ScalarT>>(S, M, data_interface_hyperparam, u_hyperparam_interface_std[k], random_number_generator);
        }
        int n_y = data_interface_hyperparam->Get_u_opt()->Dimension() / n_t;
        transient_prior_cov_k = HDSA::makePtr<HDSA::MD_Transient_Prior_Covariance<ScalarT>>(data_interface_hyperparam, u_hyperparam_interface_std[k], T, n_t, n_y);
      }
      else
      {
        int n_y = data_interface->Get_u_opt()->Dimension() / n_t;
        if (use_lumped_mass_prior)
        {
          spatial_u_prior_interface_k = HDSA::makePtr<HDSA::MD_Lumped_Mass_u_Prior_Interface<ScalarT>>(S, M, data_interface, u_hyperparam_interface_std[k], hdsa_comm, random_number_generator);
        }
        else
        {
          spatial_u_prior_interface_k = HDSA::makePtr<HDSA::MD_Numeric_Laplacian_u_Prior_Interface<ScalarT>>(S, M, data_interface, u_hyperparam_interface_std[k], random_number_generator);
        }
        transient_prior_cov_k = HDSA::makePtr<HDSA::MD_Transient_Prior_Covariance<ScalarT>>(data_interface, u_hyperparam_interface_std[k], T, n_t, n_y);
      }

      u_prior_interface_std[k] = HDSA::makePtr<HDSA::MD_Transient_Elliptic_u_Prior_Interface<ScalarT>>(spatial_u_prior_interface_k, transient_prior_cov_k);
    }
    else
    {
      if (is_stoch)
      {
        HDSA::Ptr<HDSA::MD_OUU_Data_Interface<ScalarT>> ouu_data_interface = Teuchos::rcp_dynamic_cast<HDSA::MD_OUU_Data_Interface<ScalarT>>(data_interface);
        HDSA::Ptr<HDSA::MD_OUU_Hyperparameter_Data_Interface<ScalarT>> data_interface_hyperparam = HDSA::makePtr<HDSA::MD_OUU_Hyperparameter_Data_Interface<ScalarT>>(ouu_data_interface);
        if (use_lumped_mass_prior)
        {
          u_prior_interface_std[k] = HDSA::makePtr<HDSA::MD_Lumped_Mass_u_Prior_Interface<ScalarT>>(S, M, data_interface_hyperparam, u_hyperparam_interface_std[k], hdsa_comm, random_number_generator);
        }
        else
        {
          u_prior_interface_std[k] = HDSA::makePtr<HDSA::MD_Numeric_Laplacian_u_Prior_Interface<ScalarT>>(S, M, data_interface_hyperparam, u_hyperparam_interface_std[k], random_number_generator);
        }
      }
      else
      {
        if (use_lumped_mass_prior)
        {
          u_prior_interface_std[k] = HDSA::makePtr<HDSA::MD_Lumped_Mass_u_Prior_Interface<ScalarT>>(S, M, data_interface, u_hyperparam_interface_std[k], hdsa_comm, random_number_generator);
        }
        else
        {
          u_prior_interface_std[k] = HDSA::makePtr<HDSA::MD_Numeric_Laplacian_u_Prior_Interface<ScalarT>>(S, M, data_interface, u_hyperparam_interface_std[k], random_number_generator);
        }
      }
    }
  }

  if (is_stoch)
  {
    HDSA::Ptr<HDSA::MD_u_Prior_Interface<ScalarT>> us_prior_interface;
    if (num_states > 1)
    {
      u_hyperparam_interface = HDSA::makePtr<HDSA::MD_Multi_State_u_Hyperparameter_Interface<ScalarT>>(u_hyperparam_interface_std);
      us_prior_interface = HDSA::makePtr<HDSA::MD_Multi_State_u_Prior_Interface<ScalarT>>(data_interface, u_prior_interface_std);
    }
    else
    {
      u_hyperparam_interface = u_hyperparam_interface_std[0];
      us_prior_interface = u_prior_interface_std[0];
    }

    HDSA::Ptr<HDSA::MD_OUU_Ensemble_Weighting_Matrix<ScalarT>> ensemble_weighting = HDSA::makePtr<HDSA::MD_OUU_Ensemble_Weighting_Matrix<ScalarT>>(data_interface, us_prior_interface, ens_size, max_marginal_var_percent, min_cond_variance_percent, assume_independent_ensembles);
    u_prior_interface = HDSA::makePtr<HDSA::MD_OUU_u_Prior_Interface<ScalarT>>(us_prior_interface, ensemble_weighting);
  }
  else
  {
    if (num_states > 1)
    {
      u_hyperparam_interface = HDSA::makePtr<HDSA::MD_Multi_State_u_Hyperparameter_Interface<ScalarT>>(u_hyperparam_interface_std);
      u_prior_interface = HDSA::makePtr<HDSA::MD_Multi_State_u_Prior_Interface<ScalarT>>(data_interface, u_prior_interface_std);
    }
    else
    {
      u_hyperparam_interface = u_hyperparam_interface_std[0];
      u_prior_interface = u_prior_interface_std[0];
    }
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////// z_Prior_Interface ////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////

  if (hdsa_verbosity > 1)
  {
    std::cout << "Beginning z_Prior_Interface instantiation" << std::endl;
  }

  std::string z_type;
  if (params_->getNumParams("discretized") > 0)
  {
    z_type = "spatial field";
  }
  else if (params_->have_dynamic_scalar)
  {
    z_type = "transient vector";
  }
  else
  {
    z_type = "vector";
  }
  HDSA::Ptr<HDSA::MD_z_Hyperparameter_Interface<ScalarT>> z_hyperparam_interface = HDSA::makePtr<MD_z_Hyperparameter_Interface_MrHyDE<ScalarT>>(solver_, params_, comm_, data_interface, random_number_generator, z_type, prior_num_state_solves);
  z_hyperparam_interface->Set_alpha_z(alpha_z);

  HDSA::Ptr<HDSA::MD_z_Prior_Interface<ScalarT>> z_prior_interface;
  if (z_type == "spatial field")
  {
    z_hyperparam_interface->Set_beta_z(beta_z);
    z_prior_interface = HDSA::makePtr<HDSA::MD_Numeric_Laplacian_z_Prior_Interface<ScalarT>>(S, M, data_interface, z_hyperparam_interface, u_prior_interface);
  }
  else if (z_type == "transient vector")
  {
    ScalarT T = solver_->final_time;
    int n_t = solver_->settings->sublist("Solver").get<int>("number of steps", 0);
    int num_controls = params_->getNumParams("active");
    z_hyperparam_interface->Set_beta_t(beta_t_z_prior);
    z_prior_interface = HDSA::makePtr<HDSA::MD_Transient_Vector_z_Prior_Interface<ScalarT>>(data_interface, z_hyperparam_interface, u_prior_interface, n_t, T, num_controls);
  }
  else if (z_type == "vector")
  {
    z_prior_interface = HDSA::makePtr<HDSA::MD_Vector_z_Prior_Interface<ScalarT>>(data_interface, z_hyperparam_interface, u_prior_interface);
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////// Output_Writer ////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////

  std::string opt_solution_exo_file_ = data_load_list.get<std::string>("OptimalSolutionExoFile", "error");
  bool write_exo = true;
  if (opt_solution_exo_file_ == "error")
  {
    write_exo = false;
  }
  HDSA::Ptr<Output_Writer_MrHyDE<ScalarT>> output_writer = HDSA::makePtr<Output_Writer_MrHyDE<ScalarT>>(postproc_, solver_, write_exo);
  output_writer->Write_Hyperparameters(u_hyperparam_interface, z_hyperparam_interface);

  ///////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////// Prior Discrepancy Analysis ///////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////

  if ((num_prior_samples > 0) & execute_prior_discrepancy_sampling)
  {
    if (hdsa_verbosity > 1)
    {
      std::cout << "Beginning prior discrepancy analysis" << std::endl;
    }

    HDSA::Ptr<HDSA::MD_Prior_Sampling<ScalarT>> prior_sampling = HDSA::makePtr<HDSA::MD_Prior_Sampling<ScalarT>>(data_interface, u_prior_interface, z_prior_interface);
    HDSA::Ptr<HDSA::MultiVector<ScalarT>> spatial_coords = data_interface->Read_Spatial_Node_Data();
    prior_sampling->Generate_Prior_Discrepancy_Sample_Data(num_prior_samples, u_hyperparam_interface, z_hyperparam_interface, spatial_coords);

    HDSA::Ptr<HDSA::MultiVector<ScalarT>> prior_delta_z_opt = prior_sampling->Get_prior_delta_z_opt();
    std::vector<HDSA::Ptr<HDSA::Vector<ScalarT>>> prior_z_pert = prior_sampling->Get_prior_z_pert();
    std::vector<HDSA::Ptr<HDSA::MultiVector<ScalarT>>> prior_delta_z_pert = prior_sampling->Get_prior_delta_z_pert();
    output_writer->Write_Prior_Discrepancy_Samples(prior_delta_z_opt, prior_z_pert, prior_delta_z_pert);

    if (is_transient)
    {
      std::vector<std::vector<std::vector<ScalarT>>> prior_delta_z_opt_time_evol = prior_sampling->Get_prior_delta_z_opt_time_evol();
      std::vector<std::vector<ScalarT>> prior_discrep_data_time_evol = prior_sampling->Get_prior_discrep_data_time_evol();
      output_writer->Write_Prior_Discrepancy_Time_Evolution(prior_delta_z_opt_time_evol, prior_discrep_data_time_evol);
    }
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////// Posterior Discrepancy Analysis ///////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////

  HDSA::Ptr<HDSA::MD_Posterior_Sampling<ScalarT>> post_sampling = HDSA::makePtr<HDSA::MD_Posterior_Sampling<ScalarT>>(data_interface, u_prior_interface, z_prior_interface);
  if (execute_posterior_discrepancy_sampling || execute_optimal_solution_update)
  {
    if (hdsa_verbosity > 1)
    {
      std::cout << "Beginning posterior data computation" << std::endl;
    }

    post_sampling->Compute_Posterior_Data(u_hyperparam_interface->Get_alpha_d(), num_posterior_samples);
  }

  if ((num_posterior_samples > 0) & execute_posterior_discrepancy_sampling)
  {
    if (hdsa_verbosity > 1)
    {
      std::cout << "Beginning posterior discrepancy analysis" << std::endl;
    }

    std::vector<HDSA::Ptr<HDSA::Vector<ScalarT>>> z_in;
    int N = data_interface->Get_Z()->Number_of_Vectors();
    z_in.resize(N);
    for (int k = 0; k < N; k++)
    {
      z_in[k] = (*data_interface->Get_Z())[k];
    }
    std::vector<HDSA::Ptr<HDSA::MD_Posterior_Vectors<ScalarT>>> post_delta = post_sampling->Posterior_Discrepancy_Samples(z_in);
    output_writer->Write_Posterior_Discrepancy_Samples(post_delta);
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////// Posterior Optimal Solution Analysis //////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////

  if (execute_optimal_solution_update)
  {
    HDSA::Ptr<HDSA::MD_Hessian_Analysis<ScalarT>> hessian_analysis = HDSA::makePtr<HDSA::MD_Hessian_Analysis<ScalarT>>(opt_prob_interface, z_prior_interface);
    if (hessian_num_eig_vals > 0)
    {
      if (hdsa_verbosity > 1)
      {
        std::cout << "Beginning Hessian analysis" << std::endl;
      }

      hessian_analysis->Compute_Hessian_GEVP(data_interface->Get_z_opt(), hessian_num_eig_vals, hessian_oversampling, false);
      HDSA::Ptr<HDSA::Dense_Matrix<ScalarT>> evals = hessian_analysis->Get_Evals();
      output_writer->Write_Hessian_Eigenvalues(evals);
    }

    HDSA::Ptr<HDSA::MD_Update<ScalarT>> update = HDSA::makePtr<HDSA::MD_Update<ScalarT>>(data_interface, u_prior_interface, z_prior_interface, opt_prob_interface, post_sampling, hessian_analysis);
    if (num_posterior_samples > 0)
    {
      if (hdsa_verbosity > 1)
      {
        std::cout << "Beginning posterior optimal solution analysis" << std::endl;
      }

      HDSA::Ptr<HDSA::MD_Posterior_Vectors<ScalarT>> posterior_update_samples = update->Posterior_Update_Samples();
      output_writer->Write_Optimal_Solution_Update(posterior_update_samples);

      if (hdsa_verbosity > 0)
      {
        std::cout << "z_update_mean norm = " << posterior_update_samples->mean->Norm() << std::endl;
      }
    }
    else
    {
      HDSA::Ptr<HDSA::Vector<ScalarT>> z_update_mean = update->Posterior_Update_Mean();
      output_writer->Write_Optimal_Solution_Update(z_update_mean);

      if (hdsa_verbosity > 0)
      {
        std::cout << "z_update_mean norm = " << z_update_mean->Norm() << std::endl;
      }
    }
  }
}
#endif

// ========================================================================================
// ========================================================================================

void AnalysisManager::DCISolve()
{

  // This needs to be performed by all processors
  vector<Teuchos::Array<ScalarT>> response_values = this->UQSolve();

  // Only one rank is necessary for the rest
  if (comm_->getRank() == 0)
  {
    // Evaluate model or a surrogate at these samples
    size_t Nresp = response_values.size();
    size_t resp_dim = response_values[0].size();

    View_Sc2 predvals("predicted data", Nresp, resp_dim);
    for (size_t i = 0; i < Nresp; ++i)
    { // TMW: this is me being lazy - not careful about CPU vs GPU
      for (size_t j = 0; j < resp_dim; ++j)
      {
        predvals(i, j) = response_values[i][j];
      }
    }

    // Create an empty UQ manager (just for the tools)
    UQManager UQ;

    // Get the DCI sublist
    Teuchos::ParameterList dcisettings_ = settings_->sublist("Analysis").sublist("DCI");

    // Need to evaluate the observed density at response samples using either an analytic density or a KDE built from data2
    string obs_type = dcisettings_.get<string>("observed type", "Gaussian"); // other options: uniform or data

    View_Sc1 obsdens("observed density values", Nresp);

    if (obs_type == "Gaussian")
    { // assumes 1D
      ScalarT obs_mean = dcisettings_.get<ScalarT>("observed mean", 0.0);
      ScalarT obs_var = dcisettings_.get<ScalarT>("observed variance", 1.0);
      for (size_t i = 0; i < obsdens.extent(0); ++i)
      {
        ScalarT diff = predvals(i, 0) - obs_mean;
        obsdens(i) = 1.0 / (std::sqrt(2.0 * PI) * std::sqrt(obs_var)) * std::exp(-1.0 * diff * diff / (2.0 * obs_var));
      }
    }
    else if (obs_type == "uniform")
    { // assumes 1D
      ScalarT obs_min = dcisettings_.get<ScalarT>("observed min", 0.0);
      ScalarT obs_max = dcisettings_.get<ScalarT>("observed max", 1.0);
      ScalarT scale = 1.0 / (obs_max - obs_min);
      for (size_t i = 0; i < obsdens.extent(0); ++i)
      {
        if (predvals(i, 0) > obs_min && predvals(i, 0) < obs_max)
        {
          obsdens(i) = scale;
        }
        else
        {
          obsdens(i) = 0.0;
        }
      }
    }
    else if (obs_type == "data")
    {
      // load in data - should be given by a matrix of size Ndata x resp_dim
      string obs_file = dcisettings_.get<string>("observed file", "observed.dat");
      Data obsdata = Data("observed data", obs_file);

      // Data class is designed for unstructured data sets, but this is structured, so we can simplify
      std::vector<Kokkos::View<ScalarT **, HostDevice>> datavec = obsdata.getData();
      Kokkos::View<ScalarT **, HostDevice> datavals = datavec[0];

      // build KDE and evaluate at the response data points (this is where we need it)
      View_Sc1 tmpdens = UQ.KDE(datavals, predvals);
      obsdens = tmpdens; // change pointer rather than copying data
    }

    // Evaluate the predicted density
    View_Sc1 preddens = UQ.KDE(predvals, predvals);

    // Compute the ratio of observed/predicted
    View_Sc1 ratio("DCI ratio", Nresp);

    for (size_type i = 0; i < ratio.extent(0); ++i)
    {
      ratio(i) = obsdens(i) / preddens(i);
    }

    // Diagnostics
    ScalarT meanr = 0.0, KLdivr = 0.0;
    ScalarT Nrespsc = static_cast<ScalarT>(Nresp);
    for (size_type i = 0; i < ratio.extent(0); ++i)
    {
      meanr += ratio(i) / Nrespsc;
      KLdivr += ratio(i) * std::log(ratio(i) + 1.0e-13) / Nrespsc;
    }

    cout << "DCI diagnostics:" << endl;
    cout << "    Mean of ratio: " << meanr << endl;
    cout << "    Information gained: " << KLdivr << endl;

    // Rejection sampling
    bool reject = dcisettings_.get<bool>("rejection sampling", true);
    int seed = dcisettings_.get<int>("rejection seed", 123);
    Kokkos::View<bool *, HostDevice> accept = UQ.rejectionSampling(ratio, seed);

    if (reject)
    {
      // Print the accepted samples to file
      string sname = "accepted_data.dat";
      std::ofstream ACCOUT(sname.c_str());
      ACCOUT.precision(12);
      vector<ScalarT> meanacc(resp_dim, 0.0), varacc(resp_dim, 0.0);
      int Naccept = 0;
      for (size_t r = 0; r < ratio.extent(0); r++)
      {
        if (accept(r))
        {
          for (size_t j = 0; j < resp_dim; ++j)
          {
            ACCOUT << predvals(r, j) << " ";
            meanacc[j] += predvals(r, j);
          }
          ACCOUT << endl;
          Naccept++;
        }
      }
      ACCOUT.close();

      // Print out the acceptance rate
      ScalarT accrate = static_cast<ScalarT>(Naccept) / static_cast<ScalarT>(Nresp) * 100.0;
      cout << "    Acceptance rate: " << accrate << "%" << endl;

      // Compute the mean and variance of the accepted samples
      // Useful to compare against observed if it is Gaussian

      for (size_t j = 0; j < resp_dim; ++j)
      {
        meanacc[j] *= 1.0 / static_cast<ScalarT>(Naccept);
      }
      for (size_t r = 0; r < ratio.extent(0); r++)
      {
        if (accept(r))
        {
          for (size_t j = 0; j < resp_dim; ++j)
          {
            ScalarT diff = predvals(r, j) - meanacc[j];
            varacc[j] += diff * diff / static_cast<ScalarT>(Naccept - 1);
          }
        }
      }

      // Print the mean and variance
      cout << "    Mean of accepted responses: ";
      for (size_t j = 0; j < resp_dim; ++j)
      {
        cout << meanacc[j] << "  ";
      }
      cout << endl;

      cout << "    Variance of accepted responses: ";
      for (size_t j = 0; j < resp_dim; ++j)
      {
        cout << varacc[j] << "  ";
      }
      cout << endl;
    }

    // Print the DCI data to file
    string sname = "DCI_output.dat";
    std::ofstream DCIOUT(sname.c_str());
    DCIOUT.precision(12);
    for (size_t r = 0; r < ratio.extent(0); r++)
    {
      for (size_t i = 0; i < resp_dim; ++i)
      {
        DCIOUT << predvals(r, i) << " ";
      }
      DCIOUT << preddens(r) << " " << obsdens(r);
      if (reject)
      {
        DCIOUT << "  " << accept(r);
      }
      DCIOUT << endl;
    }
    DCIOUT.close();
  }
}

// ========================================================================================
// ========================================================================================

void AnalysisManager::ScalableDCISolve()
{
  // This differs significantly from the sampling-based DCI solver
  // We first find the Bayesian MAP point, and then the DCI MUD point
  // This is meant for problems where the input parameters are a discretized field

  // Need to check and make sure the user defines everything properly

  // Find the Bayesian MAP point

  // If linear, find the MUD point and low-rank update of covariance

  // If nonlinear, apply partial linearization method
}

// ========================================================================================
// ========================================================================================

void AnalysisManager::ScalableBayesSolve()
{

  // This is meant for problems where the input parameters are a discretized field

  // Need to check and make sure the user defines everything properly

  // Solve for MAP point

  // Construct low-rank update to the covariance matrix

  // Develop some sort of diagnostic to assess validity of Laplace approximation of posterior
}

// ========================================================================================
// ========================================================================================

void AnalysisManager::restartSolve()
{

  Teuchos::ParameterList rstsettings_ = settings_->sublist("Analysis").sublist("Restart");
  string state_file = rstsettings_.get<string>("state file name", "none");
  string param_file = rstsettings_.get<string>("parameter file name", "none");
  string adjoint_file = rstsettings_.get<string>("adjoint file name", "none");
  string disc_param_file = rstsettings_.get<string>("discretized parameter file name", "none");
  string scalar_param_file = rstsettings_.get<string>("scalar parameter file name", "none");
  string mode = rstsettings_.get<string>("mode", "forward");
  string data_type = rstsettings_.get<string>("file type", "text");
  double start_time = rstsettings_.get<double>("start time", 0.0);

  solver_->initial_time = start_time;
  solver_->current_time = start_time;

  ///////////////////////////////////////////////////////////
  // Recover the state
  ///////////////////////////////////////////////////////////

  vector<vector_RCP> forward_solution, adjoint_solution;
  vector_RCP disc_params_;
  vector<ScalarT> scalar_params_;
  if (state_file != "none")
  {
    forward_solution = solver_->getRestartSolution();
    this->recoverSolution(forward_solution[0], data_type, param_file, state_file);
  }

  if (adjoint_file != "none")
  {
    adjoint_solution = solver_->getRestartAdjointSolution();
    this->recoverSolution(adjoint_solution[0], data_type, param_file, adjoint_file);
  }

  if (disc_param_file != "none")
  {
    disc_params_ = params_->getDiscretizedParams();
    this->recoverSolution(disc_params_, data_type, param_file, disc_param_file);
  }
  if (scalar_param_file != "none")
  {
  }

  solver_->use_restart = true;

  ///////////////////////////////////////////////////////////
  // Run the requested mode
  ///////////////////////////////////////////////////////////

  if (mode == "forward")
  {
  }
  else if (mode == "ROL")
  {
  }
  else if (mode == "ROL2")
  {
  }
  else
  { // don't solver_ anything, but produce visualization
    std::cout << "Unknown restart mode: " << mode << std::endl;
  }
}

// ========================================================================================
// ========================================================================================

MrHyDE_OptVector AnalysisManager::recoverParametersFromFile() {

  MrHyDE_OptVector currparams = params_->getCurrentVector();
  string paramfilebase = settings_->sublist("Analysis").get<string>("parameter recovery file", "params");
  // Assumes the parameters are actually stored in files names params.scalar.0.dat or params.field.0.mm where the number represents the time step for dynamic parameters
  
  // First, scalar parameters
  vector<Teuchos::RCP<vector<ScalarT> > > scalar_params;
  
  // Does the current vector have scalars?
  bool have_scalar = currparams.haveScalar();
  
  // Are they dynamic?
  bool have_dyn_scalar = currparams.haveDynamicScalar();
  
  // Read in from file if present
  if (have_scalar) {
    if (have_dyn_scalar) {
      std::vector<ROL::Ptr<ROL::StdVector<ScalarT> > > currscalar = currparams.getParameter();
      size_t numparams = currscalar.size();
      for (size_t t=0; t<numparams; ++t) {
        std::stringstream ss;
        ss << paramfilebase << ".scalar." << t << ".dat";
        
        std::ifstream fin(ss.str());
        if (!fin.good()) {
          TEUCHOS_TEST_FOR_EXCEPTION(!fin.good(),std::runtime_error,"Error: could not find the data file: " + ss.str());
        }
        
        Teuchos::RCP<std::vector<ScalarT> > values = Teuchos::rcp(new std::vector<ScalarT>());
        ScalarT number;
        while (fin >> number) {
          values->push_back(number);
        }
        scalar_params.push_back(values);
        fin.close();
      }
    }
    else {
      std::stringstream ss;
      ss << paramfilebase << ".scalar.0.dat";
      
      std::ifstream fin(ss.str());
      if (!fin.good()) {
        TEUCHOS_TEST_FOR_EXCEPTION(!fin.good(),std::runtime_error,"Error: could not find the data file: " + ss.str());
      }
      
      Teuchos::RCP<std::vector<ScalarT> > values = Teuchos::rcp(new std::vector<ScalarT>());
      ScalarT number;
      while (fin >> number) {
        values->push_back(number);
      }
      scalar_params.push_back(values);
      fin.close();
    }
  }
 
  
  // Next, field parameters
  std::vector<vector_RCP> field_params;
  
  // Does the current vector have fields?
  bool have_field = currparams.haveField();
  
  // Are they dynamic?
  bool have_dyn_field = currparams.haveDynamicField();
  
  // Read in from file if present
  if (have_field) {
    if (have_dyn_field) {
      std::vector<ROL::Ptr<ROL::TpetraMultiVector<ScalarT> > > currfield = currparams.getField();
      size_t numparams = currfield.size();
      for (size_t t=0; t<numparams; ++t) {
        std::stringstream ss;
        ss << paramfilebase << ".field." << t << ".mm";
        
        vector_RCP vec = solver_->linalg->readParameterVectorFromFile(ss.str());
        field_params.push_back(vec);
      }
    }
    else {
      std::stringstream ss;
      ss << paramfilebase << ".field.0.mm";
      
      vector_RCP vec = solver_->linalg->readParameterVectorFromFile(ss.str());
      field_params.push_back(vec);
    }
  }
  
  MrHyDE_OptVector newparams(field_params, scalar_params, 1.0, params_->diagParamMass, params_->paramMass, comm_->getRank());
  return newparams;
  
}

// ========================================================================================
// ========================================================================================

void AnalysisManager::writeOptVectorToFile(MrHyDE_OptVector & vec, string & filebase) {
  // First, scalar parameters
  
  // Does the current vector have scalars?
  bool have_scalar = vec.haveScalar();
  
  // Are they dynamic?
  bool have_dyn_scalar = vec.haveDynamicScalar();
  
  // Scalar parameters are on every processor, so only rank=0 needs to write
  if (have_scalar && comm_->getRank() == 0) {
    std::vector<ROL::Ptr<ROL::StdVector<ScalarT> > > currscalar = vec.getParameter();
    if (have_dyn_scalar) {
      size_t numparams = currscalar.size();
      for (size_t t=0; t<numparams; ++t) {
        std::stringstream ss;
        ss << filebase << ".scalar." << t << ".dat";
        
        std::ofstream fout(ss.str());
        if (!fout.is_open()) {
          TEUCHOS_TEST_FOR_EXCEPTION(!fout.is_open(),std::runtime_error,"Error: could not open the data file: " + ss.str());
        }
        fout.precision(12);
        auto currvec = *(currscalar[t]->getVector());
        for (size_t i=0; i<currvec.size(); ++i) {
          fout << currvec[i] << endl;
        }
        fout.close();
      }
    }
    else {
      std::stringstream ss;
      ss << filebase << ".scalar.0.dat";
      
      std::ofstream fout(ss.str());
      if (!fout.is_open()) {
        TEUCHOS_TEST_FOR_EXCEPTION(!fout.is_open(),std::runtime_error,"Error: could not open the data file: " + ss.str());
      }
      fout.precision(12);
      auto currvec = *(currscalar[0]->getVector());
      for (size_t i=0; i<currvec.size(); ++i) {
        fout << currvec[i] << endl;
      }
      fout.close();
    }
  }
  
  // Does the current vector have fields?
  bool have_field = vec.haveField();
  
  // Are they dynamic?
  bool have_dyn_field = vec.haveDynamicField();
  
  // Write to file - each processors only stores a piece of the field
  if (have_field) {
    std::vector<ROL::Ptr<ROL::TpetraMultiVector<ScalarT> > > currfield = vec.getField();
    if (have_dyn_field) {
      size_t numfields = currfield.size();
      for (size_t t=0; t<numfields; ++t) {
        std::stringstream ss;
        ss << filebase << ".field." << t << ".mm";
        string filename = ss.str();
        ROL::Ptr<ROL::TpetraMultiVector<ScalarT> > currvec = currfield[t];
        solver_->linalg->writeVectorToFile(currvec, filename);
      }
    }
    else {
      std::stringstream ss;
      ss << filebase << ".field.0.mm";
      string filename = ss.str();
      ROL::Ptr<ROL::TpetraMultiVector<ScalarT> > currvec = currfield[0];
      solver_->linalg->writeVectorToFile(currvec, filename);
      
    }
  }
  
}
