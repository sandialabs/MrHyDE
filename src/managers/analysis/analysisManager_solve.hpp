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
  HDSA::Ptr<Driver_MrHyDE<ScalarT>> hdsa_driver = HDSA::makePtr<Driver_MrHyDE<ScalarT>>(comm_, settings_, solver_, postproc_, params_); 
  hdsa_driver->HDSA_Solve();
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
