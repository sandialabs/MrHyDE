/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.

 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
 ************************************************************************/

#include "analysisManager.hpp"
#include "physicsInterface.hpp"
#include "discretizationInterface.hpp"
#include "uqManager.hpp"
#include "data.hpp"
#include "MrHyDE_Objective.hpp"
#include "MrHyDE_Stochastic_Objective.hpp"
#include "MrHyDE_OptVector.hpp"
#include "MrHyDE_Sample_Set_Reader.hpp"
#include "ROL_LineSearchStep.hpp"
#include "ROL_Algorithm.hpp"
#include "ROL_Bounds.hpp"
#include "ROL_TrustRegionStep.hpp"
#include "ROL_Solver.hpp"
#include "ROL_StochasticProblem.hpp"

#include "MrHyDE_TeuchosBatchManager.hpp"
#include "ROL_MonteCarloGenerator.hpp"
#include "ROL_DistributionFactory.hpp"

#if defined(MrHyDE_ENABLE_HDSA)
#include "HDSA_Ptr.hpp"
#include "HDSA_Random_Number_Generator.hpp"
#include "HDSA_MD_Data_Interface_MrHyDE.hpp"
#include "HDSA_MD_Opt_Prob_Interface_MrHyDE.hpp"
#include "HDSA_Write_Output_MrHyDE.hpp"
#include "HDSA_Sparse_Matrix.hpp"
#include "HDSA_MD_u_Hyperparameter_Interface.hpp"
#include "HDSA_MD_z_Hyperparameter_Interface_MrHyDE.hpp"
#include "HDSA_Prior_FE_Op_MrHyDE.hpp"
#include "HDSA_MD_u_Prior_Interface.hpp"
#include "HDSA_MD_Numeric_Laplacian_u_Prior_Interface.hpp"
#include "HDSA_MD_z_Prior_Interface.hpp"
#include "HDSA_MD_Numeric_Laplacian_z_Prior_Interface.hpp"
#include "HDSA_MD_Vector_z_Prior_Interface.hpp"
#include "HDSA_MD_Prior_Sampling.hpp"
#include "HDSA_MD_Posterior_Sampling.hpp"
#include "HDSA_MD_Hessian_Analysis.hpp"
#include "HDSA_MD_Update.hpp"
#include "HDSA_MD_OUU_Data_Interface_MrHyDE.hpp"
#include "HDSA_MD_OUU_Opt_Prob_Interface_MrHyDE.hpp"
#include "HDSA_MD_OUU_Hyperparameter_Data_Interface.hpp"
#include "HDSA_MD_OUU_u_Prior_Interface.hpp"

#include "ROL_PrimalDualRisk.hpp"
#endif

using namespace MrHyDE;

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

void AnalysisManager::run(std::string &analysis_type)
{

  if (analysis_type == "forward")
  {

    this->forwardSolve();
  }
  else if (analysis_type == "forward+adjoint")
  {

    this->forwardSolve();

    MrHyDE_OptVector sens = this->adjointSolve();
  }
  else if (analysis_type == "dry run")
  {
    cout << " **** MrHyDE has completed the dry run with verbosity: " << verbosity_ << endl;
  }
  else if (analysis_type == "UQ")
  {
    vector<Teuchos::Array<ScalarT>> response_values = this->UQSolve();
  }
  else if (analysis_type == "ROL")
  {
    this->ROLSolve();
  }
  else if (analysis_type == "ROL2")
  {
    this->ROL2Solve();
  }
  else if (analysis_type == "ROLStoch")
  {
    this->ROLStochSolve();
  }
#if defined(MrHyDE_ENABLE_HDSA)
  else if (analysis_type == "HDSA")
  {
    this->HDSASolve();
  }
  else if (analysis_type == "HDSAStoch")
  {
    this->HDSAStochSolve();
  }
  else if (analysis_type == "readExo+forward")
  {
    this->readExoForwardSolve();
  }
#endif
  else if (analysis_type == "DCI")
  {
    this->DCISolve();
  }
  else if (analysis_type == "Scalable DCI")
  {
    this->ScalableDCISolve();
  }
  else if (analysis_type == "Scalable Bayes")
  {
    this->ScalableBayesSolve();
  }
  else if (analysis_type == "restart")
  {
    this->restartSolve();
  }
  else
  { // don't solver_ anything, but produce visualization
    std::cout << "Unknown analysis option: " << analysis_type << std::endl;
    std::cout << "Valid and tested options: dry run, forward, forward+adjoint, UQ, ROL, ROL2, DCI" << std::endl;
  }
}

// ========================================================================================
// ========================================================================================

void AnalysisManager::forwardSolve()
{

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

void AnalysisManager::recoverSolution(vector_RCP &solution, string &data_type,
                                      string &plist_filename, string &filename)
{

  string extension = filename.substr(filename.size() - 4, filename.size() - 1);
  filename.erase(filename.size() - 4, 4);

  cout << extension << "  " << filename << endl;
  if (data_type == "text")
  {
    std::stringstream sfile;
    sfile << filename << "." << comm_->getRank() << extension;
    std::ifstream fnmast(sfile.str());
    if (!fnmast.good())
    {
      TEUCHOS_TEST_FOR_EXCEPTION(!fnmast.good(), std::runtime_error, "Error: could not find the data file: " + sfile.str());
    }

    std::vector<std::vector<ScalarT>> values;
    std::ifstream fin(sfile.str());

    for (std::string line; std::getline(fin, line);)
    {
      std::replace(line.begin(), line.end(), ',', ' ');
      std::istringstream in(line);
      values.push_back(std::vector<ScalarT>(std::istream_iterator<ScalarT>(in),
                                            std::istream_iterator<ScalarT>()));
    }

    typedef typename SolverNode::device_type LA_device;
    auto sol_view = solution->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
    for (size_type i = 0; i < values.size(); ++i)
    {
      sol_view(i, 0) = values[i][0];
    }
  }
  else if (data_type == "exodus")
  {
  }
  else if (data_type == "hdf5")
  {
  }
  else if (data_type == "binary")
  {
  }
  else
  {
    std::cout << "Unknown file type: " << data_type << std::endl;
  }
}

// ========================================================================================
// ========================================================================================

void AnalysisManager::updateRotationData(const int &newrandseed)
{

  // Determine how many seeds there are
  size_t localnumSeeds = 0;
  size_t numSeeds = 0;
  for (size_t block = 0; block < solver_->assembler->groups.size(); ++block)
  {
    for (size_t grp = 0; grp < solver_->assembler->groups[block].size(); ++grp)
    {
      for (size_t e = 0; e < solver_->assembler->groups[block][grp]->numElem; ++e)
      {
        if (solver_->assembler->groups[block][grp]->data_seed[e] > localnumSeeds)
        {
          localnumSeeds = solver_->assembler->groups[block][grp]->data_seed[e];
        }
      }
    }
  }
  // comm_->MaxAll(&localnumSeeds, &numSeeds, 1);
  Teuchos::reduceAll<int, size_t>(*comm_, Teuchos::REDUCE_MAX, 1, &localnumSeeds, &numSeeds);
  numSeeds += 1; // To properly allocate and iterate

  // Create a random number generator
  std::default_random_engine generator(newrandseed);

  ////////////////////////////////////////////////////////////////////////////////
  // Set seed data
  ////////////////////////////////////////////////////////////////////////////////

  int numdata = 9;

  // cout << "solver_r numSeeds = " << numSeeds << endl;

  std::normal_distribution<ScalarT> ndistribution(0.0, 1.0);
  Kokkos::View<ScalarT **, HostDevice> rotation_data("cell_data", numSeeds, numdata);
  for (size_t k = 0; k < numSeeds; k++)
  {
    ScalarT x = ndistribution(generator);
    ScalarT y = ndistribution(generator);
    ScalarT z = ndistribution(generator);
    ScalarT w = ndistribution(generator);

    ScalarT r = sqrt(x * x + y * y + z * z + w * w);
    x *= 1.0 / r;
    y *= 1.0 / r;
    z *= 1.0 / r;
    w *= 1.0 / r;

    rotation_data(k, 0) = w * w + x * x - y * y - z * z;
    rotation_data(k, 1) = 2.0 * (x * y - w * z);
    rotation_data(k, 2) = 2.0 * (x * z + w * y);

    rotation_data(k, 3) = 2.0 * (x * y + w * z);
    rotation_data(k, 4) = w * w - x * x + y * y - z * z;
    rotation_data(k, 5) = 2.0 * (y * z - w * x);

    rotation_data(k, 6) = 2.0 * (x * z - w * y);
    rotation_data(k, 7) = 2.0 * (y * z + w * x);
    rotation_data(k, 8) = w * w - x * x - y * y + z * z;
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Set cell data
  ////////////////////////////////////////////////////////////////////////////////

  for (size_t block = 0; block < solver_->assembler->groups.size(); ++block)
  {
    for (size_t grp = 0; grp < solver_->assembler->groups[block].size(); ++grp)
    {
      int numElem = solver_->assembler->groups[block][grp]->numElem;
      for (int c = 0; c < numElem; c++)
      {
        int cnode = solver_->assembler->groups[block][grp]->data_seed[c];
        for (int i = 0; i < 9; i++)
        {
          solver_->assembler->groups[block][grp]->data(c, i) = rotation_data(cnode, i);
        }
      }
    }
  }
  for (size_t block = 0; block < solver_->assembler->boundary_groups.size(); ++block)
  {
    for (size_t grp = 0; grp < solver_->assembler->boundary_groups[block].size(); ++grp)
    {
      int numElem = solver_->assembler->boundary_groups[block][grp]->numElem;
      for (int e = 0; e < numElem; ++e)
      {
        int cnode = solver_->assembler->boundary_groups[block][grp]->data_seed[e];
        for (int i = 0; i < 9; i++)
        {
          solver_->assembler->boundary_groups[block][grp]->data(e, i) = rotation_data(cnode, i);
        }
      }
    }
  }
  solver_->multiscale_manager->updateMeshData(rotation_data);
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
    string sample_file = uqsettings_.get<string>("samples output file", "sample_inputs.dat");
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

    string sname = "sample_output.dat";
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
  if ( (ROLsettings.sublist("SOL").get("Sample Set File", "error") == "error") || (ROLsettings.sublist("SOL").get("Sample Weight File", "error") == "error") )
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

  if (ROLsettings.sublist("General").get("Write Iteration History", false) && (comm_->getRank() == 0))
  {
    string outname = ROLsettings.get("Output File Name", "ROL_out.txt");
    freopen(outname.c_str(), "w", stdout);
  }
  Kokkos::fence();

  int N = sampler->numMySamples();
  std::vector<RealT> sample_weights = std::vector<RealT>(N,0.0);
  if (!ROLsettings.sublist("SOL").get("Use Primal Dual", false))
  {
    ROL::Ptr<ROL::StochasticProblem<RealT>> rolProblem = ROL::makePtr<ROL::StochasticProblem<RealT>>(obj, x);
    rolProblem->makeObjectiveStochastic(ROLsettings, sampler);
    rolProblem->finalize(false, false, *outStream);
    ROL::Solver<ScalarT> rolSolver(rolProblem, ROLsettings);
    rolSolver.solve(*outStream);
    for(int i = 0; i < N; i++)
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
    std::vector<ScalarT> obj_vals = std::vector<ScalarT>(sampler->numMySamples(),0.0);
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
  Teuchos::ParameterList HDSAsettings;

  if (settings_->sublist("Analysis").isSublist("HDSA"))
    HDSAsettings = settings_->sublist("Analysis").sublist("HDSA");
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error: MrHyDE could not find the HDSA sublist in the input file!  Abort!");

  int num_prior_samples = HDSAsettings.sublist("Configuration").get<int>("num_prior_samples", 0);
  int num_posterior_samples = HDSAsettings.sublist("Configuration").get<int>("num_posterior_samples", 0);
  int prior_num_state_solves = HDSAsettings.sublist("Configuration").get<int>("prior_num_state_solves", 0);
  bool execute_prior_discrepancy_sampling = HDSAsettings.sublist("Configuration").get<bool>("execute_prior_discrepancy_sampling", false);
  bool execute_posterior_discrepancy_sampling = HDSAsettings.sublist("Configuration").get<bool>("execute_posterior_discrepancy_sampling", false);
  bool execute_optimal_solution_update = HDSAsettings.sublist("Configuration").get<bool>("execute_optimal_solution_update", false);

  ScalarT alpha_u = HDSAsettings.sublist("HyperParameters").get<ScalarT>("alpha_u", 0.0);
  ScalarT alpha_z = HDSAsettings.sublist("HyperParameters").get<ScalarT>("alpha_z", 0.0);
  ScalarT beta_u = HDSAsettings.sublist("HyperParameters").get<ScalarT>("beta_u", 0.0);
  ScalarT beta_t = HDSAsettings.sublist("HyperParameters").get<ScalarT>("beta_t", 0.0);
  ScalarT beta_z = HDSAsettings.sublist("HyperParameters").get<ScalarT>("beta_z", 0.0);
  ScalarT alpha_d = HDSAsettings.sublist("HyperParameters").get<ScalarT>("alpha_d", 0.0);
  int prior_num_sing_vals = HDSAsettings.sublist("HyperParameters").get<int>("prior_num_sing_vals", 200);
  int prior_oversampling = HDSAsettings.sublist("HyperParameters").get<int>("prior_oversampling", 20);
  int prior_num_subspace_iter = HDSAsettings.sublist("HyperParameters").get<int>("prior_num_subspace_iter", 1);
  int hessian_num_eig_vals = HDSAsettings.sublist("HyperParameters").get<int>("hessian_num_eig_vals", 5);
  int hessian_oversampling = HDSAsettings.sublist("HyperParameters").get<int>("hessian_oversampling", 3);

  Teuchos::ParameterList data_load_list = HDSAsettings.sublist("DataLoadParameters");
  std::string random_number_file = data_load_list.get<std::string>("random_number_file", "error");
  int num_random_numbers = data_load_list.get<int>("num_random_numbers", 0);
  HDSA::Ptr<HDSA::Random_Number_Generator<ScalarT>> random_number_generator;
  if (random_number_file == "error")
  {
    random_number_generator = HDSA::makePtr<HDSA::Random_Number_Generator<ScalarT>>();
  }
  else
  {
    if (num_random_numbers == 0)
    {
      std::cout << " Error: number of random numbers not specified" << std::endl;
    }
    random_number_generator = HDSA::makePtr<HDSA::Random_Number_Generator<ScalarT>>(num_random_numbers, random_number_file);
  }

  postproc_->write_solution = false;
  postproc_->write_optimization_solution = false;
  HDSA::Ptr<MD_Data_Interface_MrHyDE<ScalarT>> data_interface = HDSA::makePtr<MD_Data_Interface_MrHyDE<ScalarT>>(comm_, solver_, params_, random_number_generator, data_load_list);
  HDSA::Ptr<HDSA::MD_Opt_Prob_Interface<ScalarT>> opt_prob_interface = HDSA::makePtr<MD_Opt_Prob_Interface_MrHyDE<ScalarT>>(solver_, postproc_, params_, data_interface);
  HDSA::Ptr<Write_Output_MrHyDE<ScalarT>> output_writer = HDSA::makePtr<Write_Output_MrHyDE<ScalarT>>(data_interface, postproc_, solver_);

  vector<string> blockNames = solver_->mesh->getBlockNames();
  HDSA::Ptr<Prior_FE_Op_MrHyDE<ScalarT>> prior_fe_op = HDSA::makePtr<Prior_FE_Op_MrHyDE<ScalarT>>(comm_, settings_, blockNames);
  HDSA::Ptr<HDSA::Sparse_Matrix<ScalarT>> M = HDSA::makePtr<HDSA::Sparse_Matrix<ScalarT>>(prior_fe_op->M[0]);
  HDSA::Ptr<HDSA::Sparse_Matrix<ScalarT>> S = HDSA::makePtr<HDSA::Sparse_Matrix<ScalarT>>(prior_fe_op->S[0]);

  bool is_transient = solver_->isTransient;
  HDSA::Ptr<HDSA::MD_u_Hyperparameter_Interface<ScalarT>> u_hyperparam_interface;
  HDSA::Ptr<HDSA::MD_u_Prior_Interface<ScalarT>> u_prior_interface;
  if (is_transient)
  {
    u_hyperparam_interface = HDSA::makePtr<HDSA::MD_u_Hyperparameter_Interface<ScalarT>>(is_transient);
    u_hyperparam_interface->Set_alpha_d(alpha_d);
    u_hyperparam_interface->Set_alpha_u(alpha_u);
    u_hyperparam_interface->Set_beta_u(beta_u);
    u_hyperparam_interface->Set_beta_t(beta_t);
    u_hyperparam_interface->Set_GSVD_Hyperparameters(prior_num_sing_vals, prior_oversampling, prior_num_subspace_iter);
    HDSA::Ptr<HDSA::MD_u_Prior_Interface<ScalarT>> spatial_u_prior_interface = HDSA::makePtr<HDSA::MD_Numeric_Laplacian_u_Prior_Interface<ScalarT>>(S, M, data_interface, u_hyperparam_interface, random_number_generator);
    ScalarT T = solver_->final_time;
    int n_t = solver_->settings->sublist("Solver").get<int>("number of steps", 0) + 1;
    int n_y = data_interface->get_u_opt()->dimension() / n_t;
    HDSA::Ptr<HDSA::MD_Transient_Prior_Covariance<ScalarT>> transient_prior_cov = HDSA::makePtr<HDSA::MD_Transient_Prior_Covariance<ScalarT>>(data_interface, u_hyperparam_interface, T, n_t, n_y);
    u_prior_interface = HDSA::makePtr<HDSA::MD_Transient_Elliptic_u_Prior_Interface<ScalarT>>(spatial_u_prior_interface, transient_prior_cov);
  }
  else
  {
    u_hyperparam_interface = HDSA::makePtr<HDSA::MD_u_Hyperparameter_Interface<ScalarT>>(is_transient);
    u_hyperparam_interface->Set_alpha_d(alpha_d);
    u_hyperparam_interface->Set_alpha_u(alpha_u);
    u_hyperparam_interface->Set_beta_u(beta_u);
    u_hyperparam_interface->Set_GSVD_Hyperparameters(prior_num_sing_vals, prior_oversampling, prior_num_subspace_iter);
    u_prior_interface = HDSA::makePtr<HDSA::MD_Numeric_Laplacian_u_Prior_Interface<ScalarT>>(S, M, data_interface, u_hyperparam_interface, random_number_generator);
  }

  HDSA::Ptr<HDSA::MD_z_Hyperparameter_Interface<ScalarT>> z_hyperparam_interface = HDSA::makePtr<MD_z_Hyperparameter_Interface_MrHyDE<ScalarT>>(data_interface, random_number_generator, "spatial field", prior_num_state_solves);
  z_hyperparam_interface->Set_alpha_z(alpha_z);
  z_hyperparam_interface->Set_beta_z(beta_z);
  HDSA::Ptr<HDSA::MD_Numeric_Laplacian_z_Prior_Interface<ScalarT>> z_prior_interface = HDSA::makePtr<HDSA::MD_Numeric_Laplacian_z_Prior_Interface<ScalarT>>(S, M, data_interface, z_hyperparam_interface, u_prior_interface);

  output_writer->Write_Hyperparameters(u_hyperparam_interface, z_hyperparam_interface);

  if ((num_prior_samples > 0) & execute_prior_discrepancy_sampling)
  {
    HDSA::Ptr<HDSA::MD_Prior_Sampling<ScalarT>> prior_sampling = HDSA::makePtr<HDSA::MD_Prior_Sampling<ScalarT>>(data_interface, u_prior_interface, z_prior_interface);
    HDSA::Ptr<HDSA::MultiVector<ScalarT>> spatial_coords = data_interface->Read_Spatial_Node_Data();
    prior_sampling->Generate_Prior_Discrepancy_Sample_Data(num_prior_samples, u_hyperparam_interface, z_hyperparam_interface, spatial_coords);

    HDSA::Ptr<HDSA::MultiVector<ScalarT>> prior_delta_z_opt = prior_sampling->Get_prior_delta_z_opt();
    std::vector<HDSA::Ptr<HDSA::Vector<ScalarT>>> prior_z_pert = prior_sampling->Get_prior_z_pert();
    std::vector<HDSA::Ptr<HDSA::MultiVector<ScalarT>>> prior_delta_z_pert = prior_sampling->Get_prior_delta_z_pert();
    output_writer->Write_Prior_Discrepancy_Samples(prior_delta_z_opt, prior_z_pert, prior_delta_z_pert);
  }

  HDSA::Ptr<HDSA::MD_Posterior_Sampling<ScalarT>> post_sampling = HDSA::makePtr<HDSA::MD_Posterior_Sampling<ScalarT>>(data_interface, u_prior_interface, z_prior_interface);
  if (execute_posterior_discrepancy_sampling || execute_optimal_solution_update)
  {
    post_sampling->Compute_Posterior_Data(u_hyperparam_interface->Get_alpha_d(), num_posterior_samples);
  }

  if ((num_posterior_samples > 0) & execute_posterior_discrepancy_sampling)
  {
    std::vector<HDSA::Ptr<HDSA::Vector<ScalarT>>> z_in;
    int N = data_interface->get_Z()->Number_of_Vectors();
    z_in.resize(N);
    for (int k = 0; k < N; k++)
    {
      z_in[k] = (*data_interface->get_Z())[k];
    }
    std::vector<HDSA::Ptr<HDSA::MD_Posterior_Vectors<ScalarT>>> post_delta = post_sampling->Posterior_Discrepancy_Samples(z_in);
    output_writer->Write_Posterior_Discrepancy_Samples(post_delta);
  }

  if (execute_optimal_solution_update)
  {
    HDSA::Ptr<HDSA::MD_Hessian_Analysis<ScalarT>> hessian_analysis = HDSA::makePtr<HDSA::MD_Hessian_Analysis<ScalarT>>(opt_prob_interface, z_prior_interface);
    if (hessian_num_eig_vals > 0)
    {
      hessian_analysis->Compute_Hessian_GEVP(*data_interface->get_z_opt(), hessian_num_eig_vals, hessian_oversampling, false);
      HDSA::Ptr<HDSA::Dense_Matrix<ScalarT>> evals = hessian_analysis->Get_Evals();
      output_writer->Write_Hessian_Eigenvalues(evals);
    }

    HDSA::Ptr<HDSA::MD_Update<ScalarT>> update = HDSA::makePtr<HDSA::MD_Update<ScalarT>>(data_interface, u_prior_interface, z_prior_interface, opt_prob_interface, post_sampling, hessian_analysis);
    if (num_posterior_samples > 0)
    {
      HDSA::Ptr<HDSA::MD_Posterior_Vectors<ScalarT>> posterior_update_samples = update->Posterior_Update_Samples();
      output_writer->Write_Optimal_Solution_Update(posterior_update_samples);
      std::cout << "z_update_mean norm = " << posterior_update_samples->mean->norm() << std::endl;
    }
    else
    {
      HDSA::Ptr<HDSA::Vector<ScalarT>> z_update_mean = update->Posterior_Update_Mean();
      output_writer->Write_Optimal_Solution_Update(z_update_mean);
      std::cout << "z_update_mean norm = " << z_update_mean->norm() << std::endl;
    }
  }
}

void AnalysisManager::HDSAStochSolve()
{
  Teuchos::ParameterList HDSAsettings;

  if (settings_->sublist("Analysis").isSublist("HDSA"))
    HDSAsettings = settings_->sublist("Analysis").sublist("HDSA");
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error: MrHyDE could not find the HDSA sublist in the input file!  Abort!");

  int num_prior_samples = HDSAsettings.sublist("Configuration").get<int>("num_prior_samples", 0);
  int num_posterior_samples = HDSAsettings.sublist("Configuration").get<int>("num_posterior_samples", 0);
  int prior_num_state_solves = HDSAsettings.sublist("Configuration").get<int>("prior_num_state_solves", 0);
  bool execute_prior_discrepancy_sampling = HDSAsettings.sublist("Configuration").get<bool>("execute_prior_discrepancy_sampling", false);
  bool execute_posterior_discrepancy_sampling = HDSAsettings.sublist("Configuration").get<bool>("execute_posterior_discrepancy_sampling", false);
  bool execute_optimal_solution_update = HDSAsettings.sublist("Configuration").get<bool>("execute_optimal_solution_update", false);

  ScalarT alpha_u = HDSAsettings.sublist("HyperParameters").get<ScalarT>("alpha_u", 0.0);
  ScalarT alpha_z = HDSAsettings.sublist("HyperParameters").get<ScalarT>("alpha_z", 0.0);
  ScalarT beta_u = HDSAsettings.sublist("HyperParameters").get<ScalarT>("beta_u", 0.0);
  ScalarT beta_t = HDSAsettings.sublist("HyperParameters").get<ScalarT>("beta_t", 0.0);
  ScalarT beta_z = HDSAsettings.sublist("HyperParameters").get<ScalarT>("beta_z", 0.0);
  ScalarT alpha_d = HDSAsettings.sublist("HyperParameters").get<ScalarT>("alpha_d", 0.0);
  int prior_num_sing_vals = HDSAsettings.sublist("HyperParameters").get<int>("prior_num_sing_vals", 200);
  int prior_oversampling = HDSAsettings.sublist("HyperParameters").get<int>("prior_oversampling", 20);
  int prior_num_subspace_iter = HDSAsettings.sublist("HyperParameters").get<int>("prior_num_subspace_iter", 1);
  int hessian_num_eig_vals = HDSAsettings.sublist("HyperParameters").get<int>("hessian_num_eig_vals", 5);
  int hessian_oversampling = HDSAsettings.sublist("HyperParameters").get<int>("hessian_oversampling", 3);

  Teuchos::ParameterList data_load_list = HDSAsettings.sublist("DataLoadParameters");
  std::string random_number_file = data_load_list.get<std::string>("random_number_file", "error");
  int num_random_numbers = data_load_list.get<int>("num_random_numbers", 0);
  HDSA::Ptr<HDSA::Random_Number_Generator<ScalarT>> random_number_generator;
  if (random_number_file == "error")
  {
    random_number_generator = HDSA::makePtr<HDSA::Random_Number_Generator<ScalarT>>();
  }
  else
  {
    if (num_random_numbers == 0)
    {
      std::cout << " Error: number of random numbers not specified" << std::endl;
    }
    random_number_generator = HDSA::makePtr<HDSA::Random_Number_Generator<ScalarT>>(num_random_numbers, random_number_file);
  }

  postproc_->write_solution = false;
  postproc_->write_optimization_solution = false;

  HDSA::Ptr<ROL::BatchManager<ScalarT>> bman = HDSA::makePtr<ROL::MrHyDETeuchosBatchManager<ScalarT, int>>(comm_);
  int ens_size = data_load_list.get<int>("Ensemble Size", 100);
  int dim = params_->getNumParams("stochastic");
  std::string sample_pt_file = data_load_list.get("Sample Set File", "error");
  std::string sample_wt_file = data_load_list.get("Sample Weight File", "error");
  HDSA::Ptr<ROL::SampleGenerator<ScalarT>> sampler = HDSA::makePtr<ROL::Sample_Set_Reader<ScalarT>>(ens_size, dim, bman, sample_pt_file, sample_wt_file);

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
  HDSA::Ptr<HDSA::MD_OUU_Data_Interface<ScalarT>> data_interface = HDSA::makePtr<MD_OUU_Data_Interface_MrHyDE<ScalarT>>(data_interface_ens, sampler, ens_size);
  HDSA::Ptr<HDSA::MD_OUU_Hyperparameter_Data_Interface<ScalarT>> data_interface_hyperparam = HDSA::makePtr<HDSA::MD_OUU_Hyperparameter_Data_Interface<ScalarT>>(data_interface);

  HDSA::Ptr<HDSA::MD_Opt_Prob_Interface<ScalarT>> opt_prob_interface_s = HDSA::makePtr<MD_Opt_Prob_Interface_MrHyDE<ScalarT>>(solver_, postproc_, params_, data_interface);

  std::vector<ScalarT> ens_weights = std::vector<ScalarT>(ens_size, 0.0);
  for (int i = 0; i < ens_size; i++)
  {
    ens_weights[i] = sampler->getMyWeight(i);
  }
  HDSA::Ptr<HDSA::MD_Opt_Prob_Interface<ScalarT>> opt_prob_interface = HDSA::makePtr<MD_OUU_Opt_Prob_Interface_MrHyDE<ScalarT>>(opt_prob_interface_s, params_, sampler, ens_weights);

  HDSA::Ptr<Write_Output_MrHyDE<ScalarT>> output_writer = HDSA::makePtr<Write_Output_MrHyDE<ScalarT>>(data_interface_ens[0], postproc_, solver_);

  vector<string> blockNames = solver_->mesh->getBlockNames();
  HDSA::Ptr<Prior_FE_Op_MrHyDE<ScalarT>> prior_fe_op = HDSA::makePtr<Prior_FE_Op_MrHyDE<ScalarT>>(comm_, settings_, blockNames);
  HDSA::Ptr<HDSA::Sparse_Matrix<ScalarT>> M = HDSA::makePtr<HDSA::Sparse_Matrix<ScalarT>>(prior_fe_op->M[0]);
  HDSA::Ptr<HDSA::Sparse_Matrix<ScalarT>> S = HDSA::makePtr<HDSA::Sparse_Matrix<ScalarT>>(prior_fe_op->S[0]);

  bool is_transient = solver_->isTransient;
  HDSA::Ptr<HDSA::MD_u_Hyperparameter_Interface<ScalarT>> us_hyperparam_interface;
  HDSA::Ptr<HDSA::MD_u_Prior_Interface<ScalarT>> us_prior_interface;
  if (is_transient)
  {
    us_hyperparam_interface = HDSA::makePtr<HDSA::MD_u_Hyperparameter_Interface<ScalarT>>(is_transient);
    us_hyperparam_interface->Set_alpha_d(alpha_d);
    us_hyperparam_interface->Set_alpha_u(alpha_u);
    us_hyperparam_interface->Set_beta_u(beta_u);
    us_hyperparam_interface->Set_beta_t(beta_t);
    us_hyperparam_interface->Set_GSVD_Hyperparameters(prior_num_sing_vals, prior_oversampling, prior_num_subspace_iter);
    HDSA::Ptr<HDSA::MD_u_Prior_Interface<ScalarT>> spatial_us_prior_interface = HDSA::makePtr<HDSA::MD_Numeric_Laplacian_u_Prior_Interface<ScalarT>>(S, M, data_interface_hyperparam, us_hyperparam_interface, random_number_generator);
    ScalarT T = solver_->final_time;
    int n_t = solver_->settings->sublist("Solver").get<int>("number of steps", 0) + 1;
    int n_y = data_interface_hyperparam->get_u_opt()->dimension() / n_t;
    HDSA::Ptr<HDSA::MD_Transient_Prior_Covariance<ScalarT>> transient_prior_cov = HDSA::makePtr<HDSA::MD_Transient_Prior_Covariance<ScalarT>>(data_interface_hyperparam, us_hyperparam_interface, T, n_t, n_y);
    us_prior_interface = HDSA::makePtr<HDSA::MD_Transient_Elliptic_u_Prior_Interface<ScalarT>>(spatial_us_prior_interface, transient_prior_cov);
  }
  else
  {
    us_hyperparam_interface = HDSA::makePtr<HDSA::MD_u_Hyperparameter_Interface<ScalarT>>(is_transient);
    us_hyperparam_interface->Set_alpha_d(alpha_d);
    us_hyperparam_interface->Set_alpha_u(alpha_u);
    us_hyperparam_interface->Set_beta_u(beta_u);
    us_hyperparam_interface->Set_GSVD_Hyperparameters(prior_num_sing_vals, prior_oversampling, prior_num_subspace_iter);
    us_prior_interface = HDSA::makePtr<HDSA::MD_Numeric_Laplacian_u_Prior_Interface<ScalarT>>(S, M, data_interface_hyperparam, us_hyperparam_interface, random_number_generator);
  }

  HDSA::Ptr<HDSA::Dense_Matrix<ScalarT>> K = HDSA::makePtr<HDSA::Dense_Matrix<ScalarT>>(ens_size, ens_size);
  for (int i = 0; i < ens_size; i++)
  {
    std::vector<ScalarT> pt_i = sampler->getMyPoint(i);
    for (int j = 0; j < ens_size; j++)
    {
      std::vector<ScalarT> pt_j = sampler->getMyPoint(j);
      ScalarT dist = 0.0;
      for (int k = 0; k < 3; k++)
      {
        dist += std::pow(pt_i[k] - pt_j[k], 2.0);
      }
      ScalarT val = std::exp(-0.5 * dist);
      K->Replace_Element(i, j, val);
    }
  }
  HDSA::Ptr<HDSA::MD_u_Prior_Interface<ScalarT>> u_prior_interface = HDSA::makePtr<HDSA::MD_OUU_u_Prior_Interface<ScalarT>>(us_prior_interface, K);

  std::string z_type = "vector";
  if (params_->getNumParams("discretized") > 0)
  {
    z_type = "spatial field";
  }
  HDSA::Ptr<HDSA::MD_z_Hyperparameter_Interface<ScalarT>> z_hyperparam_interface = HDSA::makePtr<MD_z_Hyperparameter_Interface_MrHyDE<ScalarT>>(data_interface_ens[0], random_number_generator, z_type, prior_num_state_solves);
  z_hyperparam_interface->Set_alpha_z(alpha_z);

  HDSA::Ptr<HDSA::MD_z_Prior_Interface<ScalarT>> z_prior_interface;
  if (z_type == "spatial field")
  {
    z_hyperparam_interface->Set_beta_z(beta_z);
    z_prior_interface = HDSA::makePtr<HDSA::MD_Numeric_Laplacian_z_Prior_Interface<ScalarT>>(S, M, data_interface_hyperparam, z_hyperparam_interface, u_prior_interface);
  }
  else if (z_type == "vector")
  {
    z_prior_interface = HDSA::makePtr<HDSA::MD_Vector_z_Prior_Interface<ScalarT>>(alpha_z);
  }

  output_writer->Write_Hyperparameters(us_hyperparam_interface, z_hyperparam_interface);

  if ((num_prior_samples > 0) & execute_prior_discrepancy_sampling)
  {
    HDSA::Ptr<HDSA::MD_Prior_Sampling<ScalarT>> prior_sampling = HDSA::makePtr<HDSA::MD_Prior_Sampling<ScalarT>>(data_interface, u_prior_interface, z_prior_interface);
    HDSA::Ptr<HDSA::MultiVector<ScalarT>> spatial_coords = data_interface_ens[0]->Read_Spatial_Node_Data();
    prior_sampling->Generate_Prior_Discrepancy_Sample_Data(num_prior_samples, us_hyperparam_interface, z_hyperparam_interface, spatial_coords);

    HDSA::Ptr<HDSA::MultiVector<ScalarT>> prior_delta_z_opt = prior_sampling->Get_prior_delta_z_opt();
    std::vector<HDSA::Ptr<HDSA::Vector<ScalarT>>> prior_z_pert = prior_sampling->Get_prior_z_pert();
    std::vector<HDSA::Ptr<HDSA::MultiVector<ScalarT>>> prior_delta_z_pert = prior_sampling->Get_prior_delta_z_pert();
    output_writer->Write_Prior_Discrepancy_Samples(prior_delta_z_opt, prior_z_pert, prior_delta_z_pert);
  }

  HDSA::Ptr<HDSA::MD_Posterior_Sampling<ScalarT>> post_sampling = HDSA::makePtr<HDSA::MD_Posterior_Sampling<ScalarT>>(data_interface, u_prior_interface, z_prior_interface);
  if (execute_posterior_discrepancy_sampling || execute_optimal_solution_update)
  {
    post_sampling->Compute_Posterior_Data(us_hyperparam_interface->Get_alpha_d(), num_posterior_samples);
  }

  if ((num_posterior_samples > 0) & execute_posterior_discrepancy_sampling)
  {
    std::vector<HDSA::Ptr<HDSA::Vector<ScalarT>>> z_in;
    int N = data_interface->get_Z()->Number_of_Vectors();
    z_in.resize(N);
    for (int k = 0; k < N; k++)
    {
      z_in[k] = (*data_interface->get_Z())[k];
    }
    std::vector<HDSA::Ptr<HDSA::MD_Posterior_Vectors<ScalarT>>> post_delta = post_sampling->Posterior_Discrepancy_Samples(z_in);
    output_writer->Write_Posterior_Discrepancy_Samples(post_delta);
  }

  if (execute_optimal_solution_update)
  {
    HDSA::Ptr<HDSA::MD_Hessian_Analysis<ScalarT>> hessian_analysis = HDSA::makePtr<HDSA::MD_Hessian_Analysis<ScalarT>>(opt_prob_interface, z_prior_interface);
    if (hessian_num_eig_vals > 0)
    {
      hessian_analysis->Compute_Hessian_GEVP(*data_interface->get_z_opt(), hessian_num_eig_vals, hessian_oversampling, false);
      HDSA::Ptr<HDSA::Dense_Matrix<ScalarT>> evals = hessian_analysis->Get_Evals();
      output_writer->Write_Hessian_Eigenvalues(evals);
    }

    HDSA::Ptr<HDSA::MD_Update<ScalarT>> update = HDSA::makePtr<HDSA::MD_Update<ScalarT>>(data_interface, u_prior_interface, z_prior_interface, opt_prob_interface, post_sampling, hessian_analysis);
    if (num_posterior_samples > 0)
    {
      HDSA::Ptr<HDSA::MD_Posterior_Vectors<ScalarT>> posterior_update_samples = update->Posterior_Update_Samples();
      output_writer->Write_Optimal_Solution_Update(posterior_update_samples);
      std::cout << "z_update_mean norm = " << posterior_update_samples->mean->norm() << std::endl;
    }
    else
    {
      HDSA::Ptr<HDSA::Vector<ScalarT>> z_update_mean = update->Posterior_Update_Mean();
      output_writer->Write_Optimal_Solution_Update(z_update_mean);
      std::cout << "z_update_mean norm = " << z_update_mean->norm() << std::endl;
    }
  }
}

void AnalysisManager::readExoForwardSolve()
{
  Teuchos::ParameterList read_exo_settings;

  if (settings_->sublist("Analysis").sublist("readExo+forward").isSublist("DataLoadParameters"))
    read_exo_settings = settings_->sublist("Analysis").sublist("readExo+forward").sublist("DataLoadParameters");
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error: MrHyDE could not find the readExo+forward sublist in the input file!  Abort!");

  std::string exo_file = read_exo_settings.get<std::string>("ExoFile", "error");
  std::string txt_file = read_exo_settings.get<std::string>("TxtFile", "error");

  if (exo_file != "error")
  {
    HDSA::Ptr<HDSA::Random_Number_Generator<ScalarT>> random_number_generator = HDSA::makePtr<HDSA::Random_Number_Generator<ScalarT>>();
    HDSA::Ptr<MD_Data_Interface_MrHyDE<ScalarT>> data_interface = HDSA::makePtr<MD_Data_Interface_MrHyDE<ScalarT>>(comm_, solver_, params_, random_number_generator, read_exo_settings);
    Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>> tpetra_vec = data_interface->Read_Exodus_Data(exo_file, false);
    params_->updateParams(tpetra_vec);
  }
  else if (txt_file != "error")
  {
    ScalarT val = 0.0;
    int dim = params_->getCurrentVector().dimension();
    // read in data
    std::ifstream in(txt_file);
    std::vector<ScalarT> vec = std::vector<ScalarT>(dim, 0.0);
    // read the elements in the file into a vector
    if (in)
    {
      for (int i = 0; i < dim; i++)
      {
        in >> val;
        vec[i] = val;
      }
    }
    else
    {
      std::cout << "Error loading the data from " << txt_file << std::endl;
    }

    params_->updateParams(vec, "active");
  }

  if (read_exo_settings.get("Sample Set File", "error") != "error")
  {
    int nsamp = read_exo_settings.get("Number of Samples", 100);
    int dim = params_->getNumParams("stochastic");
    ROL::Ptr<ROL::BatchManager<ScalarT>> bman = ROL::makePtr<ROL::MrHyDETeuchosBatchManager<ScalarT, int>>(comm_);
    std::string sample_pt_file = read_exo_settings.get("Sample Set File", "error");
    std::string sample_wt_file = read_exo_settings.get("Sample Weight File", "error");
    ROL::Ptr<ROL::SampleGenerator<ScalarT>> sampler = ROL::makePtr<ROL::Sample_Set_Reader<ScalarT>>(nsamp, dim, bman, sample_pt_file, sample_wt_file);

    for (int i = 0; i < sampler->numMySamples(); i++)
    {
      std::vector<ScalarT> pt_i = sampler->getMyPoint(i);
      params_->updateParams(pt_i, "stochastic");
      std::string outfile = "output_sample_" + std::to_string(i) + ".exo";
      postproc_->setNewExodusFile(outfile);
      this->forwardSolve();
    }
  }
  else
  {
    this->forwardSolve();
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

void AnalysisManager::writeSolutionToText(string &filename, vector<vector<vector_RCP>> &soln,
                                          const bool &only_write_final)
{
  typedef typename SolverNode::device_type LA_device;
  // vector<vector<vector_RCP> > soln = postproc_->soln[0]->extractAllData();
  int index = 0; // forget what this is for
  size_type numVecs = soln[index].size();
  auto v0_view = soln[index][0]->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  size_type numEnt = v0_view.extent(0);
  View_Sc2 all_data("data for writing", numEnt, numVecs);
  for (size_type v = 0; v < numVecs; ++v)
  {
    auto vec_view = soln[index][v]->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
    for (size_type i = 0; i < numEnt; ++i)
    {
      all_data(i, v) = vec_view(i, 0);
    }
  }
  std::ofstream solnOUT(filename.c_str());
  solnOUT.precision(12);
  for (size_type i = 0; i < numEnt; ++i)
  {
    if (only_write_final)
    {
      solnOUT << all_data(i, numVecs - 1) << "  ";
    }
    else
    {
      for (size_type v = 0; v < numVecs; ++v)
      {
        solnOUT << all_data(i, v) << "  ";
      }
    }
    solnOUT << endl;
  }
  solnOUT.close();
}
