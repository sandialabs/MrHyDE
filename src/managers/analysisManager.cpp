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
#include "MrHyDE_OptVector.hpp"
#include "ROL_LineSearchStep.hpp"
#include "ROL_Algorithm.hpp"
#include "ROL_Bounds.hpp"
#include "ROL_TrustRegionStep.hpp"
#include "ROL_Solver.hpp"

#if defined(MrHyDE_ENABLE_HDSA)
#include "../../../hdsalib/src/source_file.hpp"
#endif

using namespace MrHyDE;

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

AnalysisManager::AnalysisManager(const Teuchos::RCP<MpiComm> & comm,
                                 Teuchos::RCP<Teuchos::ParameterList> & settings,
                                 Teuchos::RCP<SolverManager<SolverNode> > & solver,
                                 Teuchos::RCP<PostprocessManager<SolverNode> > & postproc,
                                 Teuchos::RCP<ParameterManager<SolverNode> > & params) :
comm_(comm), settings_(settings), solver_(solver),
postproc_(postproc), params_(params) {
  
  RCP<Teuchos::Time> constructortime = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AnalysisManager - constructor");
  Teuchos::TimeMonitor constructortimer(*constructortime);
  
  verbosity_ = settings_->get<int>("verbosity",0);
  debugger_ = Teuchos::rcp(new MrHyDE_Debugger(settings_->get<int>("debug level",0), comm));
  // No debug output on this constructor
}


// ========================================================================================
// ========================================================================================

void AnalysisManager::run() {
  
  debugger_->print("**** Starting AnalysisManager::run ...");
  
  std::string analysis_type = settings_->sublist("Analysis").get<string>("analysis type","forward");
  this->run(analysis_type);
  
  debugger_->print("**** Finished analysis::run");
  
}
  
// ========================================================================================
// ========================================================================================

void AnalysisManager::run(std::string & analysis_type) {

  if (analysis_type == "forward") {
    
    DFAD objfun = this->forwardSolve();
    
  }
  else if (analysis_type == "forward+adjoint") {
    
    DFAD objfun = this->forwardSolve();
    
    MrHyDE_OptVector sens = this->adjointSolve();
    
  }
  else if (analysis_type == "dry run") {
    cout << " **** MrHyDE has completed the dry run with verbosity: " << verbosity_ << endl;
  }
  else if (analysis_type == "UQ") {
    vector<Teuchos::Array<ScalarT> > response_values = this->UQSolve();
  }
  else if (analysis_type == "ROL") {
    this->ROLSolve();
  }
  else if (analysis_type == "ROL2") {
    this->ROL2Solve();
  }
#if defined(MrHyDE_ENABLE_HDSA)
    else if (analysis_type == "HDSA") {
    this->HDSASolve();
  }
#endif
    else if (analysis_type == "DCI") {
    this->DCISolve();
  }
  else if (analysis_type == "restart") {
    this->restartSolve();
  }
  else { // don't solver_ anything, but produce visualization
    std::cout << "Unknown analysis option: " << analysis_type << std::endl;
    std::cout << "Valid and tested options: dry run, forward, forward+adjoint, UQ, ROL, ROL2, DCI" << std::endl;
  }
  
}

// ========================================================================================
// ========================================================================================

DFAD AnalysisManager::forwardSolve() {
  
  DFAD objfun = 0.0;
  solver_->forwardModel(objfun);
  postproc_->report();
  return objfun;
}


// ========================================================================================
// ========================================================================================

MrHyDE_OptVector AnalysisManager::adjointSolve() {
  
  MrHyDE_OptVector xtmp = params_->getCurrentVector();
  auto grad = xtmp.clone();
  MrHyDE_OptVector sens =
  Teuchos::dyn_cast<MrHyDE_OptVector >(const_cast<ROL::Vector<ScalarT> &>(*grad));
  sens.zero();
  solver_->adjointModel(sens);
  return sens;
  
}

// ========================================================================================
// ========================================================================================

void AnalysisManager::recoverSolution(vector_RCP & solution, string & data_type,
                                      string & plist_filename, string & filename) {
  
  string extension = filename.substr(filename.size()-4,filename.size()-1);
  filename.erase(filename.size()-4,4);
  
  cout << extension << "  " << filename << endl;
  if (data_type == "text") {
    std::stringstream sfile;
    sfile << filename << "." << comm_->getRank() << extension;
    std::ifstream fnmast(sfile.str());
    if (!fnmast.good()) {
      TEUCHOS_TEST_FOR_EXCEPTION(!fnmast.good(),std::runtime_error,"Error: could not find the data file: " + sfile.str());
    }
    
    std::vector<std::vector<ScalarT> > values;
    std::ifstream fin(sfile.str());
    
    for (std::string line; std::getline(fin, line); ) {
      std::replace(line.begin(), line.end(), ',', ' ');
      std::istringstream in(line);
      values.push_back(std::vector<ScalarT>(std::istream_iterator<ScalarT>(in),
                                            std::istream_iterator<ScalarT>()));
    }
    
    typedef typename SolverNode::device_type              LA_device;
    auto sol_view = solution->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
    for (size_type i=0; i<values.size(); ++i) {
      sol_view(i,0) = values[i][0];
    }
  }
  else if (data_type == "exodus") {
    
  }
  else if (data_type == "hdf5") {
    
  }
  else if (data_type == "binary") {
    
  }
  else {
    std::cout << "Unknown file type: " << data_type << std::endl;
  }
  
}

// ========================================================================================
// ========================================================================================

void AnalysisManager::updateRotationData(const int & newrandseed) {
  
  // Determine how many seeds there are
  size_t localnumSeeds = 0;
  size_t numSeeds = 0;
  for (size_t block=0; block<solver_->assembler->groups.size(); ++block) {
    for (size_t grp=0; grp<solver_->assembler->groups[block].size(); ++grp) {
      for (size_t e=0; e<solver_->assembler->groups[block][grp]->numElem; ++e) {
        if (solver_->assembler->groups[block][grp]->data_seed[e] > localnumSeeds) {
          localnumSeeds = solver_->assembler->groups[block][grp]->data_seed[e];
        }
      }
    }
  }
  //comm_->MaxAll(&localnumSeeds, &numSeeds, 1);
  Teuchos::reduceAll<int,size_t>(*comm_,Teuchos::REDUCE_MAX,1,&localnumSeeds,&numSeeds);
  numSeeds += 1; //To properly allocate and iterate
  
  // Create a random number generator
  std::default_random_engine generator(newrandseed);
  
  ////////////////////////////////////////////////////////////////////////////////
  // Set seed data
  ////////////////////////////////////////////////////////////////////////////////
  
  int numdata = 9;
  
  //cout << "solver_r numSeeds = " << numSeeds << endl;
  
  std::normal_distribution<ScalarT> ndistribution(0.0,1.0);
  Kokkos::View<ScalarT**,HostDevice> rotation_data("cell_data",numSeeds,numdata);
  for (size_t k=0; k<numSeeds; k++) {
    ScalarT x = ndistribution(generator);
    ScalarT y = ndistribution(generator);
    ScalarT z = ndistribution(generator);
    ScalarT w = ndistribution(generator);
    
    ScalarT r = sqrt(x*x + y*y + z*z + w*w);
    x *= 1.0/r;
    y *= 1.0/r;
    z *= 1.0/r;
    w *= 1.0/r;
    
    rotation_data(k,0) = w*w + x*x - y*y - z*z;
    rotation_data(k,1) = 2.0*(x*y - w*z);
    rotation_data(k,2) = 2.0*(x*z + w*y);
    
    rotation_data(k,3) = 2.0*(x*y + w*z);
    rotation_data(k,4) = w*w - x*x + y*y - z*z;
    rotation_data(k,5) = 2.0*(y*z - w*x);
    
    rotation_data(k,6) = 2.0*(x*z - w*y);
    rotation_data(k,7) = 2.0*(y*z + w*x);
    rotation_data(k,8) = w*w - x*x - y*y + z*z;
    
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  // Set cell data
  ////////////////////////////////////////////////////////////////////////////////
  
  for (size_t block=0; block<solver_->assembler->groups.size(); ++block) {
    for (size_t grp=0; grp<solver_->assembler->groups[block].size(); ++grp) {
      int numElem = solver_->assembler->groups[block][grp]->numElem;
      for (int c=0; c<numElem; c++) {
        int cnode = solver_->assembler->groups[block][grp]->data_seed[c];
        for (int i=0; i<9; i++) {
          solver_->assembler->groups[block][grp]->data(c,i) = rotation_data(cnode,i);
        }
      }
    }
  }
  for (size_t block=0; block<solver_->assembler->boundary_groups.size(); ++block) {
    for (size_t grp=0; grp<solver_->assembler->boundary_groups[block].size(); ++grp) {
      int numElem = solver_->assembler->boundary_groups[block][grp]->numElem;
      for (int e=0; e<numElem; ++e) {
        int cnode = solver_->assembler->boundary_groups[block][grp]->data_seed[e];
        for (int i=0; i<9; i++) {
          solver_->assembler->boundary_groups[block][grp]->data(e,i) = rotation_data(cnode,i);
        }
      }
    }
  }
  solver_->multiscale_manager->updateMeshData(rotation_data);
}

// ========================================================================================
// ========================================================================================

vector<Teuchos::Array<ScalarT> > AnalysisManager::UQSolve() {
  
  vector<Teuchos::Array<ScalarT> > response_values;
  
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
  int numsamples = uqsettings_.get<int>("samples",100);
  int maxsamples = uqsettings_.get<int>("max samples",numsamples); // needed for generating subsets of samples
  int seed = uqsettings_.get<int>("seed",1234);
  bool regenerate_rotations = uqsettings_.get<bool>("regenerate grain rotations",false);
  bool regenerate_grains = uqsettings_.get<bool>("regenerate grains",false);
  bool write_sol_text = uqsettings_.get<bool>("write solutions to text file",false);
  bool write_samples = uqsettings_.get<bool>("write samples",false);
  bool only_write_final = uqsettings_.get<bool>("only write final time",false);
  bool compute_adjoint = uqsettings_.get<bool>("compute adjoint",false);
  bool write_adjoint_text = uqsettings_.get<bool>("write adjoint to text file",false);
  int output_freq = uqsettings_.get<int>("output frequency",1);
  
  // Generate the samples (wastes memory if requires large number of samples in high-dim space)
  Kokkos::View<ScalarT**,HostDevice> samplepts = uq.generateSamples(maxsamples, seed);
  // Adjust the number of samples (if necessary)
  numsamples = std::min(numsamples, static_cast<int>(samplepts.extent(0)));
  Kokkos::View<int*,HostDevice> sampleints = uq.generateIntegerSamples(maxsamples, seed);
  
  // Write the samples to file (if requested)
  if (write_samples) {
    string sample_file = uqsettings_.get<string>("samples output file","sample_inputs.dat");
    std::ofstream sampOUT(sample_file.c_str());
    for (size_type i=0; i<samplepts.extent(0); ++i) {
      for (size_type v=0; v<samplepts.extent(1); ++v) {
        sampOUT << samplepts(i,v) << "  ";
      }
      sampOUT << endl;
    }
    sampOUT.close();
  }
  
  if (write_sol_text) {
    postproc_->save_solution = true;
  }
  
  
  if (comm_->getRank() == 0) {
    cout << "Running Monte Carlo sampling ..." << endl;
  }
  for (int j=0; j<numsamples; j++) {
    
    ////////////////////////////////////////////////////////
    // Generate a new realization
    // Update stochastic parameters
    if (numstochparams_ > 0) {
      vector<ScalarT> currparams_;
      cout << "New params: ";
      for (int i=0; i<numstochparams_; i++) {
        currparams_.push_back(samplepts(j,i));
        cout << samplepts(j,i) << " ";
      }
      cout << endl;
      params_->updateParams(currparams_,2);
      
    }
    // Update random microstructure
    if (regenerate_grains) {
      auto seeds = solver_->mesh->generateNewMicrostructure(sampleints(j));
      solver_->assembler->importNewMicrostructure(sampleints(j), seeds);
    }
    else if (regenerate_rotations) {
      this->updateRotationData(sampleints(j));
    }
    
    // Update the append string in postproc_essor for labelling
    std::stringstream ss;
    ss << "_" << j;
    postproc_->append = ss.str();
    
    ////////////////////////////////////////////////////////
    // Evaluate the new realization
    
    DFAD objfun = this->forwardSolve();
    //postproc_->report();
    Teuchos::Array<ScalarT> newresp = postproc_->collectResponses();
    
    response_values.push_back(newresp);
    
    ////////////////////////////////////////////////////////
    
    // The following output is for a specific use case.
    // Should not be used in general (unless you want TB of data)
    if (write_sol_text) {
      std::stringstream sfile;
      sfile << "solution." << j << "." << comm_->getRank() << ".dat";
      string filename = sfile.str();
      vector<vector<vector_RCP> > soln = postproc_->soln[0]->extractAllData();
      this->writeSolutionToText(filename, soln, only_write_final);
    }
    if (compute_adjoint) {
      
      postproc_->save_adjoint_solution = true;
      MrHyDE_OptVector sens = this->adjointSolve();
      
      if (write_adjoint_text) {
        std::stringstream sfile;
        sfile << "adjoint." << j << "." << comm_->getRank() << ".dat";
        string filename = sfile.str();
        vector<vector<vector_RCP> > soln = postproc_->adj_soln[0]->extractAllData();
        this->writeSolutionToText(filename, soln);
      }
    }
    
    // Update the user on the progress
    if (comm_->getRank() == 0 && j%output_freq == 0) {
      cout << "Finished evaluating sample number: " << j+1 << " out of " << numsamples << endl;
    }
  } // end sample loop
  
  
  if (comm_->getRank() == 0) {
    
    string sname = "sample_output.dat";
    std::ofstream respOUT(sname.c_str());
    respOUT.precision(12);
    for (size_t r=0; r<response_values.size(); r++) {
      for (long int s=0; s<response_values[r].size(); s++) {
        respOUT << response_values[r][s] << "  ";
      }
      respOUT << endl;
    }
    respOUT.close();
    
  }
  
  if (settings_->sublist("postprocess").get("write solution",true)) {
    //postproc_->writeSolution(avgsoln, "output_avg");
  }
  // Compute the statistics (mean, variance, probability levels, etc.)
  //uq.computeStatistics(response_values);
  
  return response_values;
}

// ========================================================================================
// ========================================================================================

void AnalysisManager::ROLSolve() {
  
  typedef ScalarT RealT;
  
  Teuchos::RCP< ROL::Objective_MILO<RealT> > obj;
  Teuchos::ParameterList ROLsettings;
  
  if (settings_->sublist("Analysis").isSublist("ROL"))
    ROLsettings= settings_->sublist("Analysis").sublist("ROL");
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: MrHyDE could not find the ROL sublist in the input file!  Abort!");
  
  // New ROL input syntax
  bool use_linesearch = settings_->sublist("Analysis").get("Use Line Search",false);
  
  ROLsettings.sublist("General").sublist("Secant").set("Type","Limited-Memory BFGS");
  ROLsettings.sublist("Step").sublist("Descent Method").set("Type","Newton Krylov");
  ROLsettings.sublist("Step").sublist("Trust Region").set("Subproblem solve","Truncated CG");
  
  RealT gtol     = ROLsettings.sublist("Status Test").get("Gradient Tolerance",1e-6);
  RealT stol     = ROLsettings.sublist("Status Test").get("Step Tolerance",1.e-12);
  int maxit      = ROLsettings.sublist("Status Test").get("Iteration Limit",100);
  
  // Turn off visualization while optimizing
  bool postproc_plot = postproc_->write_solution;
  postproc_->write_solution = false;
  
  Teuchos::RCP<std::ostream> outStream;
  outStream = Teuchos::rcp(&std::cout, false);
  // Generate data and get objective
  obj = Teuchos::rcp( new ROL::Objective_MILO<RealT> (solver_, postproc_, params_));
  
  Teuchos::RCP< ROL::Step<RealT> > step;
  
  if (use_linesearch) {
    step = Teuchos::rcp( new ROL::LineSearchStep<RealT> (ROLsettings) );
  }
  else {
    step = Teuchos::rcp( new ROL::TrustRegionStep<RealT> (ROLsettings) );
  }
  
  Teuchos::RCP<ROL::StatusTest<RealT> > status = Teuchos::rcp( new ROL::StatusTest<RealT> (gtol, stol, maxit) );
  
  ROL::Algorithm<RealT> algo(step,status,false);
  
  MrHyDE_OptVector xtmp = params_->getCurrentVector();
  
  Teuchos::RCP<ROL::Vector<ScalarT>> x = xtmp.clone();
  x->set(xtmp);

  //ScalarT roltol = 1e-8;
  //*outStream << "\nTesting objective!!\n";
  //obj->value(*x, roltol);
  //*outStream << "\nObjective evaluation works!!\n";
  
  //bound contraint
  Teuchos::RCP<ROL::Bounds<RealT> > con;
  bool bound_vars = ROLsettings.sublist("General").get("Bound Optimization Variables",false);
  
  if (bound_vars) {
    
    //read in bounds for parameters...
    vector<Teuchos::RCP<vector<ScalarT> > > activeBnds = params_->getActiveParamBounds();
    vector<vector_RCP> discBnds = params_->getDiscretizedParamBounds();
    Teuchos::RCP<ROL::Vector<ScalarT> > lo = Teuchos::rcp( new MrHyDE_OptVector(discBnds[0], activeBnds[0], comm_->getRank()) );
    Teuchos::RCP<ROL::Vector<ScalarT> > up = Teuchos::rcp( new MrHyDE_OptVector(discBnds[1], activeBnds[1], comm_->getRank()) );
    
    con = Teuchos::rcp(new ROL::Bounds<RealT>(lo,up));
    
    //create bound constraint
  }
  
  //////////////////////////////////////////////////////
  // Verification tests
  //////////////////////////////////////////////////////
  
  // Recovering a data-generating solution
  if (ROLsettings.sublist("General").get("Generate data",false)) {
    //std::cout << "Generating data ... " << std::endl;
    DFAD objfun = 0.0;
    if (params_->isParameter("datagen")) {
      vector<ScalarT> pval = {1.0};
      params_->setParam(pval,"datagen");
    }
    postproc_->response_type = "none";
    postproc_->compute_objective = false;
    solver_->forwardModel(objfun);
    //std::cout << "Storing data ... " << std::endl;
    
    for (size_t set=0; set<postproc_->soln.size(); ++set) {
      vector<vector<ScalarT> > times = postproc_->soln[set]->extractAllTimes();
      vector<vector<Teuchos::RCP<LA_MultiVector> > > data = postproc_->soln[set]->extractAllData();
      
      for (size_t i=0; i<times.size(); i++) {
        for (size_t j=0; j<times[i].size(); j++) {
          postproc_->datagen_soln[set]->store(data[i][j], times[i][j], i);
        }
      }
    }
    
    //std::cout << "Finished storing data" << std::endl;
    if (params_->isParameter("datagen")) {
      vector<ScalarT> pval = {0.0};
      params_->setParam(pval,"datagen");
    }
    postproc_->response_type = "discrete";
    postproc_->compute_objective = true;
    //std::cout << "Finished generating data for inversion " << std::endl;
  }
  
  
  // Comparing a gradient/Hessian with finite difference approximation
  if (ROLsettings.sublist("General").get("Do grad+hessvec check",true)) {
    // Gradient and Hessian check
    // direction for gradient check
    
    Teuchos::RCP<ROL::Vector<ScalarT>> d = x->clone();
    
    if (ROLsettings.sublist("General").get("FD Check Use Ones Vector",false)) {
      d->setScalar(1.0);
    }
    else {
      if (ROLsettings.sublist("General").isParameter("FD Check Seed")) {
        int seed = ROLsettings.get("FD Check Seed",1);
        srand(seed);
      }
      else {
        srand(time(NULL)); //initialize random seed
      }
      d->randomize();
      if (ROLsettings.sublist("General").isParameter("FD Scale")) {
        ScalarT scale = ROLsettings.sublist("General").get<double>("FD Scale",1.0);
        d->scale(scale);
      }
    }
    
    // check gradient and Hessian-vector computation using finite differences
    obj->checkGradient(*x, *d, (comm_->getRank() == 0));
    
  }
  
  Teuchos::Time timer("Optimization Time",true);
  
  // Run algorithm.
  vector<std::string> output;
  if (bound_vars) {
    output = algo.run(*x, *obj, *con, (comm_->getRank() == 0 )); //only processor of rank 0 print outs
  }
  else {
    output = algo.run(*x, *obj, (comm_->getRank() == 0)); //only processor of rank 0 prints out
  }
  
  
  ScalarT optTime = timer.stop();
  
  if (ROLsettings.sublist("General").get("Write Final Parameters",false) ) {
    string outname = ROLsettings.get("Output File Name","ROL_out.txt");
    std::ofstream respOUT(outname);
    respOUT.precision(16);
    if (comm_->getRank() == 0 ) {
      
      for ( unsigned i = 0; i < output.size(); i++ ) {
        std::cout << output[i];
        respOUT << output[i];
      }
      x->print(std::cout);
    }
    Kokkos::fence();
    x->print(respOUT);
    
    if (comm_->getRank() == 0 ) {
      if (verbosity_ > 5) {
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
   DFAD val = 0.0;
   solver_->forwardModel(val);
   //postproc_->writeSolution(settings_->sublist("postproc_ess").get<string>("Output File","output"));
   }
   */
  
  if (postproc_plot) {
    postproc_->write_solution = true;
    string outfile = "output_after_optimization.exo";
    postproc_->setNewExodusFile(outfile);
    DFAD objfun = 0.0;
    solver_->forwardModel(objfun);
    if (ROLsettings.sublist("General").get("Disable source on final output",false) ) {
      vector<bool> newflags(1,false);
      solver_->physics->updateFlags(newflags);
      string outfile = "output_only_control.exo";
      postproc_->setNewExodusFile(outfile);
      solver_->forwardModel(objfun);
    }
  }
}

// ========================================================================================
// ========================================================================================

void AnalysisManager::ROL2Solve() {
  
  typedef ScalarT RealT;
  
  Teuchos::RCP< ROL::Objective_MILO<RealT> > obj;
  Teuchos::ParameterList ROLsettings;
  
  if (settings_->sublist("Analysis").isSublist("ROL2"))
    ROLsettings = settings_->sublist("Analysis").sublist("ROL2");
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: MrHyDE could not find the ROL2 sublist in the input file!  Abort!");
  
  // Turn off visualization while optimizing
  bool postproc_plot = postproc_->write_solution;
  postproc_->write_solution = false;
  
  // Output stream.
  ROL::Ptr<std::ostream> outStream;
  ROL::nullstream bhs; // outputs nothing
  if (comm_->getRank() == 0 ) {
    outStream = ROL::makePtrFromRef(std::cout);
  }
  else {
    outStream = ROL::makePtrFromRef(bhs);
  }
  
  // Generate data and get objective
  obj = Teuchos::rcp( new ROL::Objective_MILO<RealT> (solver_, postproc_, params_));
  
  MrHyDE_OptVector xtmp = params_->getCurrentVector();
  
  Teuchos::RCP<ROL::Vector<ScalarT>> x = xtmp.clone();
  x->set(xtmp);
  
  //bound constraint
  Teuchos::RCP<ROL::BoundConstraint<RealT> > con;
  bool bound_vars = ROLsettings.sublist("General").get("Bound Optimization Variables",false);
  
  if (bound_vars) {
    
    //read in bounds for parameters...
    vector<Teuchos::RCP<vector<ScalarT> > > activeBnds = params_->getActiveParamBounds();
    vector<vector_RCP> discBnds = params_->getDiscretizedParamBounds();
    Teuchos::RCP<ROL::Vector<ScalarT> > lo = Teuchos::rcp( new MrHyDE_OptVector(discBnds[0], activeBnds[0], comm_->getRank()) );
    Teuchos::RCP<ROL::Vector<ScalarT> > up = Teuchos::rcp( new MrHyDE_OptVector(discBnds[1], activeBnds[1], comm_->getRank()) );
    
    //create bound constraint
    con = Teuchos::rcp(new ROL::Bounds<RealT>(lo,up));
    
  }
  
  //////////////////////////////////////////////////////
  // Verification tests
  //////////////////////////////////////////////////////
  
  // Recovering a data-generating solution
  if (ROLsettings.sublist("General").get("Generate data",false)) {
    //std::cout << "Generating data ... " << std::endl;
    DFAD objfun = 0.0;
    if (params_->isParameter("datagen")) {
      vector<ScalarT> pval = {1.0};
      params_->setParam(pval,"datagen");
    }
    postproc_->response_type = "none";
    postproc_->compute_objective = false;
    solver_->forwardModel(objfun);
    //std::cout << "Storing data ... " << std::endl;
    
    for (size_t set=0; set<postproc_->soln.size(); ++set) {
      vector<vector<ScalarT> > times = postproc_->soln[set]->extractAllTimes();
      vector<vector<Teuchos::RCP<LA_MultiVector> > > data = postproc_->soln[set]->extractAllData();
      for (size_t i=0; i<times.size(); i++) {
        for (size_t j=0; j<times[i].size(); j++) {
          postproc_->datagen_soln[set]->store(data[i][j], times[i][j], i);
        }
      }
    }
    
    //std::cout << "Finished storing data" << std::endl;
    if (params_->isParameter("datagen")) {
      vector<ScalarT> pval = {0.0};
      params_->setParam(pval,"datagen");
    }
    postproc_->response_type = "discrete";
    postproc_->compute_objective = true;
    //std::cout << "Finished generating data for inversion " << std::endl;
  }
  
  // Comparing a gradient/Hessian with finite difference approximation
  if (ROLsettings.sublist("General").get("Do grad+hessvec check",true)) {
    // Gradient and Hessian check
    // direction for gradient check
    
    Teuchos::RCP<ROL::Vector<ScalarT>> d = x->clone();
    
    if (ROLsettings.sublist("General").get("FD Check Use Ones Vector",false)) {
      d->setScalar(1.0);
    }
    else {
      if (ROLsettings.sublist("General").isParameter("FD Check Seed")) {
        int seed = ROLsettings.get("FD Check Seed",1);
        srand(seed);
      }
      else {
        srand(time(NULL)); //initialize random seed
      }
      d->randomize();
      if (ROLsettings.sublist("General").isParameter("FD Scale")) {
        ScalarT scale = ROLsettings.sublist("General").get<double>("FD Scale",1.0);
        d->scale(scale);
      }
    }
    
    // check gradient and Hessian-vector computation using finite differences
    obj->checkGradient(*x, *d, (comm_->getRank() == 0));
    
  }
  
  Teuchos::Time timer("Optimization Time",true);
  
  // Construct ROL problem.
  ROL::Ptr<ROL::Problem<RealT>> rolProblem = ROL::makePtr<ROL::Problem<RealT>>(obj, x);
  //rolProblem->check(true, *outStream);
  if (bound_vars) {
    rolProblem->addBoundConstraint(con);
  }
  
  // Construct ROL solver.
  ROL::Solver<ScalarT> rolSolver(rolProblem, ROLsettings);
  
  // Run algorithm.
  rolSolver.solve(*outStream);
  
  //ScalarT optTime = timer.stop();
  
  /*
   if (settings_->sublist("postproc_ess").get("write Hessian",false)){
   obj->printHess(settings_->sublist("postproc_ess").get("Hessian output file","hess.dat"),x,comm_->getRank());
   }
   if (settings_->sublist("Analysis").get("write output",false)) {
   DFAD val = 0.0;
   solver_->forwardModel(val);
   //postproc_->writeSolution(settings_->sublist("postproc_ess").get<string>("Output File","output"));
   }
   */
  
  if (postproc_plot) {
    postproc_->write_solution = true;
    string outfile = "output_after_optimization.exo";
    postproc_->setNewExodusFile(outfile);
    DFAD objfun = 0.0;
    solver_->forwardModel(objfun);
    if (ROLsettings.sublist("General").get("Disable source on final output",false) ) {
      vector<bool> newflags(1,false);
      solver_->physics->updateFlags(newflags);
      string outfile = "output_only_control.exo";
      postproc_->setNewExodusFile(outfile);
      solver_->forwardModel(objfun);
    }
    
  }
}

// ========================================================================================
// ========================================================================================

#if defined(MrHyDE_ENABLE_HDSA)
void AnalysisManager::HDSASolve() {
  HDSA::Ptr<HDSA::Random_Number_Generator<ScalarT> > random_number_generator = HDSA::makePtr<HDSA::Random_Number_Generator<ScalarT> >();

  HDSA::Ptr<HDSA::MD_Data_Interface<ScalarT> > data_interface = HDSA::makePtr<MD_Data_Interface_MrHyDE<ScalarT> >(solver_,random_number_generator);
  HDSA::Ptr<HDSA::MD_Opt_Prob_Interface<ScalarT> > opt_prob_interface = HDSA::makePtr<MD_Opt_Prob_Interface_MrHyDE<ScalarT> >(solver_, postproc_, params_,random_number_generator);
  HDSA::Ptr<HDSA::Vector<ScalarT> > z_tmp1 = data_interface->get_z_opt()->clone();
  HDSA::Ptr<HDSA::Vector<ScalarT> > u_tmp1 = data_interface->get_u_opt()->clone();
  //u_tmp1->setScalar(1.0);
  u_tmp1->set(*data_interface->get_u_opt());
  opt_prob_interface->Apply_Solution_Operator_z_Jacobian_Transpose(*z_tmp1,*u_tmp1,*data_interface->get_z_opt());

  // HDSA::Ptr<HDSA::Vector<ScalarT> > z_tmp1 = data_interface->get_z_opt()->clone();
  // HDSA::Ptr<HDSA::Vector<ScalarT> > z_tmp2 = data_interface->get_z_opt()->clone();
  // z_tmp1->setScalar(1.0);
  // opt_prob_interface->Apply_RS_Hessian(*z_tmp2,*z_tmp1,*data_interface->get_z_opt());

  // HDSA::Ptr<HDSA::Vector<ScalarT> > u_tmp1 = data_interface->get_u_opt()->clone();
  // u_tmp1->setScalar(1.0);
  // opt_prob_interface->Apply_Solution_Operator_z_Jacobian_Transpose(*z_tmp1,*u_tmp1,*data_interface->get_z_opt());

  vector<string> blockNames = solver_->mesh->getBlockNames();
  HDSA::Ptr< HDSA_Prior_FE_Op_MrHyDE_Interface<ScalarT>> prior_fe_op = HDSA::makePtr<HDSA_Prior_FE_Op_MrHyDE_Interface<ScalarT>>(comm_,settings_,blockNames) ;
  
  //bvbw move alpha_u and beta_u to yaml
  ScalarT alpha_u = 1.0/4.0; // 4.0; 
  ScalarT beta_u = 1.0E-2; //2.0E-2; 
  HDSA::Ptr<HDSA::Vector<ScalarT> > uvec = HDSA::makePtr<HDSA::Vector_MrHyDE_State<ScalarT> >(solver_,random_number_generator,true);
  HDSA::Ptr<HDSA::MD_Elliptic_u_Prior_Interface<ScalarT> > u_prior_interface = HDSA::makePtr<MD_Elliptic_u_Prior_Interface_MrHyDE<ScalarT> >(alpha_u,beta_u,prior_fe_op,uvec,random_number_generator);

  ScalarT alpha_z = 1.0/(600.0*600.0); //1.0E-10; 
  ScalarT beta_z =  1.0E-3; // 3.0E-2;
  HDSA::Ptr<HDSA::MD_Elliptic_z_Prior_Interface<ScalarT> > z_prior_interface = HDSA::makePtr<MD_Elliptic_z_Prior_Interface_MrHyDE<ScalarT> >(alpha_z,beta_z,prior_fe_op);
  
  HDSA::Ptr<HDSA::MD_Posterior_Sampling<ScalarT> > post_sampling = HDSA::makePtr<HDSA::MD_Posterior_Sampling<ScalarT> >(data_interface,u_prior_interface,z_prior_interface);
  ScalarT alpha_d = 1.e-5; //1.0E-4; 
  int num_post_samples = 0;
  post_sampling->Compute_Posterior_Data(alpha_d,num_post_samples);

  HDSA::Ptr<HDSA::MD_Hessian_Analysis<ScalarT> > hessian_analysis = HDSA::makePtr<HDSA::MD_Hessian_Analysis<ScalarT> >(opt_prob_interface,z_prior_interface);
  
  int num_evals = 10; //4;
  int oversampling = 10 ; // 2; 
  hessian_analysis->Compute_Hessian_GEVP(*data_interface->get_z_opt(),num_evals,oversampling);

  HDSA::Ptr<HDSA::MD_Update<ScalarT> > update = HDSA::makePtr<HDSA::MD_Update<ScalarT> >(data_interface,u_prior_interface,z_prior_interface,opt_prob_interface,post_sampling,hessian_analysis);
 
  HDSA::Ptr<HDSA::Vector<ScalarT> > mean_update = update->Posterior_Update_Mean();
  std::cout << "norm update " << mean_update->norm() << std::endl;
}
#endif

// ========================================================================================
// ========================================================================================

void AnalysisManager::DCISolve() {
  // Evaluate model or a surrogate at these samples
  vector<Teuchos::Array<ScalarT> > response_values = this->UQSolve();
  
  // Get the UQ sublist
  Teuchos::ParameterList uqsettings_ = settings_->sublist("Analysis").sublist("UQ");
  
  // Get the DCI sublist
  Teuchos::ParameterList dcisettings_ = settings_->sublist("Analysis").sublist("DCI");
  
  // Need to evaluate the observed density at samples using either an analytic density or a KDE built from data
  string obs_type = dcisettings_.get<string>("observed type","Gaussian"); // other options: uniform or data
  
  View_Sc1 obsdens("observed density values",response_values.size());
  
  if (obs_type == "Gaussian") {
    
  }
  else if (obs_type == "uniform") {
    
  }
  else if (obs_type == "data") {
    // load in data
    
    // build KDE
    
    
  }
}

// ========================================================================================
// ========================================================================================

void AnalysisManager::restartSolve() {
  
  Teuchos::ParameterList rstsettings_ = settings_->sublist("Analysis").sublist("Restart");
  string state_file = rstsettings_.get<string>("state file name","none");
  string param_file = rstsettings_.get<string>("parameter file name","none");
  string adjoint_file = rstsettings_.get<string>("adjoint file name","none");
  string disc_param_file = rstsettings_.get<string>("discretized parameter file name","none");
  string scalar_param_file = rstsettings_.get<string>("scalar parameter file name","none");
  string mode = rstsettings_.get<string>("mode","forward");
  string data_type = rstsettings_.get<string>("file type","text");
  double start_time = rstsettings_.get<double>("start time",0.0);
  
  solver_->initial_time = start_time;
  solver_->current_time = start_time;
  
  ///////////////////////////////////////////////////////////
  // Recover the state
  ///////////////////////////////////////////////////////////
  
  vector<vector_RCP> forward_solution, adjoint_solution;
  vector_RCP disc_params_;
  vector<ScalarT> scalar_params_;
  if (state_file != "none" ) {
    forward_solution = solver_->getRestartSolution();
    this->recoverSolution(forward_solution[0], data_type, param_file, state_file);
  }
  
  if (adjoint_file != "none" ) {
    adjoint_solution = solver_->getRestartAdjointSolution();
    this->recoverSolution(adjoint_solution[0], data_type, param_file, adjoint_file);
  }
  
  if (disc_param_file != "none" ) {
    disc_params_ = params_->getDiscretizedParams();
    this->recoverSolution(disc_params_, data_type, param_file, disc_param_file);
  }
  if (scalar_param_file != "none" ) {
    
  }
  
  solver_->use_restart = true;
  
  ///////////////////////////////////////////////////////////
  // Run the requested mode
  ///////////////////////////////////////////////////////////
  
  if (mode == "forward") {
    
  }
  else if (mode == "ROL") {
  }
  else if (mode == "ROL2") {
  }
  else { // don't solver_ anything, but produce visualization
    std::cout << "Unknown restart mode: " << mode << std::endl;
  }
}

// ========================================================================================
// ========================================================================================

void AnalysisManager::writeSolutionToText(string & filename, vector<vector<vector_RCP> > & soln,
                                          const bool & only_write_final) {
  typedef typename SolverNode::device_type  LA_device;
  //vector<vector<vector_RCP> > soln = postproc_->soln[0]->extractAllData();
  int index = 0; // forget what this is for
  size_type numVecs = soln[index].size();
  auto v0_view = soln[index][0]->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  size_type numEnt = v0_view.extent(0);
  View_Sc2 all_data("data for writing",numEnt,numVecs);
  for (size_type v=0; v<numVecs; ++v) {
    auto vec_view = soln[index][v]->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
    for (size_type i=0; i<numEnt; ++i) {
      all_data(i,v) = vec_view(i,0);
    }
  }
  std::ofstream solnOUT(filename.c_str());
  solnOUT.precision(12);
  for (size_type i=0; i<numEnt; ++i) {
    if (only_write_final) {
      solnOUT << all_data(i,numVecs-1) << "  ";
    }
    else {
      for (size_type v=0; v<numVecs; ++v) {
        solnOUT << all_data(i,v) << "  ";
      }
    }
    solnOUT << endl;
  }
  solnOUT.close();
}
