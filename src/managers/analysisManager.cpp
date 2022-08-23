/***********************************************************************
 This is a framework for solving Multi-resolution Hybridized
 Differential Equations (MrHyDE), an optimized version of
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "analysisManager.hpp"
#include "physicsInterface.hpp"
#include "discretizationInterface.hpp"
#include "uqManager.hpp"
#include "obj_milorol.hpp"
//#include "ROL_StdVector.hpp"
#include "MrHyDE_OptVector.hpp"
//#include "obj_milorol_simopt.hpp"
#include "ROL_LineSearchStep.hpp"
#include "ROL_Algorithm.hpp"
#include "ROL_Bounds.hpp"
#include "ROL_TrustRegionStep.hpp"

using namespace MrHyDE;

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

AnalysisManager::AnalysisManager(const Teuchos::RCP<MpiComm> & Comm_,
                                 Teuchos::RCP<Teuchos::ParameterList> & settings_,
                                 Teuchos::RCP<SolverManager<SolverNode> > & solver_,
                                 Teuchos::RCP<PostprocessManager<SolverNode> > & postproc_,
                                 Teuchos::RCP<ParameterManager<SolverNode> > & params_) :
Comm(Comm_), settings(settings_), solve(solver_),
postproc(postproc_), params(params_) {
  
  RCP<Teuchos::Time> constructortime = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AnalysisManager - constructor");
  Teuchos::TimeMonitor constructortimer(*constructortime);
  
  verbosity = settings->get<int>("verbosity",0);
  debug_level = settings->get<int>("debug level",0);
  // No debug output on this constructor
}


// ========================================================================================
/* given the parameters, solve the forward  problem */
// ========================================================================================

void AnalysisManager::run() {
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting AnalysisManager::run ..." << endl;
    }
  }
  
  std::string analysis_type = settings->sublist("Analysis").get<string>("analysis type","forward");
  DFAD objfun = 0.0;
  
  if (analysis_type == "forward") {
    
    solve->forwardModel(objfun);
    postproc->report();
    
  }
  else if (analysis_type == "forward+adjoint") {
    solve->forwardModel(objfun);
    postproc->report();
    
    solve->adjointModel(*gradient);
    
  }
  else if (analysis_type == "dry run") {
    cout << " **** MrHyDE has completed the dry run with verbosity: " << verbosity << endl;
  }
  else if (analysis_type == "dakota") {
    // placeholder for embedded dakota analysis
  }
  else if (analysis_type == "NLCG") {
    // placeholder for an "in-house" nonlinear CG optimization algorithm
  }
  else if (analysis_type == "Sampling") {
    
    // UQ is forward propagation of uncertainty computed using sampling
    // We may sample either MILO (Monte Carlo), or
    // We may sample a surrogate model ... we first need to build this surrogate model (perhaps adaptively)
    // Note: PCE models provide analytical estimates of the mean and variance, but we will make no use of this
    
    // Build the uq manager
    Teuchos::ParameterList sampsettings = settings->sublist("Analysis").sublist("Sampling");
    
    // Read in the samples
    int ptsdim = sampsettings.get<int>("dimension",2);
    Data sdata("Sample Points", ptsdim, sampsettings.get("source","samples.dat"));
    Kokkos::View<ScalarT**,HostDevice> samples = sdata.getPoints();
    int numsamples = samples.extent(0);
    
    // Evaluate MILO or a surrogate at these samples
    vector<ScalarT> response_values;
    vector<Teuchos::RCP<MrHyDE_OptVector>> gradient_values;
    
    std::stringstream ss;
    std::string sname2 = "sampledata.dat";
    std::ofstream sdataOUT(sname2.c_str());
    sdataOUT.precision(16);
    
    if(Comm->getRank() == 0)
      cout << "Evaluating samples ..." << endl;
    
    for (int j=0; j<numsamples; j++) {
      vector<ScalarT> currparams;
      DFAD objfun = 0.0;
      for (int i=0; i<ptsdim; i++)  {
        currparams.push_back(samples(j,i));
      }
      if(Comm->getRank() == 0) {
        for (int i=0; i<ptsdim; i++)  {
          sdataOUT << samples(j,i) << "  ";
        }
      }
      params->updateParams(currparams,1);
      solve->forwardModel(objfun);
      response_values.push_back(objfun.val());
      if(Comm->getRank() == 0) {
        sdataOUT << response_values[j] << "  ";
      }
      Teuchos::RCP<MrHyDE_OptVector> currgradient;
      solve->adjointModel(*currgradient);
      //vector<ScalarT> currgradient = postproc->computeSensitivities(F_soln, A_soln);
      gradient_values.push_back(currgradient);
      /*
      if(Comm->getRank() == 0) {
        for (int paramiter=0; paramiter < ptsdim; paramiter++) {
          sdataOUT << gradient_values[j][paramiter] << "  ";
        }
        sdataOUT << endl;
      }
      */
      if(Comm->getRank() == 0)
        cout << "Finished evaluating sample number: " << j+1 << " out of " << numsamples << endl;
    }
    
    sdataOUT.close();
    
  }
  
  else if (analysis_type == "UQ") {
    
    // UQ is forward propagation of uncertainty computed using sampling
    
    // Build the uq manager
    Teuchos::ParameterList uqsettings = settings->sublist("Analysis").sublist("UQ");
    vector<string> param_types = params->stochastic_distribution;
    vector<ScalarT> param_means = params->getStochasticParams("mean");
    vector<ScalarT> param_vars = params->getStochasticParams("variance");
    vector<ScalarT> param_mins = params->getStochasticParams("min");
    vector<ScalarT> param_maxs = params->getStochasticParams("max");
    UQManager uq(Comm, uqsettings, param_types, param_means, param_vars, param_mins, param_maxs);
    
    // Generate the samples for the UQ
    int numstochparams = param_types.size();
    int numsamples = uqsettings.get<int>("samples",100);
    int maxsamples = uqsettings.get<int>("max samples",numsamples); // needed for generating subsets of samples
    int seed = uqsettings.get<int>("seed",1234);
    Kokkos::View<ScalarT**,HostDevice> samplepts = uq.generateSamples(maxsamples, seed);
    // Adjust the number of samples (if necessary)
    numsamples = std::min(numsamples, static_cast<int>(samplepts.extent(0)));
    Kokkos::View<int*,HostDevice> sampleints = uq.generateIntegerSamples(maxsamples, seed);
    bool regenerate_rotations = uqsettings.get<bool>("regenerate grain rotations",false);
    bool regenerate_grains = uqsettings.get<bool>("regenerate grains",false);
    // Evaluate model or a surrogate at these samples
    vector<Kokkos::View<ScalarT***,HostDevice> > response_values;
    vector<Kokkos::View<ScalarT****,HostDevice> > response_grads;
    vector_RCP avgsoln = solve->linalg->getNewOverlappedVector(0,2);
    int output_freq = uqsettings.get<int>("output frequency",1);
    if (uqsettings.get<bool>("use surrogate",false)) {
      
    }
    else {
      if (Comm->getRank() == 0) {
        cout << "Running Monte Carlo sampling ..." << endl;
      }
      for (int j=0; j<numsamples; j++) {
        if (numstochparams > 0) {
          vector<ScalarT> currparams;
          for (int i=0; i<numstochparams; i++) {
            currparams.push_back(samplepts(j,i));
          }
          DFAD objfun = 0.0;
          params->updateParams(currparams,2);
          
        }
        if (regenerate_grains) {
          auto seeds = solve->mesh->generateNewMicrostructure(sampleints(j));
          solve->mesh->importNewMicrostructure(sampleints(j), seeds,
                                               solve->assembler->groups,
                                               solve->assembler->boundary_groups);
        }
        else if (regenerate_rotations) {
          this->updateRotationData(sampleints(j));
        }
        
        std::stringstream ss;
        ss << "_" << j;
        postproc->append = ss.str();
        
        solve->forwardModel(objfun);
        postproc->report();
        
        if (Comm->getRank() == 0 && j%output_freq == 0) {
          cout << "Finished evaluating sample number: " << j+1 << " out of " << numsamples << endl;
        }
      }
      
    }
    
    if (Comm->getRank() == 0) {
      string sptname = "sample_points.dat";
      std::ofstream sampOUT(sptname.c_str());
      sampOUT.precision(6);
      for (size_type r=0; r<samplepts.extent(0); r++) {
        for (size_type d=0; d<samplepts.extent(1); d++) {
          sampOUT << samplepts(r,d) << "  ";
        }
        sampOUT << endl;
      }
      sampOUT.close();
      
      string sname = "sample_data.dat";
      std::ofstream respOUT(sname.c_str());
      respOUT.precision(6);
      for (size_t r=0; r<response_values.size(); r++) {
        for (size_type s=0; s<response_values[r].extent(0); s++) { // sensor index
          for (size_type t=0; t<response_values[r].extent(2); t++) { // time index
            for (size_type d=0; d<response_values[r].extent(1); d++) { // data index
              respOUT << response_values[r](s,d,t) << "  ";
            }
          }
        }
        respOUT << endl;
      }
      respOUT.close();
      
      if (settings->sublist("Postprocess").get<bool>("compute response forward gradient",false)) {
        string sname = "sample_grads.dat";
        std::ofstream gradOUT(sname.c_str());
        gradOUT.precision(6);
        for (size_t r=0; r<response_grads.size(); r++) {
          for (size_type s=0; s<response_grads[r].extent(0); s++) { // sensor index
            for (size_type t=0; t<response_grads[r].extent(2); t++) { // time index
              for (size_type d=0; d<response_grads[r].extent(1); d++) { // data index
                for (size_type p=0; d<response_grads[r].extent(1); d++) { // data index
                  gradOUT << response_grads[r](s,d,t,p) << "  ";
                }
              }
            }
          }
          gradOUT << endl;
        }
        gradOUT.close();
      }
      
    }
    
    if (settings->sublist("Postprocess").get("write solution",true)) {
      //postproc->writeSolution(avgsoln, "output_avg");
    }
    // Compute the statistics (mean, variance, probability levels, etc.)
    //uq.computeStatistics(response_values);
    
  }
  else if (analysis_type == "ROL") {
    typedef ScalarT RealT;
    
    Teuchos::RCP< ROL::Objective_MILO<RealT> > obj;
    Teuchos::ParameterList ROLsettings;
    
    sensIC = settings->sublist("Analysis").get("sensitivities IC", false);
    
    if (settings->sublist("Analysis").isSublist("ROL"))
      ROLsettings = settings->sublist("Analysis").sublist("ROL");
    else
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: MrHyDE could not find the ROL sublist in the imput file!  Abort!");
    
    // New ROL input syntax
    bool use_linesearch = settings->sublist("Analysis").get("Use Line Search",false);
    
    ROLsettings.sublist("General").sublist("Secant").set("Type","Limited-Memory BFGS");
    ROLsettings.sublist("Step").sublist("Descent Method").set("Type","Newton Krylov");
    ROLsettings.sublist("Step").sublist("Trust Region").set("Subproblem Solver","Truncated CG");
    
    RealT gtol     = ROLsettings.sublist("Status Test").get("Gradient Tolerance",1e-6);
    RealT stol     = ROLsettings.sublist("Status Test").get("Step Tolerance",1.e-12);
    int maxit      = ROLsettings.sublist("Status Test").get("Iteration Limit",100);
    
    // Turn off visualization while optimizing
    bool postproc_plot = postproc->write_solution;
    postproc->write_solution = false;
    
    Teuchos::RCP<std::ostream> outStream;
    outStream = Teuchos::rcp(&std::cout, false);
    // Generate data and get objective
    obj = Teuchos::rcp( new ROL::Objective_MILO<RealT> (solve, postproc, params));
    
    Teuchos::RCP< ROL::Step<RealT> > step;
    
    if (use_linesearch) {
      step = Teuchos::rcp( new ROL::LineSearchStep<RealT> (ROLsettings) );
    }
    else {
      step = Teuchos::rcp( new ROL::TrustRegionStep<RealT> (ROLsettings) );
    }
    
    Teuchos::RCP<ROL::StatusTest<RealT> > status = Teuchos::rcp( new ROL::StatusTest<RealT> (gtol, stol, maxit) );
    
    ROL::Algorithm<RealT> algo(step,status,false);
    
    
    bool have_dynamic = params->have_dynamic;

    int numClassicParams = params->getNumParams(1);
    int numDiscParams = params->getNumParams(4);
    int numParams = numClassicParams + numDiscParams;

    // Iteration vector.
    Teuchos::RCP<vector<ScalarT> > classic_params;
    vector<vector_RCP> disc_params;
    if (numClassicParams > 0) {
      classic_params = params->getParams(1);
    }
    else {
      classic_params = Teuchos::null;
    }
    if (numDiscParams > 0) {
      if (have_dynamic) {
        disc_params = params->getDynamicDiscretizedParams();
      }
      else {
        disc_params.push_back(params->Psol);
      }
    }
    
    MrHyDE_OptVector xtmp(disc_params, classic_params, Comm->getRank());
    Teuchos::RCP<ROL::Vector<ScalarT>> x = xtmp.clone();
    x->set(xtmp);

    //bound contraint
    Teuchos::RCP<ROL::Bounds<RealT> > con;
    bool bound_vars = ROLsettings.sublist("General").get("Bound Optimization Variables",false);

    if(bound_vars){
      
      //initialize max and min vectors for bounds
      Teuchos::RCP<vector<RealT> > minvec = Teuchos::rcp( new vector<RealT> (numParams, 0.0) );
      Teuchos::RCP<vector<RealT> > maxvec = Teuchos::rcp( new vector<RealT> (numParams, 0.0) );
      
      //read in bounds for parameters...
      vector<Teuchos::RCP<vector<ScalarT> > > activeBnds = params->getActiveParamBounds();
      vector<vector_RCP> discBnds = params->getDiscretizedParamBounds();
      Teuchos::RCP<ROL::Vector<ScalarT> > lo = Teuchos::rcp( new MrHyDE_OptVector(discBnds[0], activeBnds[0], Comm->getRank()) );
      Teuchos::RCP<ROL::Vector<ScalarT> > up = Teuchos::rcp( new MrHyDE_OptVector(discBnds[1], activeBnds[1], Comm->getRank()) );
      
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
      if (params->isParameter("datagen")) {
        vector<ScalarT> pval = {1.0};
        params->setParam(pval,"datagen");
      }
      postproc->response_type = "none";
      postproc->compute_objective = false;
      solve->forwardModel(objfun);
      //std::cout << "Storing data ... " << std::endl;
      
      for (size_t set=0; set<postproc->soln.size(); ++set) {
        vector<vector<ScalarT> > times = postproc->soln[set]->times;
        vector<vector<Teuchos::RCP<LA_MultiVector> > > data = postproc->soln[set]->data;
        
        for (size_t i=0; i<times.size(); i++) {
          for (size_t j=0; j<times[i].size(); j++) {
            postproc->datagen_soln[set]->store(data[i][j], times[i][j], i);
          }
        }
      }
      
      //std::cout << "Finished storing data" << std::endl;
      if (params->isParameter("datagen")) {
        vector<ScalarT> pval = {0.0};
        params->setParam(pval,"datagen");
      }
      postproc->response_type = "discrete";
      postproc->compute_objective = true;
      //std::cout << "Finished generating data for inversion " << std::endl;
    }
    
    
    // Comparing a gradient/Hessian with finite difference approximation
    if(ROLsettings.sublist("General").get("Do grad+hessvec check",true)){
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
      }
      
      // check gradient and Hessian-vector computation using finite differences
      obj->checkGradient(*x, *d, (Comm->getRank() == 0));
      
    }
    
    Teuchos::Time timer("Optimization Time",true);
    
    // Run algorithm.
    vector<std::string> output;
    if (bound_vars) {
      output = algo.run(*x, *obj, *con, (Comm->getRank() == 0 )); //only processor of rank 0 print outs
    }
    else {
      output = algo.run(*x, *obj, (Comm->getRank() == 0)); //only processor of rank 0 prints out
    }

    
    ScalarT optTime = timer.stop();

    if (ROLsettings.sublist("General").get("Write Final Parameters",false) ) {
      string outname = ROLsettings.get("Output File Name","ROL_out.txt");
      std::ofstream respOUT(outname);
      respOUT.precision(16);
      if (Comm->getRank() == 0 ) {
        
        for ( unsigned i = 0; i < output.size(); i++ ) {
          std::cout << output[i];
          respOUT << output[i];
        }
        x->print(std::cout);
      }
      Kokkos::fence();
      x->print(respOUT);

      if (Comm->getRank() == 0 ) {
        if (verbosity > 5) {
          cout << "Optimization time: " << optTime << " seconds" << endl;
          respOUT << "\nOptimization time: " << optTime << " seconds" << endl;
        }
      }
      respOUT.close();
      string outname2 = "final_params.dat";
      std::ofstream respOUT2(outname2);
      respOUT2.precision(16);
      x->print(respOUT2);
      respOUT2.close();
    }
    
    /*
    if (settings->sublist("Postprocess").get("write Hessian",false)){
      obj->printHess(settings->sublist("Postprocess").get("Hessian output file","hess.dat"),x,Comm->getRank());
    }
    if (settings->sublist("Analysis").get("write output",false)) {
      DFAD val = 0.0;
      solve->forwardModel(val);
      //postproc->writeSolution(settings->sublist("Postprocess").get<string>("Output File","output"));
    }
    */

    if (postproc_plot) {
      postproc->write_solution = true;
      DFAD objfun = 0.0;
      solve->forwardModel(objfun);
    }
  } //ROL
  else if (analysis_type == "ROL_SIMOPT") {
    /*
    typedef ScalarT RealT;
    typedef ROL::Vector<RealT> V;
    typedef ROL::StdVector<RealT> SV;
    
    Teuchos::RCP< Objective_MILO_SimOpt<RealT> > obj;
    Teuchos::ParameterList ROLsettings;
    
    sensIC = settings->sublist("Analysis").get("sensitivities IC", false);
    
    if (settings->sublist("Analysis").isSublist("ROL"))
      ROLsettings = settings->sublist("Analysis").sublist("ROL");
    else
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: MILO could not find the ROL sublist in the imput file!  Abort!");
    
    // New ROL input syntax
    bool use_linesearch = settings->sublist("Analysis").get("Use Line Search",false);
    
    ROLsettings.sublist("General").sublist("Secant").set("Type","Limited-Memory BFGS");
    ROLsettings.sublist("Step").sublist("Descent Method").set("Type","Newton Krylov");
    ROLsettings.sublist("Step").sublist("Trust Region").set("Subproblem Solver","Truncated CG");
    
    RealT gtol     = ROLsettings.sublist("Status Test").get("Gradient Tolerance",1e-6);
    RealT stol     = ROLsettings.sublist("Status Test").get("Step Tolerance",1.e-12);
    int maxit      = ROLsettings.sublist("Status Test").get("Iteration Limit",100);
    //RealT aktol    = ROLsettings.sublist("General").sublist("Krylov").get("Absolute Tolerance",1e-4);
    //RealT rktol    = ROLsettings.sublist("General").sublist("Krylov").get("Relative Tolerance",1e-2);
    //int maxKiter   = ROLsettings.sublist("General").sublist("Krylov").get("Iteration Limit",100);
    
    Teuchos::RCP<std::ostream> outStream;
    outStream = Teuchos::rcp(&std::cout, false);
    // Generate data and get objective
    obj = Teuchos::rcp( new Objective_MILO_SimOpt<RealT> (solve, postproc, params));
    
    Teuchos::RCP< ROL::Step<RealT> > step;
    
    if(use_linesearch)
      step = Teuchos::rcp( new ROL::LineSearchStep<RealT> (ROLsettings) );
    else
      step = Teuchos::rcp( new ROL::TrustRegionStep<RealT> (ROLsettings) );
    
    //ROL::StatusTest<RealT> status(gtol, stol, maxit);
    Teuchos::RCP<ROL::StatusTest<RealT> > status = Teuchos::rcp( new ROL::StatusTest<RealT> (gtol, stol, maxit) );
    
    ROL::Algorithm<RealT> algo(step,status,false);
    //ROL::Algorithm<RealT> algo(*step,status,false);
    
    //int numParams = solve->getNumParams(1);
    //vector<ScalarT> params = solve->getParams(1);
    
    int numClassicParams = params->getNumParams(1);
    int numDiscParams = params->getNumParams(4);
    int numParams = numClassicParams + numDiscParams;
    vector<ScalarT> classic_params;
    vector<ScalarT> disc_params;
    if (numClassicParams > 0)
      classic_params = params->getParams(1);
    if (numDiscParams > 0)
      disc_params = params->getDiscretizedParamsVector();
    
    // Iteration vector.
    Teuchos::RCP<vector<RealT> > x_rcp = Teuchos::rcp( new vector<RealT> (numParams, 0.0) );
    // Set initial guess.
    
    int pprog  = 0;
    for (int i=0; i<numClassicParams; i++) {
      (*x_rcp)[pprog] = classic_params[i];
      pprog++;
    }
    for (int i=0; i<numDiscParams; i++) {
      (*x_rcp)[pprog] = disc_params[i];
      pprog++;
    }
    
    ROL::StdVector<RealT> x(x_rcp);
    
    //bound contraint
    Teuchos::RCP<ROL::Bounds<RealT> > con;
    bool bound_vars = ROLsettings.sublist("General").get("Bound Optimization Variables",false);
    if(bound_vars){
      
      //bool use_scale = ROLsettings.get("Use Scaling For Epsilon-Active Sets",false);
      //RealT scale;
      //if(use_scale){
      //  RealT tol = 1.e-12; //should probably be read in, though we're not using inexact gradients yet anyways...
      //  Teuchos::RCP<vector<RealT> > g0_rcp = Teuchos::rcp( new vector<RealT> (numParams, 0.0) );
      //  ROL::StdVector<RealT> g0p(g0_rcp);
      //  (*obj).gradient(g0p,x,tol);
      //  scale = 1.0e-2/g0p.norm();
      //}
      //else {
      //  scale = 1.0;
      //}
      
      // TMW: where is scale used?
      
      //initialize max and min vectors for bounds
      Teuchos::RCP<vector<RealT> > minvec = Teuchos::rcp( new vector<RealT> (numParams, 0.0) );
      Teuchos::RCP<vector<RealT> > maxvec = Teuchos::rcp( new vector<RealT> (numParams, 0.0) );
      
      //read in bounds for parameters...
      vector<vector<ScalarT> > classicBnds = params->getParamBounds("active");
      vector<vector<ScalarT> > discBnds = params->getParamBounds("discretized");
      
      pprog = 0;
      
      if (classicBnds[0].size() > 0) {
        for (size_t i = 0; i <classicBnds[0].size(); i++) {
          (*minvec)[pprog] = classicBnds[0][i];
          (*maxvec)[pprog] = classicBnds[1][i];
          pprog++;
        }
      }
      
      if (discBnds[0].size() > 0) {
        for (size_t i = 0; i < discBnds[0].size(); i++) {
          (*minvec)[pprog] = discBnds[0][i];
          (*maxvec)[pprog] = discBnds[1][i];
          pprog++;
        }
      }
      
      Teuchos::RCP<V> lo = Teuchos::rcp( new SV(minvec) );
      Teuchos::RCP<V> up = Teuchos::rcp( new SV(maxvec) );
      
      con = Teuchos::rcp(new ROL::Bounds<RealT>(lo,up));
      
      //create bound constraint
    }
    
    if(ROLsettings.sublist("General").get("Do grad+hessvec check",true)){
      //if(ROLsettings.get<bool>("Do grad+hessvec check","true")){
      // Gradient and Hessian check
      // direction for gradient check
      if (ROLsettings.sublist("General").isParameter("FD Check Seed")) {
        int seed = ROLsettings.get("FD Check Seed",1);
        srand(seed);
      }
      else
        srand(time(NULL)); //initialize random seed
      
      Teuchos::RCP<vector<RealT> > d_rcp = Teuchos::rcp( new vector<RealT> (numParams, 0.0) );
      bool no_random_vec = ROLsettings.sublist("General").get("FD Check Use Ones Vector",false);
      if (no_random_vec) {
        for ( int i = 0; i < numParams; i++ ) {
          (*d_rcp)[i] = 1.0;
        }
      }
      else {
        for ( int i = 0; i < numParams; i++ ) {
          (*d_rcp)[i] = 10.0*(ScalarT)rand()/(ScalarT)RAND_MAX - 5.0;
        }
      }
      ROL::StdVector<RealT> d(d_rcp);
      // check gradient and Hessian-vector computation using finite differences
      (*obj).checkGradient(x, d, (Comm->getRank() == 0 ));
      //(*obj).checkHessVec(x, d, true); //Hessian-vector is already done with FD.
      
    }
    
    Teuchos::Time timer("Optimization Time",true);
    
    // Run algorithm.
    vector<std::string> output;
    if(bound_vars)
      output = algo.run(x, *obj, *con, (Comm->getRank() == 0 )); //only processor of rank 0 print outs
    else
      output = algo.run(x, *obj, (Comm->getRank() == 0 )); //only processor of rank 0 prints out
    
    ScalarT optTime = timer.stop();
    if (Comm->getRank() == 0 ) {
      string outname = ROLsettings.get("Output File Name","ROL_out.txt");
      std::ofstream respOUT(outname);
      respOUT.precision(16);
      for ( unsigned i = 0; i < output.size(); i++ ) {
        std::cout << output[i];
        respOUT << output[i];
      }
      for (int i=0; i<numParams; i++) {
        cout << "param " << i << " = " << (*x_rcp)[i] << endl;
        respOUT << "param " << i << " = " << (*x_rcp)[i] << endl;
      }
      //bvbw
      if (verbosity > 5) {
        cout << "Optimization time: " << optTime << " seconds" << endl;
        respOUT << "\nOptimization time: " << optTime << " seconds" << endl;
      }
      respOUT.close();
      string outname2 = "final_params.dat";
      std::ofstream respOUT2(outname2);
      respOUT2.precision(16);
      for (int i=0; i<numParams; i++) {
        respOUT2 << (*x_rcp)[i] << endl;
      }
      respOUT2.close();
    }
    
    if (settings->sublist("Postprocess").get("write Hessian",false)){
      obj->printHess(settings->sublist("Postprocess").get("Hessian output file","hess.dat"),x,Comm->getRank());
    }
    if (settings->sublist("Analysis").get("write output",false)) {
      DFAD val = 0.0;
      solve->forwardModel(val);
      //postproc->writeSolution(settings->sublist("Postprocess").get<string>("Output File","output"));
    }*/
  } //ROL_SIMOPT
  else { // don't solve anything, but produce visualization
    std::cout << "Unknown analysis option: " << analysis_type << std::endl;
    std::cout << "Valid and tested options: dry run, forward, forward+adjoint, UQ, ROL" << std::endl;
  }
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished analysis::run" << endl;
    }
  }
  
}


// ========================================================================================
// ========================================================================================

void AnalysisManager::updateRotationData(const int & newrandseed) {
  
  // Determine how many seeds there are
  size_t localnumSeeds = 0;
  size_t numSeeds = 0;
  for (size_t block=0; block<solve->assembler->groups.size(); ++block) {
    for (size_t grp=0; grp<solve->assembler->groups[block].size(); ++grp) {
      for (size_t e=0; e<solve->assembler->groups[block][grp]->numElem; ++e) {
        if (solve->assembler->groups[block][grp]->data_seed[e] > localnumSeeds) {
          localnumSeeds = solve->assembler->groups[block][grp]->data_seed[e];
        }
      }
    }
  }
  //Comm->MaxAll(&localnumSeeds, &numSeeds, 1);
  Teuchos::reduceAll<int,size_t>(*Comm,Teuchos::REDUCE_MAX,1,&localnumSeeds,&numSeeds);
  numSeeds += 1; //To properly allocate and iterate
  
  // Create a random number generator
  std::default_random_engine generator(newrandseed);
  
  ////////////////////////////////////////////////////////////////////////////////
  // Set seed data
  ////////////////////////////////////////////////////////////////////////////////
  
  int numdata = 9;
  
  //cout << "solver numSeeds = " << numSeeds << endl;
  
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
  
  for (size_t block=0; block<solve->assembler->groups.size(); ++block) {
    for (size_t grp=0; grp<solve->assembler->groups[block].size(); ++grp) {
      int numElem = solve->assembler->groups[block][grp]->numElem;
      for (int c=0; c<numElem; c++) {
        int cnode = solve->assembler->groups[block][grp]->data_seed[c];
        for (int i=0; i<9; i++) {
          solve->assembler->groups[block][grp]->data(c,i) = rotation_data(cnode,i);
        }
      }
    }
  }
  for (size_t block=0; block<solve->assembler->boundary_groups.size(); ++block) {
    for (size_t grp=0; grp<solve->assembler->boundary_groups[block].size(); ++grp) {
      int numElem = solve->assembler->boundary_groups[block][grp]->numElem;
      for (int e=0; e<numElem; ++e) {
        int cnode = solve->assembler->boundary_groups[block][grp]->data_seed[e];
        for (int i=0; i<9; i++) {
          solve->assembler->boundary_groups[block][grp]->data(e,i) = rotation_data(cnode,i);
        }
      }
    }
  }
  solve->multiscale_manager->updateMeshData(rotation_data);
}
