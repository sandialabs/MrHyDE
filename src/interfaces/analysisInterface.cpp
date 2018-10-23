/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "analysisInterface.hpp"
#include "physicsInterface.hpp"
#include "discretizationInterface.hpp"
#include "uqInterface.hpp"
#include "CDBatchManager.hpp";
#include "obj_milorol.hpp"
#include "ROL_StdVector.hpp"
#include "obj_milorol_simopt.hpp"


// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

analysis::analysis(const Teuchos::RCP<Epetra_MpiComm> & LA_Comm_,
                   const Teuchos::RCP<Epetra_MpiComm> & S_Comm_,
                   Teuchos::RCP<Teuchos::ParameterList> & settings_,
                   Teuchos::RCP<solver> & solver_, Teuchos::RCP<postprocess> & postproc_) :
LA_Comm(LA_Comm_), S_Comm(S_Comm_), settings(settings_), solve(solver_), postproc(postproc_){
  verbosity = settings->get<int>("verbosity",0);
}


// ========================================================================================
/* given the parameters, solve the forward  problem */
// ========================================================================================

void analysis::run() {
  
  std::string analysis_type = settings->sublist("Analysis").get<string>("analysis type","forward");
  DFAD objfun = 0.0;
  
  if (analysis_type == "forward") {
    
    vector_RCP F_soln = solve->forwardModel(objfun);
    
    if (settings->sublist("Postprocess").get("compute response",false))
    postproc->computeResponse(F_soln);
    
    //if (settings->sublist("Postprocess").get("compute objective",false))
    //  AD objfun = postproc->computeObjective(F_soln);
    if (settings->sublist("Postprocess").get("verification",false))
    postproc->computeError(F_soln);
    if (settings->sublist("Postprocess").get("write solution",true))
    postproc->writeSolution(F_soln, settings->sublist("Postprocess").get<string>("Output File","output"));
    
    
  }
  else if (analysis_type == "forward_fr") {
    double s = settings->sublist("Physics").get<double>("frac_exp",0.2);
    //      double s = 0.2;
    
    //      double s = 0.2;
    double N = 100;
    double h = 1/(sqrt(N));
    double k = 1/(log(1/h));
    double pi = 3.14159265359;
    
    int Nplus = (int) ceil(pi*pi/(4*s*k*k));
    int Nminus = (int) ceil(pi*pi/(4*(1-s)*k*k));
    
    vector_RCP F_soln = solve->forwardModel(objfun);
    for (int ell = -Nminus; ell < Nplus; ell++) {
      double y = k*ell;
      //	vector_RCP F_soln_inter = solve->forwardModel(objfun);
      vector_RCP F_soln_inter = solve->forwardModel_fr(objfun,y,s);
      F_soln->Update(1.0,*F_soln_inter, 1.0);
    }
    if (settings->sublist("Postprocess").get("compute response",false))
    postproc->computeResponse(F_soln);
    
    //if (settings->sublist("Postprocess").get("compute objective",false))
    //  AD objfun = postproc->computeObjective(F_soln);
    if (settings->sublist("Postprocess").get("verification",false))
    postproc->computeError(F_soln);
    if (settings->sublist("Postprocess").get("write solution",true))
    postproc->writeSolution(F_soln, settings->sublist("Postprocess").get<string>("Output File","output"));
    
  }
  else if (analysis_type == "forward+adjoint") {
    vector_RCP F_soln = solve->forwardModel(objfun);
    if (settings->sublist("Postprocess").get<bool>("compute response",false))
      postproc->computeResponse(F_soln);
    
    //if (settings->sublist("Postprocess").get("compute objective",false))
    //  AD objfun = postproc->computeObjective(F_soln);
    if (settings->sublist("Postprocess").get<bool>("verification",false))
      postproc->computeError(F_soln);
    
    vector_RCP A_soln = solve->adjointModel(F_soln, gradient);
    //if (settings->sublist("Postprocess").get<bool>("compute sensitivities",false))
    //  gradient = postproc->computeSensitivities(F_soln, A_soln);
    if (settings->sublist("Postprocess").get<bool>("error estimate",false))
      double errorest = solve->computeError(F_soln, A_soln);
    
    if (settings->sublist("Postprocess").get("write solution",true)) {
      postproc->writeSolution(F_soln, settings->sublist("Postprocess").get<string>("Output File","output"));
      postproc->writeSolution(A_soln, settings->sublist("Postprocess").get<string>("Adjoint Output File","adj_output"));
    }
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
    int ptsdim = sampsettings.get<int>("dimension");
    data sdata("Sample Points", ptsdim, sampsettings.get("source","samples.dat"));
    Kokkos::View<double**,HostDevice> samples = sdata.getpoints();
    int numsamples = samples.dimension(0);
    
    // Evaluate MILO or a surrogate at these samples
    vector<double> response_values;
    vector<vector<double> > gradient_values;
    
    stringstream ss;
    std::string sname2 = "sampledata.dat";
    ofstream sdataOUT(sname2.c_str());
    sdataOUT.precision(16);
    
    
    if(S_Comm->MyPID() == 0)
    cout << "Evaluating samples ..." << endl;
    
    for (int j=0; j<numsamples; j++) {
      vector<double> currparams;
      DFAD objfun = 0.0;
      for (int i=0; i<ptsdim; i++)  {
        currparams.push_back(samples(j,i));
      }
      if(S_Comm->MyPID() == 0) {
        for (int i=0; i<ptsdim; i++)  {
          sdataOUT << samples(j,i) << "  ";
        }
      }
      solve->updateParams(currparams,1);
      vector_RCP F_soln = solve->forwardModel(objfun);
      AD currresponse = postproc->computeObjective(F_soln);
      response_values.push_back(currresponse.val());
      if(S_Comm->MyPID() == 0) {
        sdataOUT << response_values[j] << "  ";
      }
      vector<double> currgradient;
      vector_RCP A_soln = solve->adjointModel(F_soln, currgradient);
      //vector<double> currgradient = postproc->computeSensitivities(F_soln, A_soln);
      gradient_values.push_back(currgradient);
      if(S_Comm->MyPID() == 0) {
        for (size_t paramiter=0; paramiter < ptsdim; paramiter++) {
          sdataOUT << gradient_values[j][paramiter] << "  ";
        }
        sdataOUT << endl;
      }
      
      if(S_Comm->MyPID() == 0)
      cout << "Finished evaluating sample number: " << j+1 << " out of " << numsamples << endl;
    }
    
    sdataOUT.close();
    
  }
  
  else if (analysis_type == "UQ") {
    
    // UQ is forward propagation of uncertainty computed using sampling
    // We may sample either MILO (Monte Carlo), or
    // We may sample a surrogate model ... we first need to build this surrogate model (perhaps adaptively)
    // Note: PCE models provide analytical estimates of the mean and variance, but we will make no use of this
    
    // Build the uq manager
    Teuchos::ParameterList uqsettings = settings->sublist("Analysis").sublist("UQ");
    vector<string> param_types = solve->stochastic_distribution;
    vector<double> param_means = solve->getStochasticParams("mean");
    vector<double> param_vars = solve->getStochasticParams("variance");
    vector<double> param_mins = solve->getStochasticParams("min");
    vector<double> param_maxs = solve->getStochasticParams("max");
    uqmanager uq(*LA_Comm, uqsettings, param_types, param_means, param_vars, param_mins, param_maxs);
    
    // Generate the samples for the UQ
    int numstochparams = param_types.size();
    int numsamples = uqsettings.get<int>("Samples",100);
    int seed = uqsettings.get<int>("Seed",1234);
    Kokkos::View<double**,HostDevice> samplepts = uq.generateSamples(numsamples, seed);
    Kokkos::View<int*,HostDevice> sampleints = uq.generateIntegerSamples(numsamples, seed);
    bool regenerate_meshdata = uqsettings.get<bool>("Regenerate mesh data",false);
    // Evaluate MILO or a surrogate at these samples
    vector<Kokkos::View<double***,HostDevice> > response_values;
    Teuchos::RCP<Epetra_Map> emap = solve->LA_overlapped_map;
    vector_RCP avgsoln = Teuchos::rcp(new Epetra_MultiVector(*emap, 2));
    int output_freq = uqsettings.get<int>("Output Frequency",1);
    if (uqsettings.get<bool>("Use Surrogate",false)) {
      
    }
    else {
      cout << "Running Monte Carlo sampling ..." << endl;
      for (int j=0; j<numsamples; j++) {
        vector<double> currparams;
        for (int i=0; i<numstochparams; i++) {
          currparams.push_back(samplepts(j,i));
        }
        DFAD objfun = 0.0;
        solve->updateParams(currparams,2);
        if (regenerate_meshdata) {
          solve->updateMeshData(sampleints(j));
        }
        vector_RCP F_soln = solve->forwardModel(objfun);
        avgsoln->Update(1.0/(double)numsamples, *F_soln, 1.0);
        /*if (settings->sublist("Postprocess").get("write solution",true)) {
         stringstream ss;
         ss << j;
         string str = ss.str();
         postproc->writeSolution(F_soln, "sampling_data/outputMC_" + str + "_.exo");
         }*/
        if (settings->sublist("Postprocess").get<bool>("compute response",false)) {
          Kokkos::View<double***,HostDevice> currresponse = postproc->computeResponse(F_soln,0);
          for (size_t i=0; i<currresponse.dimension(0); i++) {
            for (size_t j=0; j<currresponse.dimension(1); j++) {
              for (size_t k=0; k<currresponse.dimension(2); k++) {
                double myval = currresponse(i,j,k);
                double gval = 0.0;
                LA_Comm->SumAll(&myval, &gval, 1);
                currresponse(i,j,k) = gval;
              }
            }
          }
          
          response_values.push_back(currresponse);
        }
        if (LA_Comm->MyPID() == 0 && j%output_freq == 0) {
          cout << "Finished evaluating sample number: " << j+1 << " out of " << numsamples << endl;
        }
      }
      
    }
    
    if (LA_Comm->MyPID() == 0) {
      string sptname = "sample_points.dat";
      ofstream sampOUT(sptname.c_str());
      sampOUT.precision(6);
      for (int r=0; r<samplepts.dimension(0); r++) {
        for (int d=0; d<samplepts.dimension(1); d++) {
          sampOUT << samplepts(r,d) << "  ";
        }
        sampOUT << endl;
      }
      sampOUT.close();
      
      string sname = "sample_data.dat";
      ofstream respOUT(sname.c_str());
      respOUT.precision(6);
      for (int r=0; r<response_values.size(); r++) {
        for (int s=0; s<response_values[r].dimension(0); s++) { // sensor index
          for (int t=0; t<response_values[r].dimension(2); t++) { // time index
            for (int d=0; d<response_values[r].dimension(1); d++) { // data index
              respOUT << response_values[r](s,d,t) << "  ";
            }
          }
        }
        respOUT << endl;
      }
      respOUT.close();
    }
    
    if (settings->sublist("Postprocess").get("write solution",true)) {
      postproc->writeSolution(avgsoln, "output_avg");
    }
    // Compute the statistics (mean, variance, probability levels, etc.)
    //uq.computeStatistics(response_values);
    
    
    /*
     int numParams = solve.getNumParams("stochastic");
     bool adaptive = uqsettings.get<bool>("Adaptive",false);
     std::string adaptive_criteria = uqsettings.get<std::string>("Adaptive Criteria","");
     vector<vector<double> > params = uq.getNewPoints();
     size_t numpts = params.size();
     bool done = false;
     if (numpts == 0)
     done = true;
     
     vector<Epetra_MultiVector> fwdsols;
     vector<Epetra_MultiVector> adjsols;
     vector<double> responsevals;
     vector<double> errorvals;
     //array<double> newpoints;
     
     while (!done) {
     for (int j=0; j++; j<numpts) {
     vector<double> currparams = params[j];
     solve.updateParams(currparams,"stochastic");
     Epetra_MultiVector F_soln = solve.forwardModel();
     fwdsols.push_back(F_soln);
     if (settings->sublist("Postprocess").get<bool>("compute response",false)) {
     double currresponse = postproc.computeResponse(F_soln);
     responsevals.push_back(currresponse);
     }
     //Epetra_MultiVector A_soln = solve.adjointModel(F_soln);
     //adjsols.push_back(A_soln);
     }
     if (adaptive) {
     
     }
     else
     done = true;
     }
     */
    
  }
  else if (analysis_type == "ROL") {
    typedef double RealT;
    typedef ROL::Vector<RealT> V;
    typedef ROL::StdVector<RealT> SV;
    
    Teuchos::RCP< ROL::Objective_MILO<RealT> > obj;
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
    RealT aktol    = ROLsettings.sublist("General").sublist("Krylov").get("Absolute Tolerance",1e-4);
    RealT rktol    = ROLsettings.sublist("General").sublist("Krylov").get("Relative Tolerance",1e-2);
    int maxKiter   = ROLsettings.sublist("General").sublist("Krylov").get("Iteration Limit",100);
    
    Teuchos::RCP<std::ostream> outStream;
    outStream = Teuchos::rcp(&std::cout, false);
    // Generate data and get objective
    obj = Teuchos::rcp( new ROL::Objective_MILO<RealT> (solve, postproc));
    
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
    //vector<double> params = solve->getParams(1);
    
    int numClassicParams = solve->getNumParams(1);
    int numDiscParams = solve->getNumParams(4);
    int numParams = numClassicParams + numDiscParams;
    vector<double> classic_params;
    vector<double> disc_params;
    if (numClassicParams > 0)
    classic_params = solve->getParams(1);
    if (numDiscParams > 0)
    disc_params = solve->getDiscretizedParamsVector();
    
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
      bool use_scale = ROLsettings.get("Use Scaling For Epsilon-Active Sets",false);
      RealT scale;
      if(use_scale){
        RealT tol = 1.e-12; //should probably be read in, though we're not using inexact gradients yet anyways...
        Teuchos::RCP<vector<RealT> > g0_rcp = Teuchos::rcp( new vector<RealT> (numParams, 0.0) );
        ROL::StdVector<RealT> g0p(g0_rcp);
        (*obj).gradient(g0p,x,tol);
        scale = 1.0e-2/g0p.norm();
      }
      else
      scale = 1.0;
      
      //initialize max and min vectors for bounds
      Teuchos::RCP<vector<RealT> > minvec = Teuchos::rcp( new vector<RealT> (numParams, 0.0) );
      Teuchos::RCP<vector<RealT> > maxvec = Teuchos::rcp( new vector<RealT> (numParams, 0.0) );
      
      //read in bounds for parameters...
      vector<vector<double> > classicBnds = solve->getParamBounds("active");
      vector<vector<double> > discBnds = solve->getParamBounds("discretized");
      
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
        for ( unsigned i = 0; i < numParams; i++ ) {
          (*d_rcp)[i] = 1.0;
        }
      }
      else {
        for ( unsigned i = 0; i < numParams; i++ ) {
          (*d_rcp)[i] = 10.0*(double)rand()/(double)RAND_MAX - 5.0;
        }
      }
      ROL::StdVector<RealT> d(d_rcp);
      // check gradient and Hessian-vector computation using finite differences
      (*obj).checkGradient(x, d, (LA_Comm->MyPID() == 0 && S_Comm->MyPID() == 0));
      //(*obj).checkHessVec(x, d, true); //Hessian-vector is already done with FD.
      
    }
    
    Teuchos::Time timer("Optimization Time",true);
    
    // Run algorithm.
    vector<std::string> output;
    if(bound_vars)
    output = algo.run(x, *obj, *con, (LA_Comm->MyPID() == 0 && S_Comm->MyPID() == 0)); //only processor of rank 0 print outs
    else
    output = algo.run(x, *obj, (LA_Comm->MyPID() == 0 && S_Comm->MyPID() == 0)); //only processor of rank 0 prints out
    
    double optTime = timer.stop();
    if (LA_Comm->MyPID() == 0 && S_Comm->MyPID() == 0) {
      string outname = ROLsettings.get("Output File Name","ROL_out.txt");
      ofstream respOUT(outname);
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
      ofstream respOUT2(outname2);
      respOUT2.precision(16);
      for (int i=0; i<numParams; i++) {
        respOUT2 << (*x_rcp)[i] << endl;
      }
      respOUT2.close();
    }
    
    if (settings->sublist("Postprocess").get("Write Hessian",false)){
      obj->printHess(settings->sublist("Postprocess").get("Hessian Output File","hess.dat"),x,LA_Comm->MyPID());
    }
    if (settings->sublist("Analysis").get("Write Output",false)) {
      DFAD val = 0.0;
      vector_RCP F_soln = solve->forwardModel(val);
      postproc->writeSolution(F_soln, settings->sublist("Postprocess").get<string>("Output File","output"));
    }
  } //ROL
  else if (analysis_type == "ROL_SIMOPT") {
    typedef double RealT;
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
    RealT aktol    = ROLsettings.sublist("General").sublist("Krylov").get("Absolute Tolerance",1e-4);
    RealT rktol    = ROLsettings.sublist("General").sublist("Krylov").get("Relative Tolerance",1e-2);
    int maxKiter   = ROLsettings.sublist("General").sublist("Krylov").get("Iteration Limit",100);
    
    Teuchos::RCP<std::ostream> outStream;
    outStream = Teuchos::rcp(&std::cout, false);
    // Generate data and get objective
    obj = Teuchos::rcp( new Objective_MILO_SimOpt<RealT> (solve, postproc));
    
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
    //vector<double> params = solve->getParams(1);
    
    int numClassicParams = solve->getNumParams(1);
    int numDiscParams = solve->getNumParams(4);
    int numParams = numClassicParams + numDiscParams;
    vector<double> classic_params;
    vector<double> disc_params;
    if (numClassicParams > 0)
    classic_params = solve->getParams(1);
    if (numDiscParams > 0)
    disc_params = solve->getDiscretizedParamsVector();
    
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
      bool use_scale = ROLsettings.get("Use Scaling For Epsilon-Active Sets",false);
      RealT scale;
      if(use_scale){
        RealT tol = 1.e-12; //should probably be read in, though we're not using inexact gradients yet anyways...
        Teuchos::RCP<vector<RealT> > g0_rcp = Teuchos::rcp( new vector<RealT> (numParams, 0.0) );
        ROL::StdVector<RealT> g0p(g0_rcp);
        (*obj).gradient(g0p,x,tol);
        scale = 1.0e-2/g0p.norm();
      }
      else
      scale = 1.0;
      
      //initialize max and min vectors for bounds
      Teuchos::RCP<vector<RealT> > minvec = Teuchos::rcp( new vector<RealT> (numParams, 0.0) );
      Teuchos::RCP<vector<RealT> > maxvec = Teuchos::rcp( new vector<RealT> (numParams, 0.0) );
      
      //read in bounds for parameters...
      vector<vector<double> > classicBnds = solve->getParamBounds("active");
      vector<vector<double> > discBnds = solve->getParamBounds("discretized");
      
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
        for ( unsigned i = 0; i < numParams; i++ ) {
          (*d_rcp)[i] = 1.0;
        }
      }
      else {
        for ( unsigned i = 0; i < numParams; i++ ) {
          (*d_rcp)[i] = 10.0*(double)rand()/(double)RAND_MAX - 5.0;
        }
      }
      ROL::StdVector<RealT> d(d_rcp);
      // check gradient and Hessian-vector computation using finite differences
      (*obj).checkGradient(x, d, (LA_Comm->MyPID() == 0 && S_Comm->MyPID() == 0));
      //(*obj).checkHessVec(x, d, true); //Hessian-vector is already done with FD.
      
    }
    
    Teuchos::Time timer("Optimization Time",true);
    
    // Run algorithm.
    vector<std::string> output;
    if(bound_vars)
    output = algo.run(x, *obj, *con, (LA_Comm->MyPID() == 0 && S_Comm->MyPID() == 0)); //only processor of rank 0 print outs
    else
    output = algo.run(x, *obj, (LA_Comm->MyPID() == 0 && S_Comm->MyPID() == 0)); //only processor of rank 0 prints out
    
    double optTime = timer.stop();
    if (LA_Comm->MyPID() == 0 && S_Comm->MyPID() == 0) {
      string outname = ROLsettings.get("Output File Name","ROL_out.txt");
      ofstream respOUT(outname);
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
      ofstream respOUT2(outname2);
      respOUT2.precision(16);
      for (int i=0; i<numParams; i++) {
        respOUT2 << (*x_rcp)[i] << endl;
      }
      respOUT2.close();
    }
    
    if (settings->sublist("Postprocess").get("Write Hessian",false)){
      obj->printHess(settings->sublist("Postprocess").get("Hessian Output File","hess.dat"),x,LA_Comm->MyPID());
    }
    if (settings->sublist("Analysis").get("Write Output",false)) {
      DFAD val = 0.0;
      vector_RCP F_soln = solve->forwardModel(val);
      postproc->writeSolution(F_soln, settings->sublist("Postprocess").get<string>("Output File","output"));
    }
  } //ROL_SIMOPT
  else { // don't solve anything, but produce visualization
    vector_RCP zero_sol = solve->blankState();
    if (settings->sublist("Postprocess").get("write solution",true))
    postproc->writeSolution(zero_sol, settings->sublist("Postprocess").get<string>("Output File","output"));
    
  }
}
