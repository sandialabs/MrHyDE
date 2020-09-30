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
#include "uqManager.hpp"
#include "CDBatchManager.hpp"
#include "obj_milorol.hpp"
#include "ROL_StdVector.hpp"
#include "obj_milorol_simopt.hpp"
#include "ROL_LineSearchStep.hpp"
#include "ROL_Algorithm.hpp"
#include "ROL_Bounds.hpp"
#include "ROL_TrustRegionStep.hpp"


// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

analysis::analysis(const Teuchos::RCP<MpiComm> & Comm_,
                   Teuchos::RCP<Teuchos::ParameterList> & settings_,
                   Teuchos::RCP<solver> & solver_,
                   Teuchos::RCP<PostprocessManager> & postproc_,
                   Teuchos::RCP<ParameterManager> & params_) :
Comm(Comm_), settings(settings_), solve(solver_),
postproc(postproc_), params(params_) {
  verbosity = settings->get<int>("verbosity",0);
  milo_debug_level = settings->get<int>("debug level",0);
  // No debug output on this constructor
}


// ========================================================================================
/* given the parameters, solve the forward  problem */
// ========================================================================================

void analysis::run() {
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting analysis::run ..." << endl;
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
    
    solve->adjointModel(gradient);
    
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
    Kokkos::View<ScalarT**,HostDevice> samples = sdata.getpoints();
    int numsamples = samples.extent(0);
    
    // Evaluate MILO or a surrogate at these samples
    vector<ScalarT> response_values;
    vector<vector<ScalarT> > gradient_values;
    
    stringstream ss;
    std::string sname2 = "sampledata.dat";
    ofstream sdataOUT(sname2.c_str());
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
      AD currresponse = postproc->computeObjective();
      response_values.push_back(currresponse.val());
      if(Comm->getRank() == 0) {
        sdataOUT << response_values[j] << "  ";
      }
      vector<ScalarT> currgradient;
      solve->adjointModel(currgradient);
      //vector<ScalarT> currgradient = postproc->computeSensitivities(F_soln, A_soln);
      gradient_values.push_back(currgradient);
      if(Comm->getRank() == 0) {
        for (size_t paramiter=0; paramiter < ptsdim; paramiter++) {
          sdataOUT << gradient_values[j][paramiter] << "  ";
        }
        sdataOUT << endl;
      }
      
      if(Comm->getRank() == 0)
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
    vector<string> param_types = params->stochastic_distribution;
    vector<ScalarT> param_means = params->getStochasticParams("mean");
    vector<ScalarT> param_vars = params->getStochasticParams("variance");
    vector<ScalarT> param_mins = params->getStochasticParams("min");
    vector<ScalarT> param_maxs = params->getStochasticParams("max");
    uqmanager uq(*Comm, uqsettings, param_types, param_means, param_vars, param_mins, param_maxs);
    
    // Generate the samples for the UQ
    int numstochparams = param_types.size();
    int numsamples = uqsettings.get<int>("samples",100);
    int maxsamples = uqsettings.get<int>("max samples",numsamples); // needed for generating subsets of samples
    int seed = uqsettings.get<int>("seed",1234);
    Kokkos::View<ScalarT**,HostDevice> samplepts = uq.generateSamples(maxsamples, seed);
    Kokkos::View<int*,HostDevice> sampleints = uq.generateIntegerSamples(maxsamples, seed);
    bool regenerate_meshdata = uqsettings.get<bool>("regenerate mesh data",false);
    // Evaluate MILO or a surrogate at these samples
    vector<Kokkos::View<ScalarT***,HostDevice> > response_values;
    vector<Kokkos::View<ScalarT****,HostDevice> > response_grads;
    Teuchos::RCP<const LA_Map> emap = solve->LA_overlapped_map;
    vector_RCP avgsoln = Teuchos::rcp(new LA_MultiVector(emap, 2));
    int output_freq = uqsettings.get<int>("output frequency",1);
    if (uqsettings.get<bool>("use surrogate",false)) {
      
    }
    else {
      cout << "Running Monte Carlo sampling ..." << endl;
      for (int j=0; j<numsamples; j++) {
        vector<ScalarT> currparams;
        for (int i=0; i<numstochparams; i++) {
          currparams.push_back(samplepts(j,i));
        }
        DFAD objfun = 0.0;
        params->updateParams(currparams,2);
        if (regenerate_meshdata) {
          solve->mesh->updateMeshData(sampleints(j),solve->assembler->cells, solve->multiscale_manager);
        }
        solve->forwardModel(objfun);
        //vector_RCP A_soln = solve->adjointModel(F_soln, gradient);
        //avgsoln->update(1.0/(ScalarT)numsamples, *F_soln, 1.0);
        /*if (settings->sublist("Postprocess").get("write solution",true)) {
         stringstream ss;
         ss << j;
         string str = ss.str();
         postproc->writeSolution(F_soln, "sampling_data/outputMC_" + str + "_.exo");
         }*/
        /*
        if (settings->sublist("Postprocess").get<bool>("compute response",false)) {
          Kokkos::View<ScalarT***,HostDevice> currresponse = postproc->computeResponse(0);
          for (size_t i=0; i<currresponse.extent(0); i++) {
            for (size_t j=0; j<currresponse.extent(1); j++) {
              for (size_t k=0; k<currresponse.extent(2); k++) {
                ScalarT myval = currresponse(i,j,k);
                ScalarT gval = 0.0;
                Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&myval,&gval);
                //Comm->SumAll(&myval, &gval, 1);
                currresponse(i,j,k) = gval;
              }
            }
          }
          
          response_values.push_back(currresponse);
          if (settings->sublist("Postprocess").get<bool>("compute response forward gradient",false)) {
            Kokkos::View<ScalarT****,HostDevice> currgrad("current gradient",numstochparams,currresponse.extent(0),
                                                         currresponse.extent(1),currresponse.extent(2));
            for (int i=0; i<numstochparams; i++) {
              ScalarT oldval = currparams[i];
              ScalarT pert = 1.0e-6;
              currparams[i] += pert;
              params->updateParams(currparams,2);
              DFAD objfun2 = 0.0;
              solve->forwardModel(objfun2);
              Kokkos::View<ScalarT***,HostDevice> currresponse2 = postproc->computeResponse(0);
              for (size_t i2=0; i2<currresponse2.extent(0); i2++) {
                for (size_t j=0; j<currresponse2.extent(1); j++) {
                  for (size_t k=0; k<currresponse2.extent(2); k++) {
                    ScalarT myval = currresponse2(i2,j,k);
                    ScalarT gval = 0.0;
                    Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&myval,&gval);
                    //Comm->SumAll(&myval, &gval, 1);
                    currgrad(i,i2,j,k) = (gval-currresponse(i2,j,k))/pert;
                  }
                }
              }
              //if (Comm->getRank() == 0) {
              //  cout << "Estimated derivative wrt stoch. param: " << i << endl;
              //  cout << "                                     : " << (currresponse2(0,0,0)-currresponse(0,0,0))/1.0e-6 << endl;
              //}
              currparams[i] = oldval;
            }
            response_grads.push_back(currgrad);
          }
        }*/
        if (Comm->getRank() == 0 && j%output_freq == 0) {
          cout << "Finished evaluating sample number: " << j+1 << " out of " << numsamples << endl;
        }
      }
      
    }
    
    if (Comm->getRank() == 0) {
      string sptname = "sample_points.dat";
      ofstream sampOUT(sptname.c_str());
      sampOUT.precision(6);
      for (int r=0; r<samplepts.extent(0); r++) {
        for (int d=0; d<samplepts.extent(1); d++) {
          sampOUT << samplepts(r,d) << "  ";
        }
        sampOUT << endl;
      }
      sampOUT.close();
      
      string sname = "sample_data.dat";
      ofstream respOUT(sname.c_str());
      respOUT.precision(6);
      for (int r=0; r<response_values.size(); r++) {
        for (int s=0; s<response_values[r].extent(0); s++) { // sensor index
          for (int t=0; t<response_values[r].extent(2); t++) { // time index
            for (int d=0; d<response_values[r].extent(1); d++) { // data index
              respOUT << response_values[r](s,d,t) << "  ";
            }
          }
        }
        respOUT << endl;
      }
      respOUT.close();
      
      if (settings->sublist("Postprocess").get<bool>("compute response forward gradient",false)) {
        string sname = "sample_grads.dat";
        ofstream gradOUT(sname.c_str());
        gradOUT.precision(6);
        for (int r=0; r<response_grads.size(); r++) {
          for (int s=0; s<response_grads[r].extent(0); s++) { // sensor index
            for (int t=0; t<response_grads[r].extent(2); t++) { // time index
              for (int d=0; d<response_grads[r].extent(1); d++) { // data index
                for (int p=0; d<response_grads[r].extent(1); d++) { // data index
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
    
    // Turn off visualization while optimizing
    bool postproc_plot = postproc->write_solution;
    postproc->write_solution = false;
    
    Teuchos::RCP<std::ostream> outStream;
    outStream = Teuchos::rcp(&std::cout, false);
    // Generate data and get objective
    obj = Teuchos::rcp( new ROL::Objective_MILO<RealT> (solve, postproc, params));
    
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
      bool use_scale = ROLsettings.get("Use Scaling For Epsilon-Active Sets",false);
      RealT scale;
      if(use_scale){
        RealT tol = 1.e-12; //should probably be read in, though we're not using inexact gradients yet anyways...
        Teuchos::RCP<vector<RealT> > g0_rcp = Teuchos::rcp( new vector<RealT> (numParams, 0.0) );
        ROL::StdVector<RealT> g0p(g0_rcp);
        (*obj).gradient(g0p,x,tol);
        scale = 1.0e-2/g0p.norm();
      }
      else {
        scale = 1.0;
      }
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
    
    //////////////////////////////////////////////////////
    // Verification tests
    //////////////////////////////////////////////////////
    
    // Recovering a data-generating solution
    if (ROLsettings.sublist("General").get("Generate data",false)) {
      std::cout << "Generating data ... " << std::endl;
      DFAD objfun = 0.0;
      if (params->isParameter("datagen")) {
        vector<double> pval = {1.0};
        params->setParam(pval,"datagen");
      }
      solve->response_type = "none";
      solve->forwardModel(objfun);
      std::cout << "Storing data ... " << std::endl;
      
      vector<vector<ScalarT> > times = solve->soln->times;
      vector<vector<Teuchos::RCP<LA_MultiVector> > > data = solve->soln->data;
      for (size_t i=0; i<times.size(); i++) {
        for (size_t j=0; j<times[i].size(); j++) {
          solve->datagen_soln->store(data[i][j], times[i][j], i);
        }
      }
      std::cout << "Finished storing data" << std::endl;
      if (params->isParameter("datagen")) {
        vector<double> pval = {0.0};
        params->setParam(pval,"datagen");
      }
      solve->response_type = "discrete";
      std::cout << "Finished generating data for inversion " << std::endl;
    }
    
    // Comparing a gradient/Hessian with finite difference approximation
    if(ROLsettings.sublist("General").get("Do grad+hessvec check",true)){
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
          (*d_rcp)[i] = 10.0*(ScalarT)rand()/(ScalarT)RAND_MAX - 5.0;
        }
      }
      ROL::StdVector<RealT> d(d_rcp);
      // check gradient and Hessian-vector computation using finite differences
      (*obj).checkGradient(x, d, (Comm->getRank() == 0));
      //(*obj).checkHessVec(x, d, true); //Hessian-vector is already done with FD.
      
    }
    
    
    Teuchos::Time timer("Optimization Time",true);
    
    // Run algorithm.
    vector<std::string> output;
    if(bound_vars)
    output = algo.run(x, *obj, *con, (Comm->getRank() == 0 )); //only processor of rank 0 print outs
    else
    output = algo.run(x, *obj, (Comm->getRank() == 0)); //only processor of rank 0 prints out
    
    ScalarT optTime = timer.stop();
    if (Comm->getRank() == 0 ) {
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
      obj->printHess(settings->sublist("Postprocess").get("Hessian Output File","hess.dat"),x,Comm->getRank());
    }
    if (settings->sublist("Analysis").get("Write Output",false)) {
      DFAD val = 0.0;
      solve->forwardModel(val);
      //postproc->writeSolution(settings->sublist("Postprocess").get<string>("Output File","output"));
    }
    
    if (postproc_plot) {
      postproc->write_solution = true;
      DFAD objfun = 0.0;
      solve->forwardModel(objfun);
    }
  } //ROL
  else if (analysis_type == "ROL_SIMOPT") {
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
    RealT aktol    = ROLsettings.sublist("General").sublist("Krylov").get("Absolute Tolerance",1e-4);
    RealT rktol    = ROLsettings.sublist("General").sublist("Krylov").get("Relative Tolerance",1e-2);
    int maxKiter   = ROLsettings.sublist("General").sublist("Krylov").get("Iteration Limit",100);
    
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
      bool use_scale = ROLsettings.get("Use Scaling For Epsilon-Active Sets",false);
      RealT scale;
      if(use_scale){
        RealT tol = 1.e-12; //should probably be read in, though we're not using inexact gradients yet anyways...
        Teuchos::RCP<vector<RealT> > g0_rcp = Teuchos::rcp( new vector<RealT> (numParams, 0.0) );
        ROL::StdVector<RealT> g0p(g0_rcp);
        (*obj).gradient(g0p,x,tol);
        scale = 1.0e-2/g0p.norm();
      }
      else {
        scale = 1.0;
      }
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
        for ( unsigned i = 0; i < numParams; i++ ) {
          (*d_rcp)[i] = 1.0;
        }
      }
      else {
        for ( unsigned i = 0; i < numParams; i++ ) {
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
      obj->printHess(settings->sublist("Postprocess").get("Hessian Output File","hess.dat"),x,Comm->getRank());
    }
    if (settings->sublist("Analysis").get("Write Output",false)) {
      DFAD val = 0.0;
      solve->forwardModel(val);
      //postproc->writeSolution(settings->sublist("Postprocess").get<string>("Output File","output"));
    }
  } //ROL_SIMOPT
  else { // don't solve anything, but produce visualization
    //if (settings->sublist("Postprocess").get("write solution",true))
    //postproc->writeSolution(settings->sublist("Postprocess").get<string>("Output File","output"));
    
  }
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished analysis::run" << endl;
    }
  }
  
}
