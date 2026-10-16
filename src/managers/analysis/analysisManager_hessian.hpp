/***********************************************************************
Extracts the reduced Hessian, either column by column through hessVec or by handing
MrHyDE_HessianOperator to Trilinos.

 Questions? Contact Alexey Voronin
 ************************************************************************/

// ========================================================================================
// ========================================================================================

void AnalysisManager::hessianSolve() {

  TEUCHOS_TEST_FOR_EXCEPTION(settings_->get<bool>("enable memory purge",false),
                             std::runtime_error,
                             "Error: analysis type 'hessian' requires 'enable memory purge: false' "
                             "(the control column filter needs the per-block element lists).");

  Teuchos::ParameterList & hset = settings_->sublist("Analysis").sublist("Hessian");

  const string  filebase  = hset.get<string>("output file","hess");
  const int     first_col = hset.get<int>("first column",0);
  const int     max_col   = hset.get<int>("max columns",8000);
  const ScalarT fd_step   = block_prec::scalarParam(hset,"FD step",1.0e-7);
  const bool    ctrl_only = hset.get<bool>("control columns only",false);
  const string  ctrl_list = hset.get<string>("control blocks","");
  const bool    probe_set = hset.isParameter("probe mode");
  const string  probe     = hset.get<string>("probe mode","columns");
  const int     probing   = hset.get<int>("probing size",10);
  const int     neig      = hset.get<int>("eigenvalues",0);
  const string  metric    = hset.get<string>("metric","euclidean");

  TEUCHOS_TEST_FOR_EXCEPTION(probe != "columns" && probe != "operator",std::runtime_error,
                             "Error: Analysis->Hessian->'probe mode' must be columns or operator.");
  TEUCHOS_TEST_FOR_EXCEPTION(metric != "euclidean" && metric != "riesz",std::runtime_error,
                             "Error: Analysis->Hessian->'metric' must be euclidean or riesz.");
  TEUCHOS_TEST_FOR_EXCEPTION(neig > 0 && ((probe_set && probe == "columns") || first_col != 0
                                          || hset.get<int>("number of columns",0) > 0),
                             std::runtime_error,
                             "Error: Analysis->Hessian->eigenvalues > 0 skips the column dump; remove "
                             "'probe mode: columns', 'first column' and 'number of columns', or set eigenvalues to 0.");
  TEUCHOS_TEST_FOR_EXCEPTION(metric == "riesz" && probe == "columns" && neig <= 0,std::runtime_error,
                             "Error: Analysis->Hessian->metric = riesz applies only to probe mode operator "
                             "or eigenvalues > 0; the column dump writes H itself.");
  TEUCHOS_TEST_FOR_EXCEPTION(metric == "riesz" && params_->paramRiesz.is_null(),
                             std::runtime_error,
                             "Error: Analysis->Hessian->metric = riesz needs Solver->'Parameter "
                             "Riesz map'->apply = exact so P^-1 is available.");
  for (size_t r=0; r<postproc_->objectives.size(); ++r) {
    TEUCHOS_TEST_FOR_EXCEPTION(postproc_->objectives[r].type == "discrete control",std::runtime_error,
                               "Error: analysis type 'hessian' cannot evaluate the 'discrete control' objective '"
                               + postproc_->objectives[r].name
                               + "'; its data come from the ROL/ROL2 'Generate data' step, which hessian does not run.");
  }

  const int rank = comm_->getRank();
  const bool exact = params_->isParameter("src_gate");
  const string prefix = filebase + (exact ? "_exact" : "_fd");
  const bool use_operator = (probe == "operator" || neig > 0);

  postproc_->write_solution = false;

  Teuchos::RCP<ROL::Objective_MILO<ScalarT> > obj
    = Teuchos::rcp(new ROL::Objective_MILO<ScalarT>(solver_,postproc_,params_));

  MrHyDE_OptVector xtmp = params_->getCurrentVector();
  Teuchos::RCP<ROL::Vector<ScalarT> > x = xtmp.clone();
  x->set(xtmp);

  std::vector<ROL::Ptr<ROL::TpetraMultiVector<ScalarT> > > xfield
    = Teuchos::dyn_cast<MrHyDE_OptVector>(*x).getField();

  TEUCHOS_TEST_FOR_EXCEPTION(xfield.size() == 0, std::runtime_error,
                             "Error: analysis type 'hessian' needs a discretized parameter.");

  const int nslabs = static_cast<int>(xfield.size());
  const int nspatial = static_cast<int>(params_->param_owned_map->getGlobalNumElements());
  const int ctrl_dim = nslabs*nspatial;

  TEUCHOS_TEST_FOR_EXCEPTION(params_->param_owned_map->getMinAllGlobalIndex() != 0 ||
                             params_->param_owned_map->getMaxAllGlobalIndex() != static_cast<GO>(nspatial)-1,
                             std::runtime_error,
                             "Error: analysis type 'hessian' needs parameter global ids dense in [0,n); "
                             "basis(j) addresses a parameter dof by its global id.");
  TEUCHOS_TEST_FOR_EXCEPTION(probe == "operator" && ctrl_dim > max_col,std::runtime_error,
                             "Error: probe mode operator probes all " + std::to_string(ctrl_dim)
                             + " Hessian columns, which exceeds Analysis->Hessian->'max columns' ("
                             + std::to_string(max_col)
                             + "). Raise the cap, or use probe mode columns with 'number of columns'.");

  std::vector<int> ctrl_mask;
  if (ctrl_only && !use_operator) {
    ctrl_mask = this->controlDofMask(ctrl_list, nspatial);
  }

  std::vector<int> cols;
  int num_col = 0;

  if (!use_operator) {

    num_col = hset.get<int>("number of columns",0);
    if (num_col <= 0) { num_col = ctrl_dim - first_col; }
    num_col = std::min(num_col, ctrl_dim - first_col);

    // Columns whose DoF touches no control block are identically zero, so skip them.
    for (int c=0; c<num_col; ++c) {
      const int j = first_col + c;
      if (ctrl_only && ctrl_mask[j % nspatial] == 0) { continue; }
      cols.push_back(j);
    }

    TEUCHOS_TEST_FOR_EXCEPTION(static_cast<int>(cols.size()) > max_col, std::runtime_error,
                               "Error: Hessian column count " + std::to_string(cols.size())
                               + " (dimension " + std::to_string(ctrl_dim)
                               + ") exceeds Analysis->Hessian->'max columns' ("
                               + std::to_string(max_col)
                               + "). Set 'number of columns' to extract a sub-block, or raise the cap.");
  }

  const int ncols = use_operator ? ctrl_dim : static_cast<int>(cols.size());
  const int sweeps_per_col = exact ? 2 : 4;

  if (rank == 0) {
    printSectionBanner(rank,"Hessian column dump");
    std::cout << "[HESS-DUMP] path = " << (exact ? "exact (src_gate declared)" : "fd (no src_gate)")
              << std::endl;
    std::cout << "[HESS-DUMP] dim = " << ctrl_dim << " (slabs = " << nslabs
              << ", spatial = " << nspatial << ")" << std::endl;
    if (probe == "operator" || !use_operator) {
      std::cout << "[HESS-DUMP] " << ncols << " columns, " << sweeps_per_col
                << " sweeps/column -> " << ncols*sweeps_per_col << " transient sweeps";
      if (probe == "operator") { std::cout << ", probed in blocks of " << probing; }
      std::cout << std::endl;
    }
    if (neig > 0) {
      std::cout << "[HESS-DUMP] " << neig << " eigenpairs of "
                << (metric == "riesz" ? "inv(P)*H" : "H") << " via Anasazi" << std::endl;
    }
    if (ctrl_only && !use_operator) {
      std::cout << "[HESS-DUMP] control columns only: " << ncols << " of " << num_col
                << " columns carry support" << std::endl;
    }
    if (!exact) {
      std::cout << "[HESS-DUMP] FD step = " << fd_step << std::endl;
    }
  }

  // Pin the stored forward trajectory at x so the column loop does not re-solve it.
  {
    ScalarT tol = fd_step;
    obj->value(*x,tol);
  }

  typedef Tpetra::MatrixMarket::Writer<LA_MultiVector> MMWriter;

  if (use_operator) {

    Teuchos::RCP<MrHyDE_ParamRiesz> rz;
    if (metric == "riesz") { rz = params_->paramRiesz; }

    Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,SolverNode> > Hop
      = Teuchos::rcp(new MrHyDE_HessianOperator(obj,x,params_->param_owned_map,
                                                fd_step,rz));

    if (probe == "operator") {
      Teuchos::ParameterList wp;
      wp.set("probing size",probing);
      wp.set("precision",16);
      MMWriter::writeOperator(prefix+"_op.mtx",*Hop,wp);
      if (rank == 0) {
        std::cout << "[HESS-DUMP] wrote " << prefix
                  << "_op.mtx (coordinate format, true global ids)" << std::endl;
      }
    }

    if (neig > 0) {
      this->hessianEigenvalues(hset,Hop,prefix,neig,ctrl_dim,rz.is_null() && exact);
    }

    postproc_->resetErrors();
    return;
  }

  // One row block per slab; the writer does the gather and the formatting.
  std::vector<Teuchos::RCP<LA_MultiVector> > Hblocks;
  for (int t=0; t<nslabs; ++t) {
    Hblocks.push_back(solver_->linalg->getNewParamVector(ncols));
  }

  Teuchos::RCP<ROL::Vector<ScalarT> > hv = x->clone();
  std::vector<ROL::Ptr<ROL::TpetraMultiVector<ScalarT> > > hvf
    = Teuchos::dyn_cast<MrHyDE_OptVector>(*hv).getField();

  for (int c=0; c<ncols; ++c) {

    const int j = cols[c];
    ROL::Ptr<ROL::Vector<ScalarT> > ej = x->basis(j);

    ScalarT tol = fd_step;
    obj->hessVec(*hv,*ej,*x,tol);

    for (int t=0; t<nslabs; ++t) {
      Hblocks[t]->getVectorNonConst(c)->update(1.0,*(hvf[t]->getVector()),0.0);
    }

    if (rank == 0 && ncols > 20 && (c+1)%(ncols/20) == 0) {
      std::cout << "[HESS-DUMP] column " << c+1 << " of " << ncols << std::endl;
    }
  }

  // Row order follows the gather map, not global id, so the map file is what recovers it.
  for (int t=0; t<nslabs; ++t) {
    std::stringstream fn, note;
    fn << prefix << "_slab" << t << ".mtx";
    note << (exact ? "exact" : "fd") << " hessVec, slab " << t << " row block, "
         << "row i = line i of " << prefix << ".map, "
         << "column j = line j of " << prefix << ".cols";
    MMWriter::writeDenseFile(fn.str(),*(Hblocks[t]),"reduced Hessian",note.str());
  }
  MMWriter::writeMapFile(prefix+".map",*(params_->param_owned_map));

  int cols_open = 1;
  if (rank == 0) {

    std::cout << "[HESS-DUMP] wrote " << nslabs << " row blocks " << prefix
              << "_slab*.mtx (" << nspatial << " x " << ncols << " each), "
              << prefix << ".map and " << prefix << ".cols" << std::endl;

    std::ofstream colout(prefix+".cols");
    cols_open = colout.is_open() ? 1 : 0;
    for (int c=0; c<ncols; ++c) { colout << cols[c] << std::endl; }
    colout.close();
  }
  Teuchos::broadcast(*comm_,0,1,&cols_open);
  TEUCHOS_TEST_FOR_EXCEPTION(cols_open == 0,std::runtime_error,
                             "Error: could not open the data file: " + prefix + ".cols");

  postproc_->resetErrors();
}

// ========================================================================================
// ========================================================================================

void AnalysisManager::hessianEigenvalues(Teuchos::ParameterList & hset,
                                         const Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,SolverNode> > & Hop,
                                         const string & prefix, const int & neig,
                                         const int & ctrl_dim, const bool & hermitian) {

  typedef Tpetra::Operator<ScalarT,LO,GO,SolverNode> EigOp;
  typedef Anasazi::BasicEigenproblem<ScalarT,LA_MultiVector,EigOp> EigProblem;
  typedef Anasazi::BlockKrylovSchurSolMgr<ScalarT,LA_MultiVector,EigOp> EigSolver;
  typedef Tpetra::MatrixMarket::Writer<LA_MultiVector> MMWriter;

  // Which and Block Size keep Anasazi's defaults.
  Teuchos::ParameterList def;
  def.set("Num Blocks",std::min(ctrl_dim-1,std::max(2*neig,10)));
  def.set("Maximum Restarts",100);
  def.set("Convergence Tolerance",1.0e-8);
  def.set("Verbosity",Anasazi::Errors+Anasazi::Warnings+Anasazi::FinalSummary);

  Teuchos::ParameterList ap = hset.sublist("Anasazi");
  ap.setParametersNotAlreadySet(def);

  Teuchos::RCP<LA_MultiVector> ivec
    = Teuchos::rcp(new LA_MultiVector(Hop->getDomainMap(),ap.get("Block Size",1)));
  ivec->randomize();

  Teuchos::RCP<EigProblem> prob = Teuchos::rcp(new EigProblem(Hop,ivec));
  prob->setHermitian(hermitian);
  prob->setNEV(neig);
  TEUCHOS_TEST_FOR_EXCEPTION(!prob->setProblem(),std::runtime_error,
                             "Error: Anasazi rejected the Hessian eigenproblem.");

  EigSolver mgr(prob,ap);
  const Anasazi::ReturnType eret = mgr.solve();
  Anasazi::Eigensolution<ScalarT,LA_MultiVector> sol = prob->getSolution();

  if (comm_->getRank() == 0 && eret != Anasazi::Converged) {
    std::cout << "[HESS-DUMP] WARNING: Anasazi did not converge; "
              << sol.numVecs << " of " << neig << " eigenpairs returned" << std::endl;
  }

  if (sol.numVecs > 0 && !sol.Evecs.is_null()) {
    MMWriter::writeDenseFile(prefix+"_evec.mtx",*(sol.Evecs),"reduced Hessian eigenvectors",
                             "row i = line i of " + prefix + "_control.map");
    MMWriter::writeMapFile(prefix+"_control.map",*(Hop->getDomainMap()));
  }
}

// ========================================================================================
// ========================================================================================

std::vector<int> AnalysisManager::controlDofMask(const string & control_blocks, const size_t nspatial) {

  Teuchos::RCP<DiscretizationInterface> disc = postproc_->disc;
  const size_t nblocks = disc->block_names.size();

  // A regularization can miss blocks where the control acts, so the deck names them.
  const std::vector<bool> is_ctrl = matchNamedBlocks(control_blocks,disc->block_names,
                                                     "Error: Analysis->Hessian->'control blocks'");
  TEUCHOS_TEST_FOR_EXCEPTION(std::find(is_ctrl.begin(),is_ctrl.end(),true) == is_ctrl.end(),
                             std::runtime_error,
                             "Error: Analysis->Hessian->'control columns only' needs 'control blocks' "
                             "to name the element blocks the control acts on.");

  std::vector<int> local(nspatial,0), mask(nspatial,0);
  for (size_t b=0; b<nblocks; ++b) {
    if (!is_ctrl[b]) { continue; }
    for (size_t e=0; e<disc->my_elements[b].extent(0); ++e) {
      std::vector<GO> gids;
      params_->paramDOF->getElementGIDs(disc->my_elements[b](e),gids,disc->block_names[b]);
      for (size_t i=0; i<gids.size(); ++i) { local[gids[i]] = 1; }
    }
  }
  Teuchos::reduceAll(*comm_,Teuchos::REDUCE_MAX,static_cast<int>(nspatial),local.data(),mask.data());
  return mask;
}
