/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

// ========================================================================================
// ========================================================================================

template<class Node>
void SolverManager<Node>::completeSetup() {

  debugger->print("**** Starting SolverManager::completeSetup()");
  
  /////////////////////////////////////////////////////////////////////////////
  // Create linear algebra interface
  /////////////////////////////////////////////////////////////////////////////
  
  linalg = Teuchos::rcp( new LinearAlgebraInterface<Node>(Comm, settings, disc, params) );
  
  if (store_vectors) {
    for (size_t set=0; set<setnames.size(); ++set) {
      res.push_back(linalg->getNewVector(set));
      res_over.push_back(linalg->getNewOverlappedVector(set));
      du_over.push_back(linalg->getNewOverlappedVector(set));
      du.push_back(linalg->getNewVector(set));
    }
  }
  this->setupFixedDOFs(settings);

  //---------------------------------------------------
  // Mass matrix (lumped and maybe full) for explicit
  //---------------------------------------------------
  
  if (fully_explicit) {
    this->setupExplicitMass();
  }
  
  if (use_param_mass && params->num_discretized_params > 0) {
    this->setupDiscretizedParamMass();
  }

  auto needs_refmax_aux = [](const Teuchos::RCP<LinearSolverContext<Node> > & cntxt) -> bool {
    if (cntxt.is_null()) return false;
    const bool use_block_tri = (cntxt->prec_type == "block triangular");
    const bool use_block_diag = (cntxt->prec_type == "block diagonal");
    const std::string & pivot_prec = cntxt->schur.pivot_block_preconditioner_type;
    const bool use_refmaxwell =
      (pivot_prec == "RefMaxwell" || pivot_prec == "refmaxwell" || pivot_prec == "REFMAXWELL");
    std::string schur_prec = cntxt->schur.schur_block_preconditioner_type;
    for (size_t i = 0; i < schur_prec.size(); ++i) {
      schur_prec[i] = static_cast<char>(std::toupper(static_cast<unsigned char>(schur_prec[i])));
    }
    const bool use_ads_schur = (schur_prec == "ADS");
    const bool use_refmaxwell_schur = (schur_prec == "REFMAXWELL");
    return (use_block_tri && (use_refmaxwell || use_ads_schur || use_refmaxwell_schur)) ||
           (use_block_diag && use_refmaxwell);
  };

  // Build and share RefMaxwell/ADS auxiliary data once per set across all compatible contexts.
  for (size_t set = 0; set < setnames.size(); ++set) {
    Teuchos::RCP<LinearSolverContext<Node> > src = Teuchos::null;
    if (set < linalg->context.size() && needs_refmax_aux(linalg->context[set])) {
      src = linalg->context[set];
    }
    else if (set < linalg->context_L2.size() && needs_refmax_aux(linalg->context_L2[set])) {
      src = linalg->context_L2[set];
    }
    else if (set < linalg->context_BndryL2.size() && needs_refmax_aux(linalg->context_BndryL2[set])) {
      src = linalg->context_BndryL2[set];
    }
    if (src.is_null()) continue;

    this->setupBlockTriangularAuxiliary(set, src);

    if (set < linalg->context.size() && !linalg->context[set].is_null()) {
      linalg->context[set]->refMaxwell.D0_matrix = src->refMaxwell.D0_matrix;
      linalg->context[set]->refMaxwell.D1_matrix = src->refMaxwell.D1_matrix;
      linalg->context[set]->refMaxwell.M1_matrix = src->refMaxwell.M1_matrix;
      linalg->context[set]->refMaxwell.M2_matrix = src->refMaxwell.M2_matrix;
      linalg->context[set]->refMaxwell.nodal_coords = src->refMaxwell.nodal_coords;
      linalg->context[set]->refMaxwell.nullspace = src->refMaxwell.nullspace;
      linalg->context[set]->refMaxwell.ads_null11 = src->refMaxwell.ads_null11;
      linalg->context[set]->refMaxwell.ads_null22 = src->refMaxwell.ads_null22;
    }
    if (set < linalg->context_L2.size() && !linalg->context_L2[set].is_null()) {
      linalg->context_L2[set]->refMaxwell.D0_matrix = src->refMaxwell.D0_matrix;
      linalg->context_L2[set]->refMaxwell.D1_matrix = src->refMaxwell.D1_matrix;
      linalg->context_L2[set]->refMaxwell.M1_matrix = src->refMaxwell.M1_matrix;
      linalg->context_L2[set]->refMaxwell.M2_matrix = src->refMaxwell.M2_matrix;
      linalg->context_L2[set]->refMaxwell.nodal_coords = src->refMaxwell.nodal_coords;
      linalg->context_L2[set]->refMaxwell.nullspace = src->refMaxwell.nullspace;
      linalg->context_L2[set]->refMaxwell.ads_null11 = src->refMaxwell.ads_null11;
      linalg->context_L2[set]->refMaxwell.ads_null22 = src->refMaxwell.ads_null22;
    }
    if (set < linalg->context_BndryL2.size() && !linalg->context_BndryL2[set].is_null()) {
      linalg->context_BndryL2[set]->refMaxwell.D0_matrix = src->refMaxwell.D0_matrix;
      linalg->context_BndryL2[set]->refMaxwell.D1_matrix = src->refMaxwell.D1_matrix;
      linalg->context_BndryL2[set]->refMaxwell.M1_matrix = src->refMaxwell.M1_matrix;
      linalg->context_BndryL2[set]->refMaxwell.M2_matrix = src->refMaxwell.M2_matrix;
      linalg->context_BndryL2[set]->refMaxwell.nodal_coords = src->refMaxwell.nodal_coords;
      linalg->context_BndryL2[set]->refMaxwell.nullspace = src->refMaxwell.nullspace;
      linalg->context_BndryL2[set]->refMaxwell.ads_null11 = src->refMaxwell.ads_null11;
      linalg->context_BndryL2[set]->refMaxwell.ads_null22 = src->refMaxwell.ads_null22;
    }
  }
  
  debugger->print("**** Finished SolverManager::completeSetup()");
  
}

// ========================================================================================
// ========================================================================================

// Build D0 (grad), M1 (edge mass), M2 (optional), nodal coords, and optionally D1 (curl for ADS)
// for block-triangular/block-diagonal RefMaxwell. Data is stored in cntxt->refMaxwell and shared
// across all linear solver contexts for this set in completeSetup().
template<class Node>
void SolverManager<Node>::setupBlockTriangularAuxiliary(const size_t & set,
                                                       const Teuchos::RCP<LinearSolverContext<Node> > & cntxt) {
  TEUCHOS_TEST_FOR_EXCEPTION(cntxt.is_null(), std::runtime_error,
    "Missing linear solver context for set " + std::to_string(set));
  debugger->print("**** setupBlockTriangularAuxiliary: begin set " + std::to_string(set));

  // RefMaxwell/ADS need M1 (H(curl) mass) with unit weights; otherwise use physics mass weights.
  const bool pivotHasRefMaxwell =
    (cntxt->pivot_block_sublist.name() != "empty") &&
    cntxt->pivot_block_sublist.isSublist("RefMaxwell Settings");
  const bool schurHasRefMaxwell =
    (cntxt->schur_block_sublist.name() != "empty") &&
    cntxt->schur_block_sublist.isSublist("RefMaxwell Settings");
  std::string schur_prec = cntxt->schur.schur_block_preconditioner_type;
  for (size_t i = 0; i < schur_prec.size(); ++i)
    schur_prec[i] = static_cast<char>(std::toupper(static_cast<unsigned char>(schur_prec[i])));
  const bool use_ads_schur = (schur_prec == "ADS");
  const bool use_unit_mass = pivotHasRefMaxwell || schurHasRefMaxwell || use_ads_schur;

  // Assemble full H(curl) mass matrix M1 (overlapped then exported). Used for RefMaxwell edge block.
  matrix_RCP M1_over = linalg->getNewOverlappedMatrix(set);
  vector_RCP diagM1_over = linalg->getNewOverlappedVector(set);
  assembler->updatePhysicsSet(set);
  assembler->getWeightedMass(set, M1_over, diagM1_over, use_unit_mass);

  matrix_RCP M1_full = linalg->getNewMatrix(set);
  linalg->exportMatrixFromOverlapped(set, M1_full, M1_over);
  linalg->fillComplete(M1_full);

  // One map per variable block (same as block_prec). Identifies which block is edge (HCURL) for M1/D0.
  std::vector<Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > > blockMaps = linalg->buildBlockMaps(set);
  TEUCHOS_TEST_FOR_EXCEPTION(blockMaps.empty(), std::runtime_error,
    "Block-triangular auxiliary setup requires at least one block map.");
  const int pivotBlock = cntxt->schur.pivot_block;
  TEUCHOS_TEST_FOR_EXCEPTION(pivotBlock < 0 || static_cast<size_t>(pivotBlock) >= blockMaps.size(),
    std::runtime_error,
    "Schur pivot block index " + std::to_string(pivotBlock) + " is out of range for set " +
    std::to_string(set) + " with " + std::to_string(blockMaps.size()) + " blocks.");

  // Basis names for auxiliary spaces: prefer RefMaxwell Settings, else top-level prec/schur list.
  const Teuchos::ParameterList * refmaxwellSetupListPtr = nullptr;
  if (pivotHasRefMaxwell)
    refmaxwellSetupListPtr = &cntxt->pivot_block_sublist.sublist("RefMaxwell Settings");
  else if (schurHasRefMaxwell)
    refmaxwellSetupListPtr = &cntxt->schur_block_sublist.sublist("RefMaxwell Settings");
  else {
    if (cntxt->prec_sublist.name() != "empty" && cntxt->prec_sublist.isParameter("hgrad basis name"))
      refmaxwellSetupListPtr = &cntxt->prec_sublist;
    else if (cntxt->schur_block_sublist.name() != "empty" && cntxt->schur_block_sublist.isParameter("hgrad basis name"))
      refmaxwellSetupListPtr = &cntxt->schur_block_sublist;
  }
  TEUCHOS_TEST_FOR_EXCEPTION(refmaxwellSetupListPtr == nullptr || !refmaxwellSetupListPtr->isParameter("hgrad basis name"),
    std::runtime_error,
    "Block-triangular/block-diagonal RefMaxwell/ADS auxiliary requires 'hgrad basis name' and 'hcurl basis name' in "
    "Pivot Block Settings->RefMaxwell Settings, Schur Block Settings->RefMaxwell Settings, "
    "Preconditioner Settings, or at the top level of Schur Block Settings.");
  TEUCHOS_TEST_FOR_EXCEPTION(!refmaxwellSetupListPtr->isParameter("hcurl basis name"), std::runtime_error,
    "Block-triangular auxiliary requires 'hcurl basis name' in the same list as 'hgrad basis name'.");
  const Teuchos::ParameterList & refmaxwellSetupList = *refmaxwellSetupListPtr;
  const std::string hgrad_basis = refmaxwellSetupList.template get<std::string>("hgrad basis name");
  const std::string hcurl_basis = refmaxwellSetupList.template get<std::string>("hcurl basis name");
  const int hgrad_order = refmaxwellSetupList.isParameter("hgrad basis order")
    ? refmaxwellSetupList.template get<int>("hgrad basis order")
    : 1;
  const int hcurl_order = refmaxwellSetupList.isParameter("hcurl basis order")
    ? refmaxwellSetupList.template get<int>("hcurl basis order")
    : 1;

  Teuchos::RCP<panzer::ConnManager> conn = mesh->getSTKConnManager();

  // Panzer DOF managers for auxiliary H(grad) and H(curl) on the mesh (used to build D0).
  Teuchos::RCP<panzer::DOFManager> hgrad_dof = Teuchos::rcp(new panzer::DOFManager());
  hgrad_dof->setConnManager(conn, *(Comm->getRawMpiComm()));
  hgrad_dof->setOrientationsRequired(false);

  Teuchos::RCP<panzer::DOFManager> hcurl_dof = Teuchos::rcp(new panzer::DOFManager());
  hcurl_dof->setConnManager(conn, *(Comm->getRawMpiComm()));
  hcurl_dof->setOrientationsRequired(true);

  for (size_t block = 0; block < mesh->block_names.size(); ++block) {
    std::string block_name = mesh->block_names[block];
    topo_RCP cellTopo = mesh->getCellTopology(block_name);

    basis_RCP hgrad_basis_ptr = disc->getBasis(dimension, cellTopo, "HGRAD", hgrad_order);
    Teuchos::RCP<const panzer::Intrepid2FieldPattern> hgrad_pattern =
      Teuchos::rcp(new panzer::Intrepid2FieldPattern(hgrad_basis_ptr));
    hgrad_dof->addField(block_name, hgrad_basis, hgrad_pattern, panzer::FieldType::CG);

    basis_RCP hcurl_basis_ptr = disc->getBasis(dimension, cellTopo, "HCURL", hcurl_order);
    Teuchos::RCP<const panzer::Intrepid2FieldPattern> hcurl_pattern =
      Teuchos::rcp(new panzer::Intrepid2FieldPattern(hcurl_basis_ptr));
    hcurl_dof->addField(block_name, hcurl_basis, hcurl_pattern, panzer::FieldType::CG);
  }

  hgrad_dof->buildGlobalUnknowns();
  hcurl_dof->buildGlobalUnknowns();

  // D0 = gradient: nodal (Hgrad) -> edge (Hcurl). RefMaxwell uses it for the auxiliary space.
  Teuchos::RCP<Thyra::LinearOpBase<ScalarT> > D0_thyra =
    panzer::buildInterpolation(conn, hgrad_dof, hcurl_dof,
                               hgrad_basis, hcurl_basis,
                               Intrepid2::OPERATOR_GRAD,
                               1000, false, true, false);
  auto D0_tpetra = Thyra::TpetraOperatorVectorExtraction<ScalarT,LO,GO,Node>::getTpetraOperator(D0_thyra);
  cntxt->refMaxwell.D0_matrix = Teuchos::rcp_dynamic_cast<LA_CrsMatrix>(D0_tpetra, true);

  // Which block is edge (HCURL)? Match D0 range size to a block map so M1 and D0 use the same ordering.
  const Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > aux_edge_map = cntxt->refMaxwell.D0_matrix->getRangeMap();
  const GO d0_range_size = static_cast<GO>(aux_edge_map->getGlobalNumElements());
  size_t edgeBlock = static_cast<size_t>(pivotBlock);
  for (size_t b = 0; b < blockMaps.size(); ++b) {
    if (static_cast<GO>(blockMaps[b]->getGlobalNumElements()) == d0_range_size) {
      edgeBlock = b;
      break;
    }
  }
  const Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > edge_block_map = blockMaps[edgeBlock];
  const GO edge_block_rows = edge_block_map->getGlobalNumElements();
  const GO d0_rows = aux_edge_map->getGlobalNumElements();
  TEUCHOS_TEST_FOR_EXCEPTION(d0_rows != edge_block_rows, std::runtime_error,
    "D0 size mismatch with edge block: D0 rows=" + std::to_string(d0_rows) +
    ", edge-block rows=" + std::to_string(edge_block_rows));

  // Restrict full mass to edge block: M1 is the H(curl) mass on the edge block for RefMaxwell.
  cntxt->refMaxwell.M1_matrix = linalg->extractDiagonalBlock(M1_full, edge_block_map);

  // If Panzer D0 range map differs from our edge block map, reorder D0 rows to edge_block_map.
  typedef typename LA_CrsMatrix::nonconst_local_inds_host_view_type host_inds_type;
  typedef typename LA_CrsMatrix::nonconst_values_host_view_type host_vals_type;
  const Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > nodal_map = cntxt->refMaxwell.D0_matrix->getDomainMap();
  const Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > d0_col_map = cntxt->refMaxwell.D0_matrix->getColMap();
  if (!aux_edge_map->isSameAs(*edge_block_map)) {
    Teuchos::RCP<LA_CrsMatrix> D0_remapped =
      Teuchos::rcp(new LA_CrsMatrix(edge_block_map, std::max<size_t>(1, cntxt->refMaxwell.D0_matrix->getLocalMaxNumRowEntries())));
    const LO n_aux_rows = aux_edge_map->getLocalNumElements();
    const LO n_target_rows = edge_block_map->getLocalNumElements();
    TEUCHOS_TEST_FOR_EXCEPTION(n_aux_rows != n_target_rows, std::runtime_error,
      "D0 remap: local row counts differ (aux=" + std::to_string(static_cast<long long>(n_aux_rows)) +
      ", target=" + std::to_string(static_cast<long long>(n_target_rows)) + ").");
    for (LO lid = 0; lid < n_aux_rows; ++lid) {
      const GO row_gid = edge_block_map->getGlobalElement(lid);
      size_t nent = cntxt->refMaxwell.D0_matrix->getNumEntriesInLocalRow(lid);
      if (nent == 0) continue;
      host_inds_type col_lids("d0_col_lids", nent);
      host_vals_type row_vals("d0_row_vals", nent);
      cntxt->refMaxwell.D0_matrix->getLocalRowCopy(lid, col_lids, row_vals, nent);
      std::vector<GO> col_gids;
      std::vector<ScalarT> vals;
      col_gids.reserve(nent);
      vals.reserve(nent);
      for (size_t j = 0; j < nent; ++j) {
        const GO col_gid = d0_col_map->getGlobalElement(col_lids(j));
        if (col_gid == Teuchos::OrdinalTraits<GO>::invalid()) continue;
        col_gids.push_back(col_gid);
        vals.push_back(row_vals(j));
      }
      if (!col_gids.empty()) {
        D0_remapped->insertGlobalValues(row_gid, col_gids, vals);
      }
    }
    D0_remapped->fillComplete(nodal_map, edge_block_map);
    cntxt->refMaxwell.D0_matrix = D0_remapped;
  }

  // Nodal coordinates on D0 domain (Hgrad) for RefMaxwell nullspace / mesh info.
  cntxt->refMaxwell.nodal_coords = Teuchos::rcp(
    new Tpetra::MultiVector<typename Teuchos::ScalarTraits<ScalarT>::coordinateType,LO,GO,Node>(nodal_map, dimension));
  auto coords_2d = cntxt->refMaxwell.nodal_coords->getLocalViewHost(Tpetra::Access::OverwriteAll);

  std::map<GO, std::vector<double> > gid_to_coords;
  for (size_t block = 0; block < mesh->block_names.size(); ++block) {
    const std::string block_name = mesh->block_names[block];
    const size_t num_elem = disc->my_elements[block].extent(0);
    vector<size_t> elem_ids(num_elem);
    for (size_t e = 0; e < num_elem; ++e) elem_ids[e] = e;
    DRV elem_nodes = mesh->getMyNodes(block, elem_ids);
    for (size_t e = 0; e < num_elem; ++e) {
      std::vector<GO> elem_dofs;
      LO local_elem_id = disc->my_elements[block](e);
      hgrad_dof->getElementGIDs(local_elem_id, elem_dofs, block_name);
      for (size_t n = 0; n < elem_dofs.size(); ++n) {
        if (gid_to_coords.find(elem_dofs[n]) == gid_to_coords.end()) {
          std::vector<double> coord(dimension, 0.0);
          for (int d = 0; d < dimension; ++d) coord[d] = elem_nodes(e, n, d);
          gid_to_coords[elem_dofs[n]] = coord;
        }
      }
    }
  }

  for (LO lid = 0; lid < static_cast<LO>(nodal_map->getLocalNumElements()); ++lid) {
    GO gid = nodal_map->getGlobalElement(lid);
    auto it = gid_to_coords.find(gid);
    if (it == gid_to_coords.end()) continue;
    for (int d = 0; d < dimension; ++d) {
      coords_2d(lid, d) = it->second[d];
    }
  }

  // ADS (Auxiliary-space Divergence Solver) for Schur block: D1 = curl (Hcurl -> Hdiv), M2 = face mass on target block.
  if (use_ads_schur) {
    typedef typename LA_CrsMatrix::nonconst_local_inds_host_view_type host_inds_type;
    typedef typename LA_CrsMatrix::nonconst_values_host_view_type host_vals_type;
    size_t targetBlock = 0;
    for (size_t b = 0; b < blockMaps.size(); ++b) {
      if (b != static_cast<size_t>(pivotBlock)) {
        targetBlock = b;
        break;
      }
    }
    cntxt->refMaxwell.M2_matrix = linalg->extractDiagonalBlock(M1_full, blockMaps[targetBlock]);

    const bool schurTopHasHdiv = cntxt->schur_block_sublist.isParameter("hdiv basis name");
    const bool schurRefMaxHasHdiv = (cntxt->schur_block_sublist.name() != "empty") &&
      cntxt->schur_block_sublist.isSublist("RefMaxwell Settings") &&
      cntxt->schur_block_sublist.sublist("RefMaxwell Settings").isParameter("hdiv basis name");
    const Teuchos::ParameterList & schurAdsList = schurTopHasHdiv ? cntxt->schur_block_sublist
      : (schurRefMaxHasHdiv ? cntxt->schur_block_sublist.sublist("RefMaxwell Settings") : cntxt->schur_block_sublist);
    TEUCHOS_TEST_FOR_EXCEPTION(!schurAdsList.isParameter("hdiv basis name"), std::runtime_error,
      "ADS Schur requires 'hdiv basis name' in Schur Block Settings or Schur Block Settings->RefMaxwell Settings.");
    const std::string hdiv_basis = schurAdsList.template get<std::string>("hdiv basis name");
    const int hdiv_order = schurAdsList.isParameter("hdiv basis order")
      ? schurAdsList.template get<int>("hdiv basis order") : 1;

    // H(div) DOFs for D1 (curl); then build D1 and remap its range/domain to block maps.
    Teuchos::RCP<panzer::DOFManager> hdiv_dof = Teuchos::rcp(new panzer::DOFManager());
    hdiv_dof->setConnManager(conn, *(Comm->getRawMpiComm()));
    hdiv_dof->setOrientationsRequired(true);
    for (size_t block = 0; block < mesh->block_names.size(); ++block) {
      std::string block_name = mesh->block_names[block];
      topo_RCP cellTopo = mesh->getCellTopology(block_name);
      basis_RCP hdiv_basis_ptr = disc->getBasis(dimension, cellTopo, "HDIV", hdiv_order);
      Teuchos::RCP<const panzer::Intrepid2FieldPattern> hdiv_pattern =
        Teuchos::rcp(new panzer::Intrepid2FieldPattern(hdiv_basis_ptr));
      hdiv_dof->addField(block_name, hdiv_basis, hdiv_pattern, panzer::FieldType::CG);
    }
    hdiv_dof->buildGlobalUnknowns();

    Teuchos::RCP<Thyra::LinearOpBase<ScalarT> > D1_thyra =
      panzer::buildInterpolation(conn, hcurl_dof, hdiv_dof,
                                 hcurl_basis, hdiv_basis,
                                 Intrepid2::OPERATOR_CURL,
                                 1000, false, true, false);
    auto D1_tpetra = Thyra::TpetraOperatorVectorExtraction<ScalarT,LO,GO,Node>::getTpetraOperator(D1_thyra);
    cntxt->refMaxwell.D1_matrix = Teuchos::rcp_dynamic_cast<LA_CrsMatrix>(D1_tpetra, true);

    const Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > t_block_map = blockMaps[targetBlock];
    const Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > d1_range_map = cntxt->refMaxwell.D1_matrix->getRangeMap();
    if (!d1_range_map->isSameAs(*t_block_map)) {
      Teuchos::RCP<LA_CrsMatrix> D1_remapped =
        Teuchos::rcp(new LA_CrsMatrix(t_block_map, std::max<size_t>(1, cntxt->refMaxwell.D1_matrix->getLocalMaxNumRowEntries())));
      const LO n_d1_rows = d1_range_map->getLocalNumElements();
      const LO n_t_rows = t_block_map->getLocalNumElements();
      TEUCHOS_TEST_FOR_EXCEPTION(n_d1_rows != n_t_rows, std::runtime_error,
        "D1 remap: local row counts differ (d1=" + std::to_string(n_d1_rows) + ", target=" + std::to_string(n_t_rows) + ").");
      const Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > d1_col_map = cntxt->refMaxwell.D1_matrix->getColMap();
      for (LO lid = 0; lid < n_d1_rows; ++lid) {
        const GO row_gid = t_block_map->getGlobalElement(lid);
        size_t nent = cntxt->refMaxwell.D1_matrix->getNumEntriesInLocalRow(lid);
        if (nent == 0) continue;
        host_inds_type col_lids("d1_col_lids", nent);
        host_vals_type row_vals("d1_row_vals", nent);
        cntxt->refMaxwell.D1_matrix->getLocalRowCopy(lid, col_lids, row_vals, nent);
        std::vector<GO> col_gids;
        std::vector<ScalarT> vals;
        col_gids.reserve(nent);
        vals.reserve(nent);
        for (size_t j = 0; j < nent; ++j) {
          const GO col_gid = d1_col_map->getGlobalElement(col_lids(j));
          if (col_gid == Teuchos::OrdinalTraits<GO>::invalid()) continue;
          col_gids.push_back(col_gid);
          vals.push_back(row_vals(j));
        }
        if (!col_gids.empty()) {
          D1_remapped->insertGlobalValues(row_gid, col_gids, vals);
        }
      }
      D1_remapped->fillComplete(cntxt->refMaxwell.D1_matrix->getDomainMap(), t_block_map);
      cntxt->refMaxwell.D1_matrix = D1_remapped;
    }
    TEUCHOS_TEST_FOR_EXCEPTION(!cntxt->refMaxwell.D1_matrix->getRangeMap()->isSameAs(*t_block_map), std::runtime_error,
      "ADS setup failed: D1 range map does not match target block map.");

    const Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > edge_map_for_D1 = cntxt->refMaxwell.D0_matrix->getRangeMap();
    if (!cntxt->refMaxwell.D1_matrix->getDomainMap()->isSameAs(*edge_map_for_D1)) {
      const Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > d1_row_map = cntxt->refMaxwell.D1_matrix->getRowMap();
      Teuchos::RCP<LA_CrsMatrix> D1_domain_remapped =
        Teuchos::rcp(new LA_CrsMatrix(d1_row_map, std::max<size_t>(1, cntxt->refMaxwell.D1_matrix->getLocalMaxNumRowEntries())));
      const LO n_rows = d1_row_map->getLocalNumElements();
      const Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > d1_col_map = cntxt->refMaxwell.D1_matrix->getColMap();
      for (LO lid = 0; lid < n_rows; ++lid) {
        const GO row_gid = d1_row_map->getGlobalElement(lid);
        size_t nent = cntxt->refMaxwell.D1_matrix->getNumEntriesInLocalRow(lid);
        if (nent == 0) continue;
        host_inds_type col_lids("d1_col_lids", nent);
        host_vals_type row_vals("d1_row_vals", nent);
        cntxt->refMaxwell.D1_matrix->getLocalRowCopy(lid, col_lids, row_vals, nent);
        std::vector<GO> col_gids;
        std::vector<ScalarT> vals;
        col_gids.reserve(nent);
        vals.reserve(nent);
        for (size_t j = 0; j < nent; ++j) {
          const GO col_gid = d1_col_map->getGlobalElement(col_lids(j));
          if (col_gid == Teuchos::OrdinalTraits<GO>::invalid()) continue;
          col_gids.push_back(col_gid);
          vals.push_back(row_vals(j));
        }
        if (!col_gids.empty()) {
          D1_domain_remapped->insertGlobalValues(row_gid, col_gids, vals);
        }
      }
      D1_domain_remapped->fillComplete(edge_map_for_D1, cntxt->refMaxwell.D1_matrix->getRangeMap());
      cntxt->refMaxwell.D1_matrix = D1_domain_remapped;
    }

    TEUCHOS_TEST_FOR_EXCEPTION(cntxt->refMaxwell.M2_matrix.is_null(), std::runtime_error,
      "ADS setup failed: M2 matrix is null after Schur auxiliary setup.");
    TEUCHOS_TEST_FOR_EXCEPTION(!cntxt->refMaxwell.M2_matrix->getRowMap()->isSameAs(*t_block_map) ||
                               !cntxt->refMaxwell.M2_matrix->getDomainMap()->isSameAs(*t_block_map), std::runtime_error,
      "ADS setup failed: M2 row/domain maps do not match target block map.");
    TEUCHOS_TEST_FOR_EXCEPTION(cntxt->refMaxwell.D1_matrix.is_null(), std::runtime_error,
      "ADS setup failed: D1 matrix is null after Schur auxiliary setup.");
    TEUCHOS_TEST_FOR_EXCEPTION(!cntxt->refMaxwell.D1_matrix->getRangeMap()->isSameAs(*t_block_map), std::runtime_error,
      "ADS setup failed: D1 range map does not match target block map after remap.");
    TEUCHOS_TEST_FOR_EXCEPTION(!cntxt->refMaxwell.D1_matrix->getDomainMap()->isSameAs(*cntxt->refMaxwell.D0_matrix->getRangeMap()), std::runtime_error,
      "ADS setup failed: D1 domain map does not match D0 range map after remap.");

  }

  
  debugger->print("**** setupBlockTriangularAuxiliary: end set " + std::to_string(set));
}

// ========================================================================================
// ========================================================================================

template<class Node>
void SolverManager<Node>::setupExplicitMass() {

  debugger->print("**** Starting SolverManager::setupExplicitMass()");
  
  bool compute_matrix = true;
  if (assembler->lump_mass || assembler->matrix_free) {
    compute_matrix = false;
  }
  
  for (size_t set=0; set<useBasis.size(); ++set) {
    matrix_RCP mass;
    
    assembler->updatePhysicsSet(set);
    if (compute_matrix) {
      explicitMass.push_back(linalg->getNewMatrix(set));
      if (linalg->getHaveOverlapped()) {
        mass = linalg->getNewOverlappedMatrix(set);
      }
      else {
        mass = explicitMass[set];
      }
    }
    
    diagMass.push_back(linalg->getNewVector(set));
    vector_RCP diagMass_over;
    if (linalg->getHaveOverlapped()) {
      diagMass_over = linalg->getNewOverlappedVector(set);
    } 
    else {
      diagMass_over = diagMass[set];
    }
    
    assembler->getWeightedMass(set,mass,diagMass_over);
    
    if (linalg->getHaveOverlapped()) {
      linalg->exportVectorFromOverlapped(set,diagMass[set], diagMass_over);
      if (compute_matrix) {
        linalg->exportMatrixFromOverlapped(set,explicitMass[set], mass);
      }
    }
    
  }

  debugger->print("**** Starting SolverManager::setupExplicitMass() - fillComplete");
  
  for (size_t set=0; set<useBasis.size(); ++set) {
    
    if (compute_matrix) {
      linalg->fillComplete(explicitMass[set]);
    }
    
    if (store_vectors) {
      q_pcg.push_back(linalg->getNewVector(set));
      z_pcg.push_back(linalg->getNewVector(set));
      p_pcg.push_back(linalg->getNewVector(set));
      r_pcg.push_back(linalg->getNewVector(set));
      if (linalg->getHaveOverlapped() && assembler->matrix_free) {
        q_pcg_over.push_back(linalg->getNewOverlappedVector(set));
        p_pcg_over.push_back(linalg->getNewOverlappedVector(set));
      }
    }
  }
  
  debugger->print("**** Finished SolverManager::setupExplicitMass()");
  
}


// ========================================================================================
// ========================================================================================

template<class Node>
void SolverManager<Node>::setupDiscretizedParamMass() {

  debugger->print("**** Starting SolverManager::setupDiscretizedParamMass()");
  
  // Hard coding this to always assemble the matrix
  // Can relax this and allow matrix-free later
  bool compute_matrix = true;
  
  matrix_RCP pmass;
  
  if (compute_matrix) {
    
    paramMass = linalg->getNewParamMatrix();
    
    if (linalg->getHaveOverlapped()) {
      pmass = linalg->getNewOverlappedParamMatrix();
    }
    else {
      pmass = paramMass;
    }
    
  }
  
  diagParamMass = linalg->getNewParamVector();
  vector_RCP diagParamMass_over;
  if (linalg->getHaveOverlapped()) {
    diagParamMass_over = linalg->getNewOverlappedParamVector();
  }
  else { // squeeze out memory for single rank demos
    diagParamMass_over = diagParamMass;
  }
  
  assembler->getParamMass(pmass,diagParamMass_over);
  
  if (linalg->getHaveOverlapped()) {
    linalg->exportParamVectorFromOverlapped(diagParamMass, diagParamMass_over);
    if (compute_matrix) {
      linalg->exportParamMatrixFromOverlapped(paramMass, pmass);
    }
  }
  

  if (compute_matrix) {
    linalg->fillComplete(paramMass);
  }
  
  params->setParamMass(diagParamMass, paramMass);
  
  debugger->print("**** Finished SolverManager::setupDiscretizedParamMass()");
  
}

//========================================================================
//========================================================================

template<class Node>
void SolverManager<Node>::setButcherTableau(const vector<string> & tableau, const int & set) {

  for (size_t block=0; block<assembler->groups.size(); ++block) {

    // TODO the RK scheme cannot be specified block by block

    auto myTableau = tableau[set];

    // only filling in the non-zero entries

    if (myTableau == "BWE" || myTableau == "DIRK-1,1") {
      butcher_A = Kokkos::View<ScalarT**,HostDevice>("butcher_A",1,1);
      butcher_A(0,0) = 1.0;
      butcher_b = Kokkos::View<ScalarT*,HostDevice>("butcher_b",1);
      butcher_b(0) = 1.0;
      butcher_c = Kokkos::View<ScalarT*,HostDevice>("butcher_c",1);
      butcher_c(0) = 1.0;
    }
    else if (myTableau == "FWE") {
      butcher_A = Kokkos::View<ScalarT**,HostDevice>("butcher_A",1,1);
      butcher_b = Kokkos::View<ScalarT*,HostDevice>("butcher_b",1);
      butcher_b(0) = 1.0;
      butcher_c = Kokkos::View<ScalarT*,HostDevice>("butcher_c",1);
    }
    else if (myTableau == "CN") {
      butcher_A = Kokkos::View<ScalarT**,HostDevice>("butcher_A",2,2);
      butcher_A(1,0) = 0.5;
      butcher_A(1,1) = 0.5;
      butcher_b = Kokkos::View<ScalarT*,HostDevice>("butcher_b",2);
      butcher_b(0) = 0.5;
      butcher_b(1) = 0.5;
      butcher_c = Kokkos::View<ScalarT*,HostDevice>("butcher_c",2);
      butcher_c(1) = 1.0;
    }
    else if (myTableau == "SSPRK-3,3") {
      butcher_A = Kokkos::View<ScalarT**,HostDevice>("butcher_A",3,3);
      butcher_A(1,0) = 1.0;
      butcher_A(2,0) = 0.25;
      butcher_A(2,1) = 0.25;
      butcher_b = Kokkos::View<ScalarT*,HostDevice>("butcher_b",3);
      butcher_b(0) = 1.0/6.0;
      butcher_b(1) = 1.0/6.0;
      butcher_b(2) = 2.0/3.0;
      butcher_c = Kokkos::View<ScalarT*,HostDevice>("butcher_c",3);
      butcher_c(1) = 1.0;
      butcher_c(2) = 1.0/2.0;
    }
    else if (myTableau == "RK-4,4") { // Classical RK4
      butcher_A = Kokkos::View<ScalarT**,HostDevice>("butcher_A",4,4);
      butcher_A(1,0) = 0.5;
      butcher_A(2,1) = 0.5;
      butcher_A(3,2) = 1.0;
      butcher_b = Kokkos::View<ScalarT*,HostDevice>("butcher_b",4);
      butcher_b(0) = 1.0/6.0;
      butcher_b(1) = 1.0/3.0;
      butcher_b(2) = 1.0/3.0;
      butcher_b(3) = 1.0/6.0;
      butcher_c = Kokkos::View<ScalarT*,HostDevice>("butcher_c",4);
      butcher_c(1) = 1.0/2.0;
      butcher_c(2) = 1.0/2.0;
      butcher_c(3) = 1.0;
    }
    else if (myTableau == "DIRK-1,2") {
      butcher_A = Kokkos::View<ScalarT**,HostDevice>("butcher_A",1,1);
      butcher_A(0,0) = 0.5;
      butcher_b = Kokkos::View<ScalarT*,HostDevice>("butcher_b",1);
      butcher_b(0) = 1.0;
      butcher_c = Kokkos::View<ScalarT*,HostDevice>("butcher_c",1);
      butcher_c(0) = 0.5;
    }
    else if (myTableau == "DIRK-2,2") { // 2-stage, 2nd order
      butcher_A = Kokkos::View<ScalarT**,HostDevice>("butcher_A",2,2);
      butcher_A(0,0) = 1.0/4.0;
      butcher_A(1,0) = 1.0/2.0;
      butcher_A(1,1) = 1.0/4.0;
      butcher_b = Kokkos::View<ScalarT*,HostDevice>("butcher_b",2);
      butcher_b(0) = 1.0/2.0;
      butcher_b(1) = 1.0/2.0;
      butcher_c = Kokkos::View<ScalarT*,HostDevice>("butcher_c",2);
      butcher_c(0) = 1.0/4.0;
      butcher_c(1) = 3.0/4.0;
    }
    else if (myTableau == "DIRK-2,3") { // 2-stage, 3rd order
      butcher_A = Kokkos::View<ScalarT**,HostDevice>("butcher_A",2,2);
      butcher_A(0,0) = 1.0/2.0 + std::sqrt(3)/6.0;
      butcher_A(1,0) = -std::sqrt(3)/3.0;
      butcher_A(1,1) = 1.0/2.0  + std::sqrt(3)/6.0;
      butcher_b = Kokkos::View<ScalarT*,HostDevice>("butcher_b",2);
      butcher_b(0) = 1.0/2.0;
      butcher_b(1) = 1.0/2.0;
      butcher_c = Kokkos::View<ScalarT*,HostDevice>("butcher_c",2);
      butcher_c(0) = 1.0/2.0 + std::sqrt(3)/6.0;;
      butcher_c(1) = 1.0/2.0 - std::sqrt(3)/6.0;;
    }
    else if (myTableau == "DIRK-3,3") { // 3-stage, 3rd order
      ScalarT p = 0.4358665215;
      butcher_A = Kokkos::View<ScalarT**,HostDevice>("butcher_A",3,3);
      butcher_A(0,0) = p;
      butcher_A(1,0) = (1.0-p)/2.0;
      butcher_A(1,1) = p;
      butcher_A(2,0) = -3.0*p*p/2.0+4.0*p-1.0/4.0;
      butcher_A(2,1) = 3.0*p*p/2.0 - 5.0*p + 5.0/4.0;
      butcher_A(2,2) = p;
      butcher_b = Kokkos::View<ScalarT*,HostDevice>("butcher_b",3);
      butcher_b(0) = -3.0*p*p/2.0+4.0*p-1.0/4.0;
      butcher_b(1) = 3.0*p*p/2.0-5.0*p+5.0/4.0;
      butcher_b(2) = p;
      butcher_c = Kokkos::View<ScalarT*,HostDevice>("butcher_c",3);
      butcher_c(0) = p;
      butcher_c(1) = (1.0+p)/2.0;
      butcher_c(2) = 1.0;
    }
    else if (myTableau == "leap-frog") { // Leap-frog for Maxwells
      butcher_A = Kokkos::View<ScalarT**,HostDevice>("butcher_A",2,2);
      butcher_A(1,0) = 1.0;
      butcher_b = Kokkos::View<ScalarT*,HostDevice>("butcher_b",2);
      butcher_b(0) = 1.0;
      butcher_b(1) = 1.0;
      butcher_c = Kokkos::View<ScalarT*,HostDevice>("butcher_c",2);
      butcher_c(0) = 0.0;
      butcher_c(1) = 0.0;
    }
    else if (myTableau == "custom") {

      string delimiter = ", ";
      string line_delimiter = "; ";
      size_t pos = 0;
      string b_A = settings->sublist("Solver").get<string>("transient Butcher A","1.0");
      string b_b = settings->sublist("Solver").get<string>("transient Butcher b","1.0");
      string b_c = settings->sublist("Solver").get<string>("transient Butcher c","1.0");
      vector<vector<double>> A_vals;
      if (b_A.find(delimiter) == string::npos) {
        vector<double> row;
        row.push_back(std::stod(b_A));
        A_vals.push_back(row);
      }
      else {
        string token;
        size_t linepos = 0;
        vector<string> lines;
        while ((linepos = b_A.find(line_delimiter)) != string::npos) {
          string line = b_A.substr(0,linepos);
          lines.push_back(line);
          b_A.erase(0, linepos + line_delimiter.length());
        }
        lines.push_back(b_A);
        for (size_t k=0; k<lines.size(); k++) {
          string line = lines[k];
          vector<double> row;
          while ((pos = line.find(delimiter)) != string::npos) {
            token = line.substr(0, pos);
            row.push_back(std::stod(token));
            line.erase(0, pos + delimiter.length());
          }
          row.push_back(std::stod(line));
          A_vals.push_back(row);
        }
      }
      // Make sure A is square
      size_t A_nrows = A_vals.size();
      for (size_t i=0; i<A_nrows; i++) {
        if (A_vals[i].size() != A_nrows) {
          TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: custom Butcher A is not a square matrix");
        }
      }

      vector<double> b_vals;
      if (b_b.find(delimiter) == string::npos) {
        b_vals.push_back(std::stod(b_b));
      }
      else {
        string token;
        while ((pos = b_b.find(delimiter)) != string::npos) {
          token = b_b.substr(0, pos);
          b_vals.push_back(std::stod(token));
          b_b.erase(0, pos + delimiter.length());
        }
        b_vals.push_back(std::stod(b_b));
      }

      // Make sure size of b matches A
      if (b_vals.size() != A_nrows) {
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: custom Butcher b does not match size of A");
      }

      vector<double> c_vals;
      if (b_c.find(delimiter) == string::npos) {
        c_vals.push_back(std::stod(b_c));
      }
      else {
        string token;
        while ((pos = b_c.find(delimiter)) != string::npos) {
          token = b_c.substr(0, pos);
          c_vals.push_back(std::stod(token));
          b_c.erase(0, pos + delimiter.length());
        }
        c_vals.push_back(std::stod(b_c));
      }

      // Make sure size of c matches A
      if (c_vals.size() != A_nrows) {
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: custom Butcher c does not match size of A");
      }

      // Create the views
      butcher_A = Kokkos::View<ScalarT**,HostDevice>("butcher_A",A_nrows,A_nrows);
      butcher_b = Kokkos::View<ScalarT*,HostDevice>("butcher_b",A_nrows);
      butcher_c = Kokkos::View<ScalarT*,HostDevice>("butcher_c",A_nrows);
      for (size_t i=0; i<A_nrows; i++) {
        for (size_t j=0; j<A_nrows; j++) {
          butcher_A(i,j) = A_vals[i][j];
        }
        butcher_b(i) = b_vals[i];
        butcher_c(i) = c_vals[i];
      }

    }
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: unrecognized Butcher tableau:" + tableau[set]);
    }
    Kokkos::View<ScalarT**,AssemblyDevice> dev_butcher_A("butcher_A on device",butcher_A.extent(0),butcher_A.extent(1));
    Kokkos::View<ScalarT*,AssemblyDevice> dev_butcher_b("butcher_b on device",butcher_b.extent(0));
    Kokkos::View<ScalarT*,AssemblyDevice> dev_butcher_c("butcher_c on device",butcher_c.extent(0));
  
    auto tmp_butcher_A = Kokkos::create_mirror_view(dev_butcher_A);
    auto tmp_butcher_b = Kokkos::create_mirror_view(dev_butcher_b);
    auto tmp_butcher_c = Kokkos::create_mirror_view(dev_butcher_c);
  
    Kokkos::deep_copy(tmp_butcher_A, butcher_A);
    Kokkos::deep_copy(tmp_butcher_b, butcher_b);
    Kokkos::deep_copy(tmp_butcher_c, butcher_c);
  
    Kokkos::deep_copy(dev_butcher_A, tmp_butcher_A);
    Kokkos::deep_copy(dev_butcher_b, tmp_butcher_b);
    Kokkos::deep_copy(dev_butcher_c, tmp_butcher_c);

    //block_butcher_A.push_back(dev_butcher_A);
    //block_butcher_b.push_back(dev_butcher_b);
    //block_butcher_c.push_back(dev_butcher_c);
  
    int newnumstages = butcher_A.extent(0);

    maxnumstages[set] = std::max(numstages[set],newnumstages);
    numstages[set] = newnumstages;
  
    assembler->setWorksetButcher(set, block, dev_butcher_A, dev_butcher_b, dev_butcher_c);

  } // end for blocks
}

// ========================================================================================
// ========================================================================================

template<class Node>
void SolverManager<Node>::setBackwardDifference(const vector<int> & order, const int & set) { // using order as an input to allow for dynamic changes

  // TODO rearrange this? and setButcher...

  for (size_t block=0; block<assembler->groups.size(); ++block) {

    // TODO currently, the BDF wts cannot be specified block by block

    Kokkos::View<ScalarT*,AssemblyDevice> dev_BDF_wts;
    Kokkos::View<ScalarT*,HostDevice> BDF_wts;

    // Note that these do not include 1/deltat (added in wkset)
    // Not going to work properly for adaptive time stepping if BDForder>1

    auto myOrder = order[set];

    if (isTransient) {

      if (myOrder == 1) {
        BDF_wts = Kokkos::View<ScalarT*,HostDevice>("BDF weights to compute u_dot",2);
        BDF_wts(0) = 1.0;
        BDF_wts(1) = -1.0;
      }
      else if (myOrder == 2) {
        BDF_wts = Kokkos::View<ScalarT*,HostDevice>("BDF weights to compute u_dot",3);
        BDF_wts(0) = 1.5;
        BDF_wts(1) = -2.0;
        BDF_wts(2) = 0.5;
      }
      else if (myOrder == 3) {
        BDF_wts = Kokkos::View<ScalarT*,HostDevice>("BDF weights to compute u_dot",4);
        BDF_wts(0) = 11.0/6.0;
        BDF_wts(1) = -3.0;
        BDF_wts(2) = 1.5;
        BDF_wts(3) = -1.0/3.0;
      }
      else if (myOrder == 4) {
        BDF_wts = Kokkos::View<ScalarT*,HostDevice>("BDF weights to compute u_dot",5);
        BDF_wts(0) = 25.0/12.0;
        BDF_wts(1) = -4.0;
        BDF_wts(2) = 3.0;
        BDF_wts(3) = -4.0/3.0;
        BDF_wts(4) = 1.0/4.0;
      }
      else if (myOrder == 5) {
        BDF_wts = Kokkos::View<ScalarT*,HostDevice>("BDF weights to compute u_dot",6);
        BDF_wts(0) = 137.0/60.0;
        BDF_wts(1) = -5.0;
        BDF_wts(2) = 5.0;
        BDF_wts(3) = -10.0/3.0;
        BDF_wts(4) = 75.0/60.0;
        BDF_wts(5) = -1.0/5.0;
      }
      else if (myOrder == 6) {
        BDF_wts = Kokkos::View<ScalarT*,HostDevice>("BDF weights to compute u_dot",7);
        BDF_wts(0) = 147.0/60.0;
        BDF_wts(1) = -6.0;
        BDF_wts(2) = 15.0/2.0;
        BDF_wts(3) = -20.0/3.0;
        BDF_wts(4) = 225.0/60.0;
        BDF_wts(5) = -72.0/60.0;
        BDF_wts(6) = 1.0/6.0;
      }

      int newnumsteps = BDF_wts.extent(0)-1;

      maxnumsteps[set] = std::max(maxnumsteps[set],newnumsteps);
      numsteps[set] = newnumsteps;

    }
    else { // for steady state solves, u_dot = 0.0*u
      BDF_wts = Kokkos::View<ScalarT*,HostDevice>("BDF weights to compute u_dot",1);
      BDF_wts(0) = 1.0;
      numsteps[set] = 1;
      maxnumsteps[set] = 1;
    }

    dev_BDF_wts = Kokkos::View<ScalarT*,AssemblyDevice>("BDF weights on device",BDF_wts.extent(0));
    Kokkos::deep_copy(dev_BDF_wts, BDF_wts);
    
    assembler->setWorksetBDF(set, block, dev_BDF_wts);
  } // end loop blocks
}

/////////////////////////////////////////////////////////////////////////////
// Worksets
/////////////////////////////////////////////////////////////////////////////

template<class Node>
void SolverManager<Node>::finalizeWorkset() {
  
  debugger->print("**** Starting SolverManager::finalizeWorkset ...");
  
  this->finalizeWorkset(assembler->wkset, params->paramvals_KV, params->paramdot_KV);
#ifndef MrHyDE_NO_AD
  this->finalizeWorkset(assembler->wkset_AD, params->paramvals_KVAD, params->paramdot_KVAD);
  this->finalizeWorkset(assembler->wkset_AD2, params->paramvals_KVAD2, params->paramdot_KVAD2);
  this->finalizeWorkset(assembler->wkset_AD4, params->paramvals_KVAD4, params->paramdot_KVAD4);
  this->finalizeWorkset(assembler->wkset_AD8, params->paramvals_KVAD8, params->paramdot_KVAD8);
  this->finalizeWorkset(assembler->wkset_AD16, params->paramvals_KVAD16, params->paramdot_KVAD16);
  this->finalizeWorkset(assembler->wkset_AD18, params->paramvals_KVAD18, params->paramdot_KVAD18);
  this->finalizeWorkset(assembler->wkset_AD24, params->paramvals_KVAD24, params->paramdot_KVAD24);
  this->finalizeWorkset(assembler->wkset_AD32, params->paramvals_KVAD32, params->paramdot_KVAD32);
#endif
  
  debugger->print("**** Finished SolverManager::finalizeWorkset");
  
  
}

template<class Node>
template<class EvalT>
void SolverManager<Node>::finalizeWorkset(vector<Teuchos::RCP<Workset<EvalT> > > & wkset,
                                          Kokkos::View<EvalT**,AssemblyDevice> paramvals_KV,
                                          Kokkos::View<EvalT**,AssemblyDevice> paramdot_KV) {

  // Determine the offsets for each set as a Kokkos View
  for (size_t block=0; block<wkset.size(); ++block) {
    if (wkset[block]->isInitialized) {
      for (size_t set=0; set<physics->set_names.size(); set++) {
        vector<vector<int> > voffsets = disc->offsets[set][block];
        size_t maxoff = 0;
        for (size_t i=0; i<voffsets.size(); i++) {
          if (voffsets[i].size() > maxoff) {
            maxoff = voffsets[i].size();
          }
        }
        
        Kokkos::View<int**,AssemblyDevice> offsets_view("offsets on assembly device",voffsets.size(),maxoff);
        auto host_offsets = Kokkos::create_mirror_view(offsets_view);
        for (size_t i=0; i<voffsets.size(); i++) {
          for (size_t j=0; j<voffsets[i].size(); j++) {
            host_offsets(i,j) = voffsets[i][j];
          }
        }
        Kokkos::deep_copy(offsets_view,host_offsets);
        wkset[block]->set_offsets.push_back(offsets_view);
        if (set == 0) {
          wkset[block]->offsets = offsets_view;
        }

      }
    }
  }
  
  for (size_t block=0; block<wkset.size(); ++block) {
    if (wkset[block]->isInitialized) {
      
      vector<vector<int> > block_useBasis;
      vector<vector<string> > block_varlist;
      
      for (size_t set=0; set<useBasis.size(); ++set) {
        block_useBasis.push_back(useBasis[set][block]);
        block_varlist.push_back(varlist[set][block]);
      }
      wkset[block]->set_usebasis = block_useBasis;
      wkset[block]->set_varlist = block_varlist;
      wkset[block]->usebasis = block_useBasis[0];
      wkset[block]->varlist = block_varlist[0];
      
    }
  }
  
  for (size_t block=0; block<wkset.size(); ++block) {
    if (wkset[block]->isInitialized) {
      // set defaults for time integration params since these
      // won't get set if the total number of sets is 1
      wkset[block]->butcher_A = wkset[block]->set_butcher_A[0];
      wkset[block]->butcher_b = wkset[block]->set_butcher_b[0];
      wkset[block]->butcher_c = wkset[block]->set_butcher_c[0];
      wkset[block]->BDF_wts = wkset[block]->set_BDF_wts[0];
      // update workset for first physics set
      wkset[block]->updatePhysicsSet(0);

    }
  }
  
  // Parameters do not depend on physics sets
  for (size_t block=0; block<wkset.size(); ++block) {
    if (wkset[block]->isInitialized) {
      size_t maxpoff = 0;
      for (size_t i=0; i<params->paramoffsets.size(); i++) {
        if (params->paramoffsets[i].size() > maxpoff) {
          maxpoff = params->paramoffsets[i].size();
        }
      }
      
      Kokkos::View<int**,AssemblyDevice> poffsets_view("param offsets on assembly device",params->paramoffsets.size(),maxpoff);
      auto host_poffsets = Kokkos::create_mirror_view(poffsets_view);
      for (size_t i=0; i<params->paramoffsets.size(); i++) {
        for (size_t j=0; j<params->paramoffsets[i].size(); j++) {
          host_poffsets(i,j) = params->paramoffsets[i][j];
        }
      }
      Kokkos::deep_copy(poffsets_view,host_poffsets);
      wkset[block]->paramusebasis = params->discretized_param_usebasis;
      wkset[block]->paramoffsets = poffsets_view;
      wkset[block]->param_varlist = params->discretized_param_names;

    }
  }
  
  for (size_t block=0; block<wkset.size(); ++block) {
    if (wkset[block]->isInitialized) {
      wkset[block]->createSolutionFields();
    }
  }
  
  for (size_t block=0; block<wkset.size(); ++block) {
    if (wkset[block]->isInitialized) {
      vector<vector<int> > block_useBasis;
      for (size_t set=0; set<useBasis.size(); ++set) {
        block_useBasis.push_back(useBasis[set][block]);
      }
      assembler->groupData[block]->setSolutionFields(maxnumsteps, maxnumstages);
      for (size_t grp=0; grp<assembler->groups[block].size(); ++grp) {
        assembler->groups[block][grp]->setUseBasis(block_useBasis, maxnumsteps, maxnumstages, false);
        assembler->groups[block][grp]->setUpSubGradient(params->num_active_params);
      }
      
      wkset[block]->params_AD = paramvals_KV;
      wkset[block]->params_dot_AD = paramdot_KV;
      wkset[block]->paramnames = params->paramnames;
      wkset[block]->setTime(current_time);

      if (assembler->boundary_groups.size() > block) { // avoid seg faults
        for (size_t grp=0; grp<assembler->boundary_groups[block].size(); ++grp) {
          if (assembler->boundary_groups[block][grp]->numElem > 0) {
            assembler->boundary_groups[block][grp]->setUseBasis(block_useBasis, maxnumsteps, maxnumstages, false);
          }
        }
      }
    }
  }
  
  
}

// ========================================================================================
// Set up the logicals and data structures for the fixed DOF (Dirichlet and point constraints)
// ========================================================================================

template<class Node>
void SolverManager<Node>::setupFixedDOFs(Teuchos::RCP<Teuchos::ParameterList> & settings) {
  
  Teuchos::TimeMonitor localtimer(*fixeddofsetuptimer);
  
  debugger->print("**** Starting SolverManager::setupFixedDOFs()");
  
  if (!disc->have_dirichlet) {
    usestrongDBCs = false;
  }
  
  size_t numSets = physics->set_names.size();
  
  scalarDirichletData = vector<bool>(numSets,false);
  staticDirichletData = vector<bool>(numSets,true);
  
  if (usestrongDBCs) {
    for (size_t set=0; set<numSets; ++set) {
      fixedDOF_soln.push_back(linalg->getNewOverlappedVector(set));
    }
    for (size_t set=0; set<numSets; ++set) {
    
      scalarDirichletData[set] = settings->sublist("Physics").sublist("Dirichlet conditions").get<bool>("scalar data", false);
      staticDirichletData[set] = settings->sublist("Physics").sublist("Dirichlet conditions").get<bool>("static data", true);
      
      if (scalarDirichletData[set] && !staticDirichletData[set]) {
        if (Comm->getRank() == 0) {
          cout << "Warning: The Dirichlet data was set to scalar and non-static.  This should not happen." << endl;
        }
      }
      
      if (scalarDirichletData[set]) {
        vector<vector<ScalarT> > setDirichletValues;
        for (size_t block=0; block<blocknames.size(); ++block) {
          
          std::string blockID = blocknames[block];
          Teuchos::ParameterList dbc_settings = physics->physics_settings[set][block].sublist("Dirichlet conditions");
          vector<ScalarT> blockDirichletValues;
          
          for (size_t var=0; var<varlist[set][block].size(); var++ ) {
            ScalarT value = 0.0;
            if (dbc_settings.isSublist(varlist[set][block][var])) {
              if (dbc_settings.sublist(varlist[set][block][var]).isParameter("all boundaries")) {
                value = dbc_settings.sublist(varlist[set][block][var]).template get<ScalarT>("all boundaries");
              }
              else {
                Teuchos::ParameterList currdbcs = dbc_settings.sublist(varlist[set][block][var]);
                Teuchos::ParameterList::ConstIterator d_itr = currdbcs.begin();
                while (d_itr != currdbcs.end()) {
                  value = currdbcs.get<ScalarT>(d_itr->first);
                  d_itr++;
                }
              }
            }
            blockDirichletValues.push_back(value);
          }
          setDirichletValues.push_back(blockDirichletValues);
        }
        scalarDirichletValues.push_back(setDirichletValues);
      }
    }
  }
  
  debugger->print("**** Finished SolverManager::setupFixedDOFs()");
  
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template<class Node>
void SolverManager<Node>::finalizeParams() {
  
  //for (size_t block=0; block<blocknames.size(); ++block) {
  //  assembler->wkset[block]->paramusebasis = params->discretized_param_usebasis;
  //  assembler->wkset[block]->paramoffsets = params->paramoffsets[0];
  // }
  
}

////////////////////////////////////////////////////////////////////////////////
// The following function is not updated for multi-set
////////////////////////////////////////////////////////////////////////////////

template<class Node>
void SolverManager<Node>::finalizeMultiscale() {
#ifndef MrHyDE_NO_AD
  if (multiscale_manager->subgridModels.size() > 0 ) {
    for (size_t k=0; k<multiscale_manager->subgridModels.size(); k++) {
      multiscale_manager->subgridModels[k]->paramvals_KVAD = params->paramvals_KVAD;
    }
    
    multiscale_manager->macro_wkset = assembler->wkset_AD;
    vector<Kokkos::View<int*,AssemblyDevice>> macro_numDOF;
    for (size_t block=0; block<assembler->groupData.size(); ++block) {
      macro_numDOF.push_back(assembler->groupData[block]->set_num_dof[0]);
    }
    multiscale_manager->setMacroInfo(disc->basis_pointers, disc->basis_types,
                                     physics->var_list[0], useBasis[0], disc->offsets[0],
                                     macro_numDOF, params->paramnames, params->discretized_param_names);
    
    vector<vector<int> > sgmodels = assembler->identifySubgridModels();
    ScalarT my_cost = multiscale_manager->initialize(sgmodels);
    ScalarT gmin = 0.0;
    Teuchos::reduceAll(*Comm,Teuchos::REDUCE_MIN,1,&my_cost,&gmin);
    ScalarT gmax = 0.0;
    Teuchos::reduceAll(*Comm,Teuchos::REDUCE_MAX,1,&my_cost,&gmin);
    
    assembler->multiscale_manager = multiscale_manager;
    if (Comm->getRank() == 0 && verbosity>0) {
      cout << "***** Load Balancing Factor " << gmax/gmin <<  endl;
    }
    
  }
#endif  
}
