/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
************************************************************************/

// ========================================================================================
// ========================================================================================

namespace {
// Detect block settings that require mass matrices or distance-laplacian coordinates.
inline bool anyBlockSettingsRequestsAuxiliary(const Teuchos::ParameterList & solverList) {
  for (Teuchos::ParameterList::ConstIterator it = solverList.begin(); it != solverList.end(); ++it) {
    const std::string key = solverList.name(it);
    if (key.compare(0, 6, "Block ") != 0) continue;
    if (!solverList.isSublist(key)) continue;
    const Teuchos::ParameterList & sub = solverList.sublist(key);
    if (sub.isParameter("use mass matrix") && sub.get<bool>("use mass matrix")) return true;
    if (sub.isParameter("hgrad basis name") && sub.isParameter("hcurl basis name")) return true;
  }
  return false;
}

// Build edge coordinates for MueLu distance-laplacian aggregation.
template<class ScalarT, class LO, class GO, class Node, class GidCoordMap>
Teuchos::RCP<Tpetra::MultiVector<typename Teuchos::ScalarTraits<ScalarT>::coordinateType,LO,GO,Node> >
buildEdgeAveragedNodeCoords(const Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,Node> > & D0_matrix,
                        const Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > & edge_block_map,
                        const GidCoordMap & gid_to_coords,
                        const int dimension) {
  using CoordScalar = typename Teuchos::ScalarTraits<ScalarT>::coordinateType;
  using CoordMV = Tpetra::MultiVector<CoordScalar,LO,GO,Node>;
  using CrsMatrix = Tpetra::CrsMatrix<ScalarT,LO,GO,Node>;
  Teuchos::RCP<CoordMV> edge_coords = Teuchos::rcp(new CoordMV(edge_block_map, dimension));
  auto view = edge_coords->getLocalViewHost(Tpetra::Access::OverwriteAll);
  const auto D0_col_map = D0_matrix->getColMap();
  typedef typename CrsMatrix::nonconst_local_inds_host_view_type host_inds_t;
  typedef typename CrsMatrix::nonconst_values_host_view_type    host_vals_t;
  const LO n_rows = static_cast<LO>(edge_block_map->getLocalNumElements());
  for (LO lid = 0; lid < n_rows; ++lid) {
    size_t nent = D0_matrix->getNumEntriesInLocalRow(lid);
    if (nent == 0) continue;
    host_inds_t col_lids("d0_edge_col_lids", nent);
    host_vals_t row_vals("d0_edge_row_vals", nent);
    D0_matrix->getLocalRowCopy(lid, col_lids, row_vals, nent);
    std::vector<double> acc(dimension, 0.0);
    int found = 0;
    for (size_t j = 0; j < nent; ++j) {
      const GO col_gid = D0_col_map->getGlobalElement(col_lids(j));
      if (col_gid == Teuchos::OrdinalTraits<GO>::invalid()) continue;
      auto it = gid_to_coords.find(col_gid);
      if (it == gid_to_coords.end()) continue;
      for (int d = 0; d < dimension; ++d) acc[d] += it->second[d];
      ++found;
    }
    if (found == 0) continue;
    const double inv = 1.0 / static_cast<double>(found);
    for (int d = 0; d < dimension; ++d) {
      view(lid, d) = static_cast<CoordScalar>(acc[d] * inv);
    }
  }
  return edge_coords;
}
} // namespace

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

  auto needs_block_auxiliary = [this](const Teuchos::RCP<LinearSolverContext<Node> > & cntxt) -> bool {
    if (cntxt.is_null()) return false;
    const bool use_block_tri = (cntxt->prec_type == "block triangular");
    const bool use_block_diag = (cntxt->prec_type == "block diagonal");
    std::string pivot_prec = cntxt->schur.pivot_block_preconditioner_type;
    for (size_t i = 0; i < pivot_prec.size(); ++i)
      pivot_prec[i] = static_cast<char>(std::toupper(static_cast<unsigned char>(pivot_prec[i])));
    std::string schur_prec = cntxt->schur.schur_block_preconditioner_type;
    for (size_t i = 0; i < schur_prec.size(); ++i)
      schur_prec[i] = static_cast<char>(std::toupper(static_cast<unsigned char>(schur_prec[i])));
    const bool use_refmaxwell = (pivot_prec == "REFMAXWELL" || pivot_prec == "MAXWELL1");
    const bool use_refmaxwell_schur = (schur_prec == "REFMAXWELL" || schur_prec == "MAXWELL1");
    const bool needs_refmaxwell_auxiliary =
      (use_block_tri && (use_refmaxwell || use_refmaxwell_schur)) ||
      (use_block_diag && use_refmaxwell);
    if (needs_refmaxwell_auxiliary) return true;
    // Block-diagonal mass swaps and coordinate aggregation also need auxiliary data.
    if (use_block_diag && settings != Teuchos::null) {
      return anyBlockSettingsRequestsAuxiliary(settings->sublist("Solver"));
    }
    return false;
  };

  // Share RefMaxwell auxiliary data across contexts for this set.
  for (size_t set = 0; set < setnames.size(); ++set) {
    Teuchos::RCP<LinearSolverContext<Node> > source_context = Teuchos::null;
    if (set < linalg->context.size() && needs_block_auxiliary(linalg->context[set])) {
      source_context = linalg->context[set];
    }
    else if (set < linalg->context_L2.size() && needs_block_auxiliary(linalg->context_L2[set])) {
      source_context = linalg->context_L2[set];
    }
    else if (set < linalg->context_BndryL2.size() && needs_block_auxiliary(linalg->context_BndryL2[set])) {
      source_context = linalg->context_BndryL2[set];
    }
    if (source_context.is_null()) continue;

    this->setupBlockTriangularAuxiliary(set, source_context);

    auto share_auxiliary_data = [&](auto & ctxVec) {
      if (set < ctxVec.size() && !ctxVec[set].is_null() && ctxVec[set] != source_context) {
        auto & dst = ctxVec[set]->refMaxwell;
        const auto & src = source_context->refMaxwell;
        dst.D0_matrix = src.D0_matrix;
        dst.M1_matrix = src.M1_matrix;
        dst.block_mass_matrices = src.block_mass_matrices;
        dst.block_dof_coords = src.block_dof_coords;
        dst.nodal_coords = src.nodal_coords;
        dst.nodal_lumped_mass = src.nodal_lumped_mass;
        dst.nullspace = src.nullspace;
      }
    };
    share_auxiliary_data(linalg->context);
    share_auxiliary_data(linalg->context_L2);
    share_auxiliary_data(linalg->context_BndryL2);
  }
  
  debugger->print("**** Finished SolverManager::completeSetup()");
  
}

// ========================================================================================
// ========================================================================================

// D0, M1, and nodal coords for RefMaxwell; stored in cntxt->refMaxwell and shared in completeSetup().
template<class Node>
void SolverManager<Node>::setupBlockTriangularAuxiliary(const size_t & set,
                                                       const Teuchos::RCP<LinearSolverContext<Node> > & cntxt) {
  TEUCHOS_TEST_FOR_EXCEPTION(cntxt.is_null(), std::runtime_error,
    "Missing linear solver context for set " + std::to_string(set));
  debugger->print("**** setupBlockTriangularAuxiliary: begin set " + std::to_string(set));

  // Unit-weight M1 for RefMaxwell/Maxwell1; otherwise physics weights.
  const bool pivotHasRefMaxwell =
    (cntxt->pivot_block_sublist.name() != "empty") &&
    (cntxt->pivot_block_sublist.isSublist("RefMaxwell Settings") ||
     cntxt->pivot_block_sublist.isSublist("Maxwell1 Settings"));
  const bool schurHasRefMaxwell =
    (cntxt->schur_block_sublist.name() != "empty") &&
    (cntxt->schur_block_sublist.isSublist("RefMaxwell Settings") ||
     cntxt->schur_block_sublist.isSublist("Maxwell1 Settings"));
  const bool use_unit_mass = pivotHasRefMaxwell || schurHasRefMaxwell;

  // Assemble full H(curl) mass matrix M1 (overlapped then exported). Used for RefMaxwell edge block.
  matrix_RCP M1_over = linalg->getNewOverlappedMatrix(set);
  vector_RCP diagM1_over = linalg->getNewOverlappedVector(set);
  assembler->updatePhysicsSet(set);
  {
    Teuchos::ParameterList & solverList = settings->sublist("Solver");
    TEUCHOS_TEST_FOR_EXCEPTION(solverList.get<bool>("sparse mass format", false) ||
                               solverList.get<bool>("lump mass", false) ||
                               solverList.get<bool>("matrix free", false), std::runtime_error,
      "Auxiliary setup needs an assembled M1: disable sparse mass, lumping, matrix-free.");
  }
  assembler->getWeightedMass(set, M1_over, diagM1_over, use_unit_mass);

  matrix_RCP assembled_mass_matrix = linalg->getNewMatrix(set);
  linalg->exportMatrixFromOverlapped(set, assembled_mass_matrix, M1_over);
  linalg->fillComplete(assembled_mass_matrix);

  // One map per variable block (same as block_prec). Identifies which block is edge (HCURL) for M1/D0.
  std::vector<Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > > blockMaps = linalg->buildBlockMaps(set);
  TEUCHOS_TEST_FOR_EXCEPTION(blockMaps.empty(), std::runtime_error,
    "Block-triangular auxiliary setup requires at least one block map.");
  const int pivotBlock = cntxt->schur.pivot_block;

  // Cache block mass matrices for optional substitution.
  cntxt->refMaxwell.block_mass_matrices.assign(blockMaps.size(), Teuchos::null);
  for (size_t b = 0; b < blockMaps.size(); ++b) {
    cntxt->refMaxwell.block_mass_matrices[b] = linalg->extractDiagonalBlock(assembled_mass_matrix, blockMaps[b]);
  }

  // Complete HGRAD and HCURL settings request distance-laplacian coordinates.
  bool needs_distance_laplacian_coords = false;
  std::string dl_hgrad_basis, dl_hcurl_basis;
  int dl_hgrad_order = 1, dl_hcurl_order = 1;
  if (settings != Teuchos::null) {
    Teuchos::ParameterList & solverList = settings->sublist("Solver");
    for (Teuchos::ParameterList::ConstIterator it = solverList.begin(); it != solverList.end(); ++it) {
      const std::string key = solverList.name(it);
      if (key.compare(0, 6, "Block ") != 0) continue;
      if (!solverList.isSublist(key)) continue;
      const Teuchos::ParameterList & sub = solverList.sublist(key);
      if (sub.isParameter("hgrad basis name") && sub.isParameter("hcurl basis name")) {
        needs_distance_laplacian_coords = true;
        dl_hgrad_basis = sub.get<std::string>("hgrad basis name");
        dl_hcurl_basis = sub.get<std::string>("hcurl basis name");
        if (sub.isParameter("hgrad basis order"))
          dl_hgrad_order = sub.get<int>("hgrad basis order");
        if (sub.isParameter("hcurl basis order"))
          dl_hcurl_order = sub.get<int>("hcurl basis order");
        break;
      }
    }
  }

  // The mass-only path does not need D0 or coordinates.
  if (!pivotHasRefMaxwell && !schurHasRefMaxwell && !needs_distance_laplacian_coords) {
    debugger->print("**** setupBlockTriangularAuxiliary: done (mass-only, set " + std::to_string(set) + ")");
    return;
  }

  TEUCHOS_TEST_FOR_EXCEPTION(pivotBlock < 0 || static_cast<size_t>(pivotBlock) >= blockMaps.size(),
    std::runtime_error,
    "Schur pivot block index " + std::to_string(pivotBlock) + " is out of range for set " +
    std::to_string(set) + " with " + std::to_string(blockMaps.size()) + " blocks.");

  // Basis setting precedence: RefMaxwell / Maxwell1, preconditioner/Schur, then block settings.
  const Teuchos::ParameterList * refmaxwellSetupListPtr = nullptr;
  auto pickSublist = [](const Teuchos::ParameterList & p) -> const Teuchos::ParameterList * {
    if (p.isSublist("RefMaxwell Settings")) return &p.sublist("RefMaxwell Settings");
    if (p.isSublist("Maxwell1 Settings"))   return &p.sublist("Maxwell1 Settings");
    return nullptr;
  };
  if (pivotHasRefMaxwell)
    refmaxwellSetupListPtr = pickSublist(cntxt->pivot_block_sublist);
  else if (schurHasRefMaxwell)
    refmaxwellSetupListPtr = pickSublist(cntxt->schur_block_sublist);
  if (refmaxwellSetupListPtr == nullptr) {
    if (cntxt->prec_sublist.name() != "empty" && cntxt->prec_sublist.isParameter("hgrad basis name"))
      refmaxwellSetupListPtr = &cntxt->prec_sublist;
    else if (cntxt->schur_block_sublist.name() != "empty" && cntxt->schur_block_sublist.isParameter("hgrad basis name"))
      refmaxwellSetupListPtr = &cntxt->schur_block_sublist;
  }
  std::string hgrad_basis, hcurl_basis;
  int hgrad_order = 1, hcurl_order = 1;
  if (refmaxwellSetupListPtr != nullptr && refmaxwellSetupListPtr->isParameter("hgrad basis name")) {
    TEUCHOS_TEST_FOR_EXCEPTION(!refmaxwellSetupListPtr->isParameter("hcurl basis name"), std::runtime_error,
      "Block-triangular auxiliary requires 'hcurl basis name' in the same list as 'hgrad basis name'.");
    const Teuchos::ParameterList & refmaxwellSetupList = *refmaxwellSetupListPtr;
    hgrad_basis = refmaxwellSetupList.template get<std::string>("hgrad basis name");
    hcurl_basis = refmaxwellSetupList.template get<std::string>("hcurl basis name");
    if (refmaxwellSetupList.isParameter("hgrad basis order"))
      hgrad_order = refmaxwellSetupList.template get<int>("hgrad basis order");
    if (refmaxwellSetupList.isParameter("hcurl basis order"))
      hcurl_order = refmaxwellSetupList.template get<int>("hcurl basis order");
  } else if (needs_distance_laplacian_coords) {
    hgrad_basis = dl_hgrad_basis;
    hcurl_basis = dl_hcurl_basis;
    hgrad_order = dl_hgrad_order;
    hcurl_order = dl_hcurl_order;
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
      "RefMaxwell auxiliary setup needs 'hgrad basis name' and 'hcurl basis name' in Pivot or Schur Block Settings.");
  }

  if ((hgrad_order != 1 || hcurl_order != 1) && Comm->getRank() == 0) {
    std::cout << "WARNING: RefMaxwell/Maxwell1 auxiliary spaces assume order 1 "
              << "(hgrad basis order = " << hgrad_order
              << ", hcurl basis order = " << hcurl_order << ")."
              << std::endl;
  }

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

  // identify edge block by basis type
  const Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > aux_edge_map = cntxt->refMaxwell.D0_matrix->getRangeMap();
  size_t edgeBlock = static_cast<size_t>(pivotBlock);
  const auto & setBasis = useBasis[set][0];
  for (size_t v = 0; v < setBasis.size() && v < blockMaps.size(); ++v) {
    const LO bind = setBasis[v];
    if (bind >= 0 && static_cast<size_t>(bind) < disc->basis_types[0].size() &&
        disc->basis_types[0][bind].substr(0,5) == "HCURL") {
      edgeBlock = v;
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
  cntxt->refMaxwell.M1_matrix = linalg->extractDiagonalBlock(assembled_mass_matrix, edge_block_map);

  // The auxiliary and primary edge maps can order GIDs differently.
  // Match D0 rows by HCURL field offset instead of local index.
  typedef typename LA_CrsMatrix::nonconst_local_inds_host_view_type host_inds_type;
  typedef typename LA_CrsMatrix::nonconst_values_host_view_type host_vals_type;
  const Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > nodal_map = cntxt->refMaxwell.D0_matrix->getDomainMap();
  const Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > d0_col_map = cntxt->refMaxwell.D0_matrix->getColMap();
  if (!aux_edge_map->isSameAs(*edge_block_map)) {
    const int hcurl_field_num = hcurl_dof->getFieldNum(hcurl_basis);
    std::unordered_map<GO,GO> aux2prim;
    GO conflictGid = -1;
    for (size_t b = 0; b < mesh->block_names.size(); ++b) {
      const std::string & block_name = mesh->block_names[b];
      const auto & block_offsets = disc->offsets[set][b];
      TEUCHOS_TEST_FOR_EXCEPTION(edgeBlock >= block_offsets.size(), std::runtime_error,
        "D0 remap: edge variable " + std::to_string(edgeBlock) + " is invalid for " +
        block_name + " (" + std::to_string(block_offsets.size()) + " variables).");
      const std::vector<int> & E_off = block_offsets[edgeBlock];
      const std::vector<int> aux_off = hcurl_dof->getGIDFieldOffsets(block_name, hcurl_field_num);
      TEUCHOS_TEST_FOR_EXCEPTION(aux_off.size() != E_off.size(), std::runtime_error,
        "D0 remap: HCURL offset count mismatch on " + block_name +
        " (auxiliary=" + std::to_string(aux_off.size()) +
        ", primary=" + std::to_string(E_off.size()) + ").");
      const size_t num_elem = disc->my_elements[b].extent(0);
      for (size_t e = 0; e < num_elem; ++e) {
        LO local_elem_id = disc->my_elements[b](e);
        std::vector<GO> aux_gids, prim_gids;
        hcurl_dof->getElementGIDs(local_elem_id, aux_gids, block_name);
        prim_gids = disc->getGIDs(set, b, local_elem_id);
        for (size_t j = 0; j < aux_off.size(); ++j) {
          const GO aux_gid = aux_gids[aux_off[j]];
          const GO prim_gid = prim_gids[E_off[j]];
          auto ins = aux2prim.emplace(aux_gid, prim_gid);
          if (!ins.second && ins.first->second != prim_gid && conflictGid < 0) conflictGid = aux_gid;
        }
      }
    }

    Teuchos::RCP<LA_CrsMatrix> D0_remapped =
      Teuchos::rcp(new LA_CrsMatrix(edge_block_map, std::max<size_t>(1, cntxt->refMaxwell.D0_matrix->getLocalMaxNumRowEntries())));
    const LO n_aux_rows = aux_edge_map->getLocalNumElements();
    GO unmappedGid = -1;
    for (LO lid = 0; lid < n_aux_rows; ++lid) {
      const GO aux_row_gid = aux_edge_map->getGlobalElement(lid);
      auto it = aux2prim.find(aux_row_gid);
      if (it == aux2prim.end()) {
        if (unmappedGid < 0) unmappedGid = aux_row_gid;
        continue;
      }
      const GO row_gid = it->second;
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
    GO bad[2] = {conflictGid, unmappedGid}, worst[2] = {-1, -1};
    Teuchos::reduceAll<int, GO>(*(aux_edge_map->getComm()), Teuchos::REDUCE_MAX, 2, bad, worst);
    TEUCHOS_TEST_FOR_EXCEPTION(worst[0] >= 0, std::runtime_error,
      "D0 remap: auxiliary edge GID " << worst[0] << " maps to two primary GIDs.");
    TEUCHOS_TEST_FOR_EXCEPTION(worst[1] >= 0, std::runtime_error,
      "D0 remap: no primary GID for auxiliary GID " << worst[1] << ".");
    D0_remapped->fillComplete(nodal_map, edge_block_map);
    cntxt->refMaxwell.D0_matrix = D0_remapped;
  }

  // Nodal coordinates on D0 domain (Hgrad) for RefMaxwell nullspace / mesh info.
  cntxt->refMaxwell.nodal_coords = Teuchos::rcp(
    new Tpetra::MultiVector<typename Teuchos::ScalarTraits<ScalarT>::coordinateType,LO,GO,Node>(nodal_map, dimension));
  auto coords_2d = cntxt->refMaxwell.nodal_coords->getLocalViewHost(Tpetra::Access::OverwriteAll);

  typedef Intrepid2::CellTools<PHX::Device::execution_space> AuxCellTools;
  typedef Intrepid2::FunctionSpaceTools<PHX::Device::execution_space> AuxFuncTools;

  // m_n = integral(N_n), accumulated on owned+ghosted then export-added.
  std::vector<GO> og_gids;
  hgrad_dof->getOwnedAndGhostedIndices(og_gids);
  Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > og_map =
    Teuchos::rcp(new Tpetra::Map<LO,GO,Node>(
      Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
      Teuchos::ArrayView<const GO>(og_gids), 0, mesh->comm));
  Teuchos::RCP<LA_MultiVector> mass_og = Teuchos::rcp(new LA_MultiVector(og_map, 1));
  mass_og->putScalar(0.0);
  auto mass_og_view = mass_og->getLocalViewHost(Tpetra::Access::ReadWrite);

  std::map<GO, std::vector<double> > gid_to_coords;
  for (size_t block = 0; block < mesh->block_names.size(); ++block) {
    const std::string block_name = mesh->block_names[block];
    const size_t num_elem = disc->my_elements[block].extent(0);
    // STK expects all-mesh element IDs, not block-local indices.
    vector<size_t> elem_ids(num_elem);
    for (size_t e = 0; e < num_elem; ++e) elem_ids[e] = disc->my_elements[block](e);
    DRV elem_nodes = mesh->getMyNodes(block, elem_ids);

    topo_RCP blockTopo = mesh->cell_topo[block];
    basis_RCP blockHgrad = disc->getBasis(dimension, blockTopo, "HGRAD", hgrad_order);
    DRV refPts, refWts;
    disc->getQuadrature(blockTopo, 2, refPts, refWts);
    const int nP = static_cast<int>(refPts.extent(0));
    const int nB = static_cast<int>(blockHgrad->getCardinality());
    DRV refVals("hgrad vals", nB, nP);
    blockHgrad->getValues(refVals, refPts, Intrepid2::OPERATOR_VALUE);
    const int nC = static_cast<int>(num_elem);
    DRV wts("wts", nC, nP);
    {
      DRV jac("jac", nC, nP, dimension, dimension), det("det", nC, nP);
      AuxCellTools::setJacobian(jac, refPts, elem_nodes, *blockTopo);
      AuxCellTools::setJacobianDet(det, jac);
      AuxFuncTools::computeCellMeasure(wts, det, refWts);
    }
    auto wts_h = Kokkos::create_mirror_view(wts);
    Kokkos::deep_copy(wts_h, wts);
    auto vals_h = Kokkos::create_mirror_view(refVals);
    Kokkos::deep_copy(vals_h, refVals);

    for (size_t e = 0; e < num_elem; ++e) {
      std::vector<GO> elem_dofs;
      LO local_elem_id = disc->my_elements[block](e);
      hgrad_dof->getElementGIDs(local_elem_id, elem_dofs, block_name);
      const size_t num_nodes = static_cast<size_t>(elem_nodes.extent(1));
      for (size_t n = 0; n < elem_dofs.size() && n < num_nodes; ++n) {
        if (gid_to_coords.find(elem_dofs[n]) == gid_to_coords.end()) {
          std::vector<double> coord(dimension, 0.0);
          for (int d = 0; d < dimension; ++d) coord[d] = elem_nodes(e, n, d);
          gid_to_coords[elem_dofs[n]] = coord;
        }
        const LO og_lid = og_map->getLocalElement(elem_dofs[n]);
        if (og_lid == Teuchos::OrdinalTraits<LO>::invalid()) continue;
        ScalarT acc = 0.0;
        for (int q = 0; q < nP; ++q) acc += wts_h(e, q) * vals_h(static_cast<int>(n), q);
        mass_og_view(og_lid, 0) += acc;
      }
    }
  }

  {
    mass_og_view = decltype(mass_og_view)();
    Teuchos::RCP<LA_MultiVector> mass_owned = Teuchos::rcp(new LA_MultiVector(nodal_map, 1));
    mass_owned->putScalar(0.0);
    Tpetra::Export<LO,GO,Node> og_to_owned(og_map, nodal_map);
    mass_owned->doExport(*mass_og, og_to_owned, Tpetra::ADD);
    ScalarT lmin = std::numeric_limits<ScalarT>::max(), gmin = 0.0;
    {
      auto mv = mass_owned->getLocalViewHost(Tpetra::Access::ReadOnly);
      for (size_t i = 0; i < mv.extent(0); ++i) lmin = std::min(lmin, mv(i,0));
    }
    Teuchos::reduceAll(*(nodal_map->getComm()), Teuchos::REDUCE_MIN, 1, &lmin, &gmin);
    TEUCHOS_TEST_FOR_EXCEPTION(gmin <= 0.0, std::runtime_error,
      "Lumped nodal mass has a non-positive entry (" << gmin << ").");
    // sum(m_n) is |Omega|. norm1 is collective, so every rank must call it.
    const ScalarT msum = mass_owned->getVector(0)->norm1();
    if (verbosity >= 5 && nodal_map->getComm()->getRank() == 0) {
      std::cout << "[AUX] lumped nodal mass: sum = " << std::setprecision(14)
                << msum << std::setprecision(6) << std::endl;
    }
    cntxt->refMaxwell.nodal_lumped_mass = mass_owned;
  }

  GO missing_coords = 0;
  for (LO lid = 0; lid < static_cast<LO>(nodal_map->getLocalNumElements()); ++lid) {
    GO gid = nodal_map->getGlobalElement(lid);
    auto it = gid_to_coords.find(gid);
    if (it == gid_to_coords.end()) { ++missing_coords; continue; }
    for (int d = 0; d < dimension; ++d) {
      coords_2d(lid, d) = it->second[d];
    }
  }
  {
    // Nodes left at the origin would silently corrupt distance-based aggregation.
    GO global_missing = 0;
    Teuchos::reduceAll<int, GO>(*(nodal_map->getComm()), Teuchos::REDUCE_SUM, 1,
                                &missing_coords, &global_missing);
    TEUCHOS_TEST_FOR_EXCEPTION(global_missing > 0, std::runtime_error,
      "RefMaxwell setup: " << global_missing << " nodes have no coordinates.");
  }

  if (cntxt->refMaxwell.block_dof_coords.size() != blockMaps.size()) {
    cntxt->refMaxwell.block_dof_coords.assign(blockMaps.size(), Teuchos::null);
  }
  auto edge_coords = buildEdgeAveragedNodeCoords<ScalarT,LO,GO,Node>(
    cntxt->refMaxwell.D0_matrix, edge_block_map, gid_to_coords, dimension);
  cntxt->refMaxwell.block_dof_coords[edgeBlock] = edge_coords;
  debugger->print("**** setupBlockTriangularAuxiliary: populated block_dof_coords[edgeBlock=" +
                  std::to_string(edgeBlock) + "] length " +
                  std::to_string(edge_coords->getGlobalLength()));


  
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

  
  auto myTableau = tableau[set];
  
  Kokkos::View<ScalarT**,HostDevice> tmp_butcher_A;
  Kokkos::View<ScalarT*,HostDevice> tmp_butcher_b, tmp_butcher_c;
  
  // only filling in the non-zero entries
  
  if (myTableau == "BWE" || myTableau == "DIRK-1,1") {
    tmp_butcher_A = Kokkos::View<ScalarT**,HostDevice>("butcher_A",1,1);
    tmp_butcher_A(0,0) = 1.0;
    tmp_butcher_b = Kokkos::View<ScalarT*,HostDevice>("butcher_b",1);
    tmp_butcher_b(0) = 1.0;
    tmp_butcher_c = Kokkos::View<ScalarT*,HostDevice>("butcher_c",1);
    tmp_butcher_c(0) = 1.0;
  }
  else if (myTableau == "FWE") {
    tmp_butcher_A = Kokkos::View<ScalarT**,HostDevice>("butcher_A",1,1);
    tmp_butcher_b = Kokkos::View<ScalarT*,HostDevice>("butcher_b",1);
    tmp_butcher_b(0) = 1.0;
    tmp_butcher_c = Kokkos::View<ScalarT*,HostDevice>("butcher_c",1);
  }
  else if (myTableau == "CN") {
    tmp_butcher_A = Kokkos::View<ScalarT**,HostDevice>("tmp_butcher_A",2,2);
    tmp_butcher_A(1,0) = 0.5;
    tmp_butcher_A(1,1) = 0.5;
    tmp_butcher_b = Kokkos::View<ScalarT*,HostDevice>("tmp_butcher_b",2);
    tmp_butcher_b(0) = 0.5;
    tmp_butcher_b(1) = 0.5;
    tmp_butcher_c = Kokkos::View<ScalarT*,HostDevice>("tmp_butcher_c",2);
    tmp_butcher_c(1) = 1.0;
  }
  else if (myTableau == "SSPRK-3,3") {
    tmp_butcher_A = Kokkos::View<ScalarT**,HostDevice>("tmp_butcher_A",3,3);
    tmp_butcher_A(1,0) = 1.0;
    tmp_butcher_A(2,0) = 0.25;
    tmp_butcher_A(2,1) = 0.25;
    tmp_butcher_b = Kokkos::View<ScalarT*,HostDevice>("tmp_butcher_b",3);
    tmp_butcher_b(0) = 1.0/6.0;
    tmp_butcher_b(1) = 1.0/6.0;
    tmp_butcher_b(2) = 2.0/3.0;
    tmp_butcher_c = Kokkos::View<ScalarT*,HostDevice>("tmp_butcher_c",3);
    tmp_butcher_c(1) = 1.0;
    tmp_butcher_c(2) = 1.0/2.0;
  }
  else if (myTableau == "RK-4,4") { // Classical RK4
    tmp_butcher_A = Kokkos::View<ScalarT**,HostDevice>("tmp_butcher_A",4,4);
    tmp_butcher_A(1,0) = 0.5;
    tmp_butcher_A(2,1) = 0.5;
    tmp_butcher_A(3,2) = 1.0;
    tmp_butcher_b = Kokkos::View<ScalarT*,HostDevice>("tmp_butcher_b",4);
    tmp_butcher_b(0) = 1.0/6.0;
    tmp_butcher_b(1) = 1.0/3.0;
    tmp_butcher_b(2) = 1.0/3.0;
    tmp_butcher_b(3) = 1.0/6.0;
    tmp_butcher_c = Kokkos::View<ScalarT*,HostDevice>("tmp_butcher_c",4);
    tmp_butcher_c(1) = 1.0/2.0;
    tmp_butcher_c(2) = 1.0/2.0;
    tmp_butcher_c(3) = 1.0;
  }
  else if (myTableau == "DIRK-1,2") {
    tmp_butcher_A = Kokkos::View<ScalarT**,HostDevice>("tmp_butcher_A",1,1);
    tmp_butcher_A(0,0) = 0.5;
    tmp_butcher_b = Kokkos::View<ScalarT*,HostDevice>("tmp_butcher_b",1);
    tmp_butcher_b(0) = 1.0;
    tmp_butcher_c = Kokkos::View<ScalarT*,HostDevice>("tmp_butcher_c",1);
    tmp_butcher_c(0) = 0.5;
  }
  else if (myTableau == "DIRK-2,2") { // 2-stage, 2nd order
    tmp_butcher_A = Kokkos::View<ScalarT**,HostDevice>("tmp_butcher_A",2,2);
    tmp_butcher_A(0,0) = 1.0/4.0;
    tmp_butcher_A(1,0) = 1.0/2.0;
    tmp_butcher_A(1,1) = 1.0/4.0;
    tmp_butcher_b = Kokkos::View<ScalarT*,HostDevice>("tmp_butcher_b",2);
    tmp_butcher_b(0) = 1.0/2.0;
    tmp_butcher_b(1) = 1.0/2.0;
    tmp_butcher_c = Kokkos::View<ScalarT*,HostDevice>("tmp_butcher_c",2);
    tmp_butcher_c(0) = 1.0/4.0;
    tmp_butcher_c(1) = 3.0/4.0;
  }
  else if (myTableau == "DIRK-2,3") { // 2-stage, 3rd order
    tmp_butcher_A = Kokkos::View<ScalarT**,HostDevice>("tmp_butcher_A",2,2);
    tmp_butcher_A(0,0) = 1.0/2.0 + std::sqrt(3)/6.0;
    tmp_butcher_A(1,0) = -std::sqrt(3)/3.0;
    tmp_butcher_A(1,1) = 1.0/2.0  + std::sqrt(3)/6.0;
    tmp_butcher_b = Kokkos::View<ScalarT*,HostDevice>("tmp_butcher_b",2);
    tmp_butcher_b(0) = 1.0/2.0;
    tmp_butcher_b(1) = 1.0/2.0;
    tmp_butcher_c = Kokkos::View<ScalarT*,HostDevice>("tmp_butcher_c",2);
    tmp_butcher_c(0) = 1.0/2.0 + std::sqrt(3)/6.0;;
    tmp_butcher_c(1) = 1.0/2.0 - std::sqrt(3)/6.0;;
  }
  else if (myTableau == "DIRK-3,3") { // 3-stage, 3rd order
    ScalarT p = 0.4358665215;
    tmp_butcher_A = Kokkos::View<ScalarT**,HostDevice>("tmp_butcher_A",3,3);
    tmp_butcher_A(0,0) = p;
    tmp_butcher_A(1,0) = (1.0-p)/2.0;
    tmp_butcher_A(1,1) = p;
    tmp_butcher_A(2,0) = -3.0*p*p/2.0+4.0*p-1.0/4.0;
    tmp_butcher_A(2,1) = 3.0*p*p/2.0 - 5.0*p + 5.0/4.0;
    tmp_butcher_A(2,2) = p;
    tmp_butcher_b = Kokkos::View<ScalarT*,HostDevice>("tmp_butcher_b",3);
    tmp_butcher_b(0) = -3.0*p*p/2.0+4.0*p-1.0/4.0;
    tmp_butcher_b(1) = 3.0*p*p/2.0-5.0*p+5.0/4.0;
    tmp_butcher_b(2) = p;
    tmp_butcher_c = Kokkos::View<ScalarT*,HostDevice>("tmp_butcher_c",3);
    tmp_butcher_c(0) = p;
    tmp_butcher_c(1) = (1.0+p)/2.0;
    tmp_butcher_c(2) = 1.0;
  }
  else if (myTableau == "leap-frog") { // Leap-frog for Maxwells
    tmp_butcher_A = Kokkos::View<ScalarT**,HostDevice>("tmp_butcher_A",2,2);
    tmp_butcher_A(1,0) = 1.0;
    tmp_butcher_b = Kokkos::View<ScalarT*,HostDevice>("tmp_butcher_b",2);
    tmp_butcher_b(0) = 1.0;
    tmp_butcher_b(1) = 1.0;
    tmp_butcher_c = Kokkos::View<ScalarT*,HostDevice>("tmp_butcher_c",2);
    tmp_butcher_c(0) = 0.0;
    tmp_butcher_c(1) = 0.0;
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
    tmp_butcher_A = Kokkos::View<ScalarT**,HostDevice>("tmp_butcher_A",A_nrows,A_nrows);
    tmp_butcher_b = Kokkos::View<ScalarT*,HostDevice>("tmp_butcher_b",A_nrows);
    tmp_butcher_c = Kokkos::View<ScalarT*,HostDevice>("tmp_butcher_c",A_nrows);
    for (size_t i=0; i<A_nrows; i++) {
      for (size_t j=0; j<A_nrows; j++) {
        tmp_butcher_A(i,j) = A_vals[i][j];
      }
      tmp_butcher_b(i) = b_vals[i];
      tmp_butcher_c(i) = c_vals[i];
    }
    
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: unrecognized Butcher tableau:" + tableau[set]);
  }
  Kokkos::View<ScalarT**,AssemblyDevice> dev_butcher_A("butcher_A on device",tmp_butcher_A.extent(0),tmp_butcher_A.extent(1));
  Kokkos::View<ScalarT*,AssemblyDevice> dev_butcher_b("butcher_b on device",tmp_butcher_b.extent(0));
  Kokkos::View<ScalarT*,AssemblyDevice> dev_butcher_c("butcher_c on device",tmp_butcher_c.extent(0));
  
  auto tmp2_butcher_A = Kokkos::create_mirror_view(dev_butcher_A);
  auto tmp2_butcher_b = Kokkos::create_mirror_view(dev_butcher_b);
  auto tmp2_butcher_c = Kokkos::create_mirror_view(dev_butcher_c);
  
  Kokkos::deep_copy(tmp2_butcher_A, tmp_butcher_A);
  Kokkos::deep_copy(tmp2_butcher_b, tmp_butcher_b);
  Kokkos::deep_copy(tmp2_butcher_c, tmp_butcher_c);
  
  Kokkos::deep_copy(dev_butcher_A, tmp_butcher_A);
  Kokkos::deep_copy(dev_butcher_b, tmp_butcher_b);
  Kokkos::deep_copy(dev_butcher_c, tmp_butcher_c);
  
  int newnumstages = tmp_butcher_A.extent(0);
  
  maxnumstages[set] = std::max(numstages[set],newnumstages);
  numstages[set] = newnumstages;
  
  for (size_t block=0; block<assembler->groups.size(); ++block) {
    assembler->setWorksetButcher(set, block, dev_butcher_A, dev_butcher_b, dev_butcher_c);
  }
  
  if (butcher_A.size() > set) {
    butcher_A[set] = tmp_butcher_A;
    butcher_b[set] = tmp_butcher_b;
    butcher_c[set] = tmp_butcher_c;
  }
  else {
    butcher_A.push_back(tmp_butcher_A);
    butcher_b.push_back(tmp_butcher_b);
    butcher_c.push_back(tmp_butcher_c);
  }
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
void SolverManager<Node>::setBackwardDifference(const vector<int> & order, const int & set) { // using order as an input to allow for dynamic changes
  
  
  Kokkos::View<ScalarT*,AssemblyDevice> dev_BDF_wts;
  Kokkos::View<ScalarT*,HostDevice> tmp_BDF_wts;
  
  // Note that these do not include 1/deltat (added in wkset)
  // Not going to work properly for adaptive time stepping if BDForder>1
  
  auto myOrder = order[set];
  
  if (isTransient) {
    
    if (myOrder == 1) {
      tmp_BDF_wts = Kokkos::View<ScalarT*,HostDevice>("tmp_BDF weights to compute u_dot",2);
      tmp_BDF_wts(0) = 1.0;
      tmp_BDF_wts(1) = -1.0;
    }
    else if (myOrder == 2) {
      tmp_BDF_wts = Kokkos::View<ScalarT*,HostDevice>("tmp_BDF weights to compute u_dot",3);
      tmp_BDF_wts(0) = 1.5;
      tmp_BDF_wts(1) = -2.0;
      tmp_BDF_wts(2) = 0.5;
    }
    else if (myOrder == 3) {
      tmp_BDF_wts = Kokkos::View<ScalarT*,HostDevice>("tmp_BDF weights to compute u_dot",4);
      tmp_BDF_wts(0) = 11.0/6.0;
      tmp_BDF_wts(1) = -3.0;
      tmp_BDF_wts(2) = 1.5;
      tmp_BDF_wts(3) = -1.0/3.0;
    }
    else if (myOrder == 4) {
      tmp_BDF_wts = Kokkos::View<ScalarT*,HostDevice>("tmp_BDF weights to compute u_dot",5);
      tmp_BDF_wts(0) = 25.0/12.0;
      tmp_BDF_wts(1) = -4.0;
      tmp_BDF_wts(2) = 3.0;
      tmp_BDF_wts(3) = -4.0/3.0;
      tmp_BDF_wts(4) = 1.0/4.0;
    }
    else if (myOrder == 5) {
      tmp_BDF_wts = Kokkos::View<ScalarT*,HostDevice>("tmp_BDF weights to compute u_dot",6);
      tmp_BDF_wts(0) = 137.0/60.0;
      tmp_BDF_wts(1) = -5.0;
      tmp_BDF_wts(2) = 5.0;
      tmp_BDF_wts(3) = -10.0/3.0;
      tmp_BDF_wts(4) = 75.0/60.0;
      tmp_BDF_wts(5) = -1.0/5.0;
    }
    else if (myOrder == 6) {
      tmp_BDF_wts = Kokkos::View<ScalarT*,HostDevice>("tmp_BDF weights to compute u_dot",7);
      tmp_BDF_wts(0) = 147.0/60.0;
      tmp_BDF_wts(1) = -6.0;
      tmp_BDF_wts(2) = 15.0/2.0;
      tmp_BDF_wts(3) = -20.0/3.0;
      tmp_BDF_wts(4) = 225.0/60.0;
      tmp_BDF_wts(5) = -72.0/60.0;
      tmp_BDF_wts(6) = 1.0/6.0;
    }
    
    int newnumsteps = tmp_BDF_wts.extent(0)-1;
    
    maxnumsteps[set] = std::max(maxnumsteps[set],newnumsteps);
    numsteps[set] = newnumsteps;
    
  }
  else { // for steady state solves, u_dot = 0.0*u
    tmp_BDF_wts = Kokkos::View<ScalarT*,HostDevice>("tmp_BDF weights to compute u_dot",1);
    tmp_BDF_wts(0) = 1.0;
    numsteps[set] = 1;
    maxnumsteps[set] = 1;
  }
  
  dev_BDF_wts = Kokkos::View<ScalarT*,AssemblyDevice>("BDF weights on device",tmp_BDF_wts.extent(0));
  Kokkos::deep_copy(dev_BDF_wts, tmp_BDF_wts);
  
  for (size_t block=0; block<assembler->groups.size(); ++block) {
    assembler->setWorksetBDF(set, block, dev_BDF_wts);
  } // end loop blocks
  
  if (BDF_wts.size() > set) {
    BDF_wts[set] = tmp_BDF_wts;
  }
  else {
    BDF_wts.push_back(tmp_BDF_wts);
  }
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
      
      if (mesh->getPhaseDimension() > 0) {
        for (size_t set=0; set<physics->set_names.size(); set++) {
          vector<vector<int> > voffsets = disc->phase_offsets[set][0];
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
          wkset[block]->phase_set_offsets.push_back(offsets_view);
          if (set == 0) {
            wkset[block]->phase_offsets = offsets_view;
          }

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
      
      if (mesh->getPhaseDimension() > 0) {
        vector<vector<int> > block_useBasis;
        vector<vector<string> > block_varlist;
        for (size_t set=0; set<phase_useBasis.size(); ++set) {
          block_useBasis.push_back(phase_useBasis[set][0]);
          block_varlist.push_back(varlist[set][0]);
        }
        wkset[block]->phase_set_usebasis = block_useBasis;
        wkset[block]->phase_set_varlist = block_varlist;
        wkset[block]->phase_usebasis = block_useBasis[0];
        wkset[block]->phase_varlist = block_varlist[0];
        
      }
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
