/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
************************************************************************/

// ========================================================================================
// ========================================================================================

template<class Node>
Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,Node> > LinearAlgebraInterface<Node>::getNewMatrix(const size_t & set) {
  Teuchos::TimeMonitor mattimer(*newmatrixtimer);
  matrix_RCP newmat;
  if (options[set]->reuse_jacobian) {
    if (options[set]->have_jacobian) {
      newmat = options[set]->jac;
    }
    else {
      newmat = Teuchos::rcp(new LA_CrsMatrix(owned_map[set], max_entries));
      options[set]->jac = newmat;
    }
  }
  else {
    newmat = Teuchos::rcp(new LA_CrsMatrix(owned_map[set], max_entries));
  }
  return newmat;
}

// ========================================================================================
// ========================================================================================

template<class Node>
vector<Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,Node> > > LinearAlgebraInterface<Node>::getNewPreviousMatrix(const size_t & set,
                                                                                                                 const size_t & numsteps) {
  Teuchos::TimeMonitor mattimer(*newmatrixtimer);
  vector<matrix_RCP> newmat;
  if (options[set]->reuse_jacobian) {
    if (options[set]->have_previous_jacobian) {
      newmat = options[set]->jac_prev;
    }
    else {
      for (size_t k=0; k<numsteps; ++k) {
        matrix_RCP M = Teuchos::rcp(new LA_CrsMatrix(owned_map[set], max_entries));
        newmat.push_back(M);
      }
      options[set]->jac_prev = newmat;
      options[set]->have_previous_jacobian = true;
    }
  }
  else {
    for (size_t k=0; k<numsteps; ++k) {
      matrix_RCP M = Teuchos::rcp(new LA_CrsMatrix(owned_map[set], max_entries));
      newmat.push_back(M);
    }
  }
  return newmat;
}

// ========================================================================================
// ========================================================================================

template<class Node>
Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,Node> > LinearAlgebraInterface<Node>::getNewParamMatrix() {
  Teuchos::TimeMonitor mattimer(*newmatrixtimer);
  matrix_RCP newmat = Teuchos::rcp(new LA_CrsMatrix(param_owned_map, max_entries));
  return newmat;
}

// ========================================================================================
// ========================================================================================

template<class Node>
Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,Node> > LinearAlgebraInterface<Node>::getNewParamStateMatrix(const size_t & set) {
  Teuchos::TimeMonitor mattimer(*newmatrixtimer);
  matrix_RCP newmat = Teuchos::rcp(new LA_CrsMatrix(param_owned_map, max_entries));
  return newmat;
}

// ========================================================================================
// ========================================================================================

template<class Node>
Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,Node> > LinearAlgebraInterface<Node>::getNewMatrix(const size_t & set, vector<size_t> & maxent) {
  Teuchos::TimeMonitor mattimer(*newmatrixtimer);
  matrix_RCP newmat = Teuchos::rcp(new LA_CrsMatrix(owned_map[set], maxent));
  return newmat;
}

// ========================================================================================
// ========================================================================================

template<class Node>
Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,Node> > LinearAlgebraInterface<Node>::getNewOverlappedMatrix(const size_t & set) {
  Teuchos::TimeMonitor mattimer(*newmatrixtimer);
  matrix_RCP newmat = Teuchos::rcp(new LA_CrsMatrix(/** @brief Overlapped CRS graphs (owned graphs unused). */
                                                    overlapped_graph[set]));
  return newmat;
}

// ========================================================================================
// ========================================================================================

template<class Node>
Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,Node> > LinearAlgebraInterface<Node>::getNewOverlappedRectangularMatrix(Teuchos::RCP<const LA_Map> & colmap,
                                                                                                                     const size_t & set) {
  Teuchos::TimeMonitor mattimer(*newmatrixtimer);
  matrix_RCP newmat = Teuchos::rcp(new LA_CrsMatrix(/** @brief Overlapped maps for each set. */
                                                    overlapped_map[set], colmap, max_entries));
  return newmat;
}

// ========================================================================================
// ========================================================================================

template<class Node>
Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,Node> > LinearAlgebraInterface<Node>::getNewRectangularMatrix(Teuchos::RCP<const LA_Map> & colmap,
                                                                                                           const size_t & set) {
  Teuchos::TimeMonitor mattimer(*newmatrixtimer);
  matrix_RCP newmat = Teuchos::rcp(new LA_CrsMatrix(/** @brief Owned maps for each set. */
                                                    owned_map[set], colmap, max_entries));
  return newmat;
}

// ========================================================================================
// ========================================================================================

template<class Node>
Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,Node> > LinearAlgebraInterface<Node>::getNewOverlappedParamMatrix() {
  Teuchos::TimeMonitor mattimer(*newmatrixtimer);
  matrix_RCP newmat = Teuchos::rcp(new LA_CrsMatrix(param_overlapped_graph));
  return newmat;
}

// ========================================================================================
// ========================================================================================

template<class Node>
Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,Node> > LinearAlgebraInterface<Node>::getNewOverlappedParamStateMatrix(const size_t & set) {
  Teuchos::TimeMonitor mattimer(*newmatrixtimer);
  matrix_RCP newmat = Teuchos::rcp(new LA_CrsMatrix(paramstate_overlapped_graph[set]));
  return newmat;
}

// ========================================================================================
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::exportMatrixFromOverlapped(const size_t & set, matrix_RCP & mat, matrix_RCP & mat_over) {
  Teuchos::TimeMonitor mattimer(*exporttimer);
  mat->setAllToScalar(0.0);
  mat->doExport(*mat_over, *(exporter[set]), Tpetra::ADD);
}

// ========================================================================================
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::exportParamStateMatrixFromOverlapped(const size_t & set, matrix_RCP & mat, matrix_RCP & mat_over) {
  Teuchos::TimeMonitor mattimer(*exporttimer);
  mat->setAllToScalar(0.0);
  mat->doExport(*mat_over, *(param_exporter), Tpetra::ADD);
}

// ========================================================================================
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::exportParamMatrixFromOverlapped(matrix_RCP & mat, matrix_RCP & mat_over) {
  Teuchos::TimeMonitor mattimer(*exporttimer);
  mat->setAllToScalar(0.0);
  mat->doExport(*mat_over, *param_exporter, Tpetra::ADD);
}

// ========================================================================================
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::fillCompleteParamState(const size_t & set, matrix_RCP & mat) {
  Teuchos::TimeMonitor mattimer(*fillcompletetimer);
  mat->fillComplete(owned_map[set], param_owned_map);
}

// ========================================================================================
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::fillComplete(matrix_RCP & mat) {
  Teuchos::TimeMonitor mattimer(*fillcompletetimer);
  mat->fillComplete();
}
