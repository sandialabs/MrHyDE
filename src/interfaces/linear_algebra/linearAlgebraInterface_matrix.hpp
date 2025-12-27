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
  if (context[set]->reuse_matrix) {
    if (context[set]->have_matrix) {
      newmat = context[set]->matrix;
    }
    else {
      newmat = Teuchos::rcp(new LA_CrsMatrix(owned_map[set], max_entries));
      context[set]->matrix = newmat;
      context[set]->have_matrix = true;
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
Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,Node> > LinearAlgebraInterface<Node>::getNewL2Matrix(const size_t & set) {
  Teuchos::TimeMonitor mattimer(*newmatrixtimer);
  matrix_RCP newmat;
  if (context_L2[set]->reuse_matrix) {
    if (context_L2[set]->have_matrix) {
      newmat = context_L2[set]->matrix;
    }
    else {
      newmat = Teuchos::rcp(new LA_CrsMatrix(owned_map[set], max_entries));
      context_L2[set]->matrix = newmat;
      context_L2[set]->have_matrix = true;
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
Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,Node> > LinearAlgebraInterface<Node>::getNewBndryL2Matrix(const size_t & set) {
  Teuchos::TimeMonitor mattimer(*newmatrixtimer);
  matrix_RCP newmat;
  if (context_BndryL2[set]->reuse_matrix) {
    if (context_BndryL2[set]->have_matrix) {
      newmat = context_BndryL2[set]->matrix;
    }
    else {
      newmat = Teuchos::rcp(new LA_CrsMatrix(owned_map[set], max_entries));
      context_BndryL2[set]->matrix = newmat;
      context_BndryL2[set]->have_matrix = true;
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
  
  // The context objects for the previous Jacobians/matrices only get allocated if this gets called
  if (context_prev.size() == 0) {
    size_t numsets = context.size();
    for (size_t step=0; step<numsteps; ++step) {
      vector<Teuchos::RCP<LinearSolverContext<Node> > > currcontext;
      for (size_t st=0; st<numsets; ++st) {
        // Create the solver Context for the previous state Jacobians
        // Reusing whatever if defined for state Jacobians
        // Could be generalized, but unlikely to be needed
        for (size_t set=0; set<setnames.size(); ++set) {
          Teuchos::ParameterList solvesettings;
          if (settings->sublist("Solver").isSublist("State linear solver")) { // for detailed control
            solvesettings = settings->sublist("Solver").sublist("State linear solver");
          }
          else { // use generic Context
            solvesettings = settings->sublist("Solver");
          }
          currcontext.push_back(Teuchos::rcp( new LinearSolverContext<Node>(solvesettings) ));
        }
      }
      context_prev.push_back(currcontext);
    }
  }
  
  for (size_t k=0; k<numsteps; ++k) {
    if (context_prev[k][set]->reuse_matrix) {
      if (context_prev[k][set]->have_matrix) {
        newmat.push_back(context_prev[k][set]->matrix);
      }
      else {
        matrix_RCP M = Teuchos::rcp(new LA_CrsMatrix(owned_map[set], max_entries));
        newmat.push_back(M);
        context_prev[k][set]->matrix = M;
      }
    }
    else {
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
  matrix_RCP newmat;
  if (context_param_state[set]->reuse_matrix) {
    if (context_param_state[set]->have_matrix) {
      newmat = context_param_state[set]->matrix;
    }
    else {
      newmat = Teuchos::rcp(new LA_CrsMatrix(param_owned_map, max_entries));
      context_param_state[set]->matrix = newmat;
      context_param_state[set]->have_matrix = true;
    }
  }
  else {
    newmat = Teuchos::rcp(new LA_CrsMatrix(param_owned_map, max_entries));
  }
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
