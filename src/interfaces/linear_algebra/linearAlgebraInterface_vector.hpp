/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
************************************************************************/

template<class Node>
Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > LinearAlgebraInterface<Node>::getNewVector(const size_t & set, const int & numvecs) {
  Teuchos::TimeMonitor vectimer(*newvectortimer);
  vector_RCP newvec = Teuchos::rcp(new LA_MultiVector(owned_map[set],numvecs));
  return newvec;
}

// ========================================================================================
// ========================================================================================

template<class Node>
Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > LinearAlgebraInterface<Node>::getNewOverlappedVector(const size_t & set, const int & numvecs){
  Teuchos::TimeMonitor vectimer(*newovervectortimer);
  vector_RCP newvec;
  if (have_overlapped) {
    newvec = Teuchos::rcp(new LA_MultiVector(overlapped_map[set],numvecs));
  }
  else {
    newvec = Teuchos::rcp(new LA_MultiVector(owned_map[set],numvecs));
  }
  return newvec;
}

// ========================================================================================
// ========================================================================================

template<class Node>
Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > LinearAlgebraInterface<Node>::getNewParamVector(const int & numvecs) {
  Teuchos::TimeMonitor vectimer(*newvectortimer);
  vector_RCP newvec = Teuchos::rcp(new LA_MultiVector(param_owned_map, numvecs));
  return newvec;
}

// ========================================================================================
// ========================================================================================

template<class Node>
Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > LinearAlgebraInterface<Node>::getNewOverlappedParamVector(const int & numvecs) {
  Teuchos::TimeMonitor vectimer(*newvectortimer);
  vector_RCP newvec = Teuchos::rcp(new LA_MultiVector(param_overlapped_map, numvecs));
  return newvec;
}

// ========================================================================================
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::exportVectorFromOverlapped(const size_t & set, vector_RCP & vec, vector_RCP & vec_over) {
  Teuchos::TimeMonitor mattimer(*exporttimer);
  if (comm->getSize() > 1) {
    vec->putScalar(0.0);
    vec->doExport(*vec_over, *(/** @brief Exporters for owned→overlapped communication. */
                               exporter[set]), Tpetra::ADD);
  }
  else {
    vec->assign(*vec_over);
  }
}

// ========================================================================================
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::exportVectorFromOverlappedReplace(const size_t & set, vector_RCP & vec, vector_RCP & vec_over) {
  Teuchos::TimeMonitor mattimer(*exporttimer);
  if (comm->getSize() > 1) {
    vec->putScalar(0.0);
    vec->doExport(*vec_over, *(exporter[set]), Tpetra::REPLACE);
  }
  else {
    vec->assign(*vec_over);
  }
}

// ========================================================================================
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::exportParamVectorFromOverlapped(vector_RCP & vec, vector_RCP & vec_over) {
  Teuchos::TimeMonitor mattimer(*exporttimer);
  vec->putScalar(0.0);
  vec->doExport(*vec_over, *param_exporter, Tpetra::ADD);
}

// ========================================================================================
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::exportParamVectorFromOverlappedReplace(vector_RCP & vec, vector_RCP & vec_over) {
  Teuchos::TimeMonitor mattimer(*exporttimer);
  vec->putScalar(0.0);
  vec->doExport(*vec_over, *param_exporter, Tpetra::REPLACE);
}

// ========================================================================================
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::importVectorToOverlapped(const size_t & set, vector_RCP & vec_over, const vector_RCP & vec) {
  Teuchos::TimeMonitor mattimer(*importtimer);
  vec_over->putScalar(0.0);
  vec_over->doImport(*vec, *(/** @brief Importers for overlapped→owned communication. */
                             importer[set]), Tpetra::ADD);
}

// ========================================================================================
// ========================================================================================


