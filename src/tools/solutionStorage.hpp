/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef SOLSTORAGE_H
#define SOLSTORAGE_H

#include "trilinos.hpp"
#include "preferences.hpp"
// Add includes for PyTorch

class SolutionStorage {
public:
  
  SolutionStorage() {} ;
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
  SolutionStorage(const Teuchos::RCP<Teuchos::ParameterList> & settings) {
  
    // Maximum number of vectors to store
    maxStorage = settings->sublist("Solver").get<LO>("maximum storage", 100);
    timeTOL = settings->sublist("Solver").get<ScalarT>("storage time tol", 1.0e-13);
    
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////

  bool extract(Teuchos::RCP<Epetra_MultiVector> & vec, const size_t & index, const ScalarT & currtime) {
    Teuchos::TimeMonitor localtimer(*solnStorageExtractTimer);
    bool found = false;
    for (size_t j=0; j<times[index].size(); j++) {
      if (abs(times[index][j] - currtime) < timeTOL) {
        vec = data[index][j];
        found = true;
      }
    }
    return found;
  }

  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
  bool extract(Teuchos::RCP<Epetra_MultiVector> & vec, const size_t & index, const ScalarT & currtime, int & timeindex) {
    Teuchos::TimeMonitor localtimer(*solnStorageExtractTimer);
    bool found = false;
    for (size_t j=0; j<times[index].size(); j++) {
      if (abs(times[index][j] - currtime) < timeTOL) {
        vec = data[index][j];
        timeindex = j;
        found = true;
      }
    }
    return found;
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
  bool extractPrevious(Teuchos::RCP<Epetra_MultiVector> & vec, const size_t & index, const ScalarT & currtime, ScalarT & prevtime) {
    Teuchos::TimeMonitor localtimer(*solnStorageExtractTimer);
    bool found = false;
    for (size_t j=0; j<times[index].size(); j++) {
      if (abs(times[index][j] - currtime) < timeTOL) {
        found = true;
        if (j>0) {
          vec = data[index][j-1];
          prevtime = times[index][j-1];
        }
        else {
          vec = data[index][j];
          prevtime = times[index][j];
        }
      }
    }
    return found;
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
  bool extractNext(Teuchos::RCP<Epetra_MultiVector> & vec, const size_t & index, const ScalarT & currtime, ScalarT & nexttime) {
    Teuchos::TimeMonitor localtimer(*solnStorageExtractTimer);
    bool found = false;
    for (size_t j=0; j<times[index].size(); j++) {
      if (abs(times[index][j] - currtime) < timeTOL) {
        found = true;
        if (j<(times[index].size()-1)) {
          vec = data[index][j+1];
          nexttime = times[index][j+1];
        }
        else {
          vec = data[index][j];
          nexttime = times[index][j];
        }
      }
    }
    return found;
  }

  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
  bool extractLast(Teuchos::RCP<Epetra_MultiVector> & vec, const size_t & index, ScalarT & lasttime) {
    Teuchos::TimeMonitor localtimer(*solnStorageExtractTimer);
    bool found = true;
    vec = data[index][times[index].size()-1];
    lasttime = times[index][times[index].size()-1];
    return found;
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////

  void store(Teuchos::RCP<Epetra_MultiVector> & newvec, const ScalarT & currtime, const LO & index) {
    
    Teuchos::TimeMonitor localtimer(*solnStorageStoreTimer);
    
    Teuchos::RCP<Epetra_MultiVector> vecstore = Teuchos::rcp( new Epetra_MultiVector(*newvec));
    
    // Deep copy of data
    *vecstore = *newvec;
    
    if (times.size() <= index) {
      vector<ScalarT> newtime = {currtime};
      vector<Teuchos::RCP<Epetra_MultiVector> > newdata = {vecstore};
      times.push_back(newtime);
      data.push_back(newdata);
    }
    else {
      size_t timeindex;
      bool foundtime = false;
      for (size_t j=0; j<times[index].size(); j++) {
        if (abs(times[index][j] - currtime) < timeTOL) {
          foundtime = true;
          timeindex = j;
        }
      }
      if (foundtime) {
        data[index][timeindex] = vecstore;
      }
      else {
        data[index].push_back(vecstore);
        times[index].push_back(currtime);
      }
    }
    
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void trainDNN() {
    Teuchos::TimeMonitor localtimer(*solnStorageTrainDNNTimer);
    
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void useDNN() {
    Teuchos::TimeMonitor localtimer(*solnStorageUseDNNTimer);
    
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////

  LO maxStorage;
  ScalarT timeTOL;
  
  vector<vector<ScalarT> > times;
  vector<vector<Teuchos::RCP<Epetra_MultiVector> > > data;
  
  // Additional data needed for ML
  vector<vector<Kokkos::View<ScalarT*,AssemblyDevice> > > inputs;
  vector<vector<Kokkos::View<AD*,AssemblyDevice> > > outputs;
  
  // Add data structures for PyTorch
  
  // Timers
  Teuchos::RCP<Teuchos::Time> solnStorageStoreTimer = Teuchos::TimeMonitor::getNewCounter("MILO::SolutionStorage::store");
  Teuchos::RCP<Teuchos::Time> solnStorageExtractTimer = Teuchos::TimeMonitor::getNewCounter("MILO::SolutionStorage::extract");
  Teuchos::RCP<Teuchos::Time> solnStorageTrainDNNTimer = Teuchos::TimeMonitor::getNewCounter("MILO::SolutionStorage::trainDNN");
  Teuchos::RCP<Teuchos::Time> solnStorageUseDNNTimer = Teuchos::TimeMonitor::getNewCounter("MILO::SolutionStorage::useDNN");
  
  
  
};

#endif
