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

#ifndef SOLSTORAGE_H
#define SOLSTORAGE_H

#include "trilinos.hpp"
#include "preferences.hpp"
// Add includes for PyTorch


namespace MrHyDE {
  
  template<class Node>
  class SolutionStorage {
    
    typedef Tpetra::MultiVector<ScalarT,LO,GO,Node> V;
    typedef typename Node::device_type V_device;
    
  public:
    
    SolutionStorage() {} ;
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    SolutionStorage(const Teuchos::RCP<Teuchos::ParameterList> & settings) {
      
      // Maximum number of vectors to store (not used right now)
      maxStorage = settings->sublist("Solver").get<LO>("maximum storage", 100);
      // Relative tolerance for storing/extraction solution at given time
      timeRelTOL = settings->sublist("Solver").get<ScalarT>("storage time tol", 1.0e-10);
      
    }
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    bool extract(Teuchos::RCP<V> & vec, const size_t & timeindex) {
      // defaults to index = 0
      // most common use case
      Teuchos::TimeMonitor localtimer(*solnStorageExtractTimer);
      bool found = false;
      if (data.size()>0) {
        if (data[0].size()>timeindex) {
          vec = data[0][timeindex];
          found = true;
        }
      }
      return found;
    }
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    bool extract(Teuchos::RCP<V> & vec, const size_t & timeindex, const size_t & index) {
      // defaults to index = 0
      // most common use case
      Teuchos::TimeMonitor localtimer(*solnStorageExtractTimer);
      bool found = false;
      if (data.size()>index) {
        if (data[index].size()>timeindex) {
          vec = data[index][timeindex];
          found = true;
        }
      }
      return found;
    }
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    bool extract(Teuchos::RCP<V> & vec, const size_t & index, const ScalarT & currtime) {
      Teuchos::TimeMonitor localtimer(*solnStorageExtractTimer);
      bool found = false;
      if (abs(currtime)>1.0e-16) { // ok to use relative tolerance
        for (size_t j=0; j<times[index].size(); j++) {
          if (abs(times[index][j] - currtime)/currtime < timeRelTOL) {
            vec = data[index][j];
            found = true;
          }
        }
      }
      else { // use as absolute tolerance
        for (size_t j=0; j<times[index].size(); j++) {
          if (abs(times[index][j] - currtime) < timeRelTOL) {
            vec = data[index][j];
            found = true;
          }
        }
      }
      return found;
    }
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    bool extract(Teuchos::RCP<V> & vec, const size_t & index, const ScalarT & currtime, int & timeindex) {
      Teuchos::TimeMonitor localtimer(*solnStorageExtractTimer);
      bool found = false;
      
      if (abs(currtime)>1.0e-16) { // ok to use relative tolerance
        for (size_t j=0; j<times[index].size(); j++) {
          if (abs(times[index][j] - currtime)/currtime < timeRelTOL) {
            vec = data[index][j];
            timeindex = j;
            found = true;
          }
        }
      }
      else {
        for (size_t j=0; j<times[index].size(); j++) {
          if (abs(times[index][j] - currtime) < timeRelTOL) {
            vec = data[index][j];
            timeindex = j;
            found = true;
          }
        }
      }
      return found;
    }
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    bool extractPrevious(Teuchos::RCP<V> & vec, const size_t & index, const ScalarT & currtime, ScalarT & prevtime) {
      Teuchos::TimeMonitor localtimer(*solnStorageExtractTimer);
      bool found = false;
      for (size_t j=0; j<times[index].size(); j++) {
        if (abs(times[index][j] - currtime)/currtime < timeRelTOL) {
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
    
    bool extractNext(Teuchos::RCP<V> & vec, const size_t & index, const ScalarT & currtime, ScalarT & nexttime) {
      Teuchos::TimeMonitor localtimer(*solnStorageExtractTimer);
      bool found = false;
      for (size_t j=0; j<times[index].size(); j++) {
        if (abs(times[index][j] - currtime)/currtime < timeRelTOL) {
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
    
    bool extractLast(Teuchos::RCP<V> & vec, const size_t & index, ScalarT & lasttime) {
      Teuchos::TimeMonitor localtimer(*solnStorageExtractTimer);
      bool found = true;
      vec = data[index][times[index].size()-1];
      lasttime = times[index][times[index].size()-1];
      return found;
    }
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void store(Teuchos::RCP<V> & newvec, const ScalarT & currtime, const size_t & index) {
      
      Teuchos::TimeMonitor localtimer(*solnStorageStoreTimer);
      
      // Deep copy of data
      Teuchos::RCP<V> vecstore = copyData(newvec);
      
      
      if (times.size() <= index) {
        vector<ScalarT> newtime = {currtime};
        vector<Teuchos::RCP<V> > newdata = {vecstore};
        times.push_back(newtime);
        data.push_back(newdata);
      }
      else {
        size_t timeindex;
        bool foundtime = false;
        for (size_t j=0; j<times[index].size(); j++) {
          if (abs(currtime)>1.0e-16) { // ok to use relative tolerance
            if (abs(times[index][j] - currtime)/currtime < timeRelTOL) {
              foundtime = true;
              timeindex = j;
            }
          }
          else { // use as absolute tolerance
            if (abs(times[index][j] - currtime) < timeRelTOL) {
              foundtime = true;
              timeindex = j;
            }
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
    
    
    Teuchos::RCP<V> copyData(Teuchos::RCP<V> & src) {
      
      Teuchos::RCP<V> dest = Teuchos::rcp( new V(src->getMap(),1));
      
      auto src_kv = src->template getLocalView<V_device>();
      auto dest_kv = dest->template getLocalView<V_device>();
      Kokkos::deep_copy(dest_kv, src_kv);
      
      return dest;
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
    ScalarT timeRelTOL;
    
    vector<vector<ScalarT> > times;
    vector<vector<Teuchos::RCP<V> > > data;
    
    // Additional data needed for ML
    vector<vector<Kokkos::View<ScalarT*,V_device> > > inputs;
    vector<vector<Kokkos::View<AD*,V_device> > > outputs;
    
    // Add data structures for PyTorch
    
    // Timers
    Teuchos::RCP<Teuchos::Time> solnStorageStoreTimer = Teuchos::TimeMonitor::getNewCounter("MILO::SolutionStorage::store");
    Teuchos::RCP<Teuchos::Time> solnStorageExtractTimer = Teuchos::TimeMonitor::getNewCounter("MILO::SolutionStorage::extract");
    Teuchos::RCP<Teuchos::Time> solnStorageTrainDNNTimer = Teuchos::TimeMonitor::getNewCounter("MILO::SolutionStorage::trainDNN");
    Teuchos::RCP<Teuchos::Time> solnStorageUseDNNTimer = Teuchos::TimeMonitor::getNewCounter("MILO::SolutionStorage::useDNN");
    
    
    
  };
  
}

#endif
