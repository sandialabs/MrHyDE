/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#ifndef MRHYDE_SOLUTION_STORAGE_H
#define MRHYDE_SOLUTION_STORAGE_H

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
      max_storage_ = settings->sublist("Solver").get<LO>("maximum storage", 100);
      // Relative tolerance for storing/extraction solution at given time
      time_rel_TOL_ = settings->sublist("Solver").get<ScalarT>("storage time tol", 1.0e-10);
      
    }
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    bool extract(Teuchos::RCP<V> & vec, const size_t & timeindex) {
      // defaults to index = 0
      // most common use case
      Teuchos::TimeMonitor localtimer(*soln_storage_extract_timer_);
      bool found = false;
      if (data_.size()>0) {
        if (data_[0].size()>timeindex) {
          vec = data_[0][timeindex];
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
      Teuchos::TimeMonitor localtimer(*soln_storage_extract_timer_);
      bool found = false;
      if (data_.size()>index) {
        if (data_[index].size()>timeindex) {
          vec = data_[index][timeindex];
          found = true;
        }
      }
      return found;
    }
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    bool extract(Teuchos::RCP<V> & vec, const size_t & index, const ScalarT & currtime) {
      Teuchos::TimeMonitor localtimer(*soln_storage_extract_timer_);
      bool found = false;
      if (std::abs(currtime)>1.0e-16) { // ok to use relative tolerance
        for (size_t j=0; j<times_[index].size(); j++) {
          if (std::abs(times_[index][j] - currtime)/currtime < time_rel_TOL_) {
            vec = data_[index][j];
            found = true;
          }
        }
      }
      else { // use as absolute tolerance
        for (size_t j=0; j<times_[index].size(); j++) {
          if (std::abs(times_[index][j] - currtime) < time_rel_TOL_) {
            vec = data_[index][j];
            found = true;
          }
        }
      }
      return found;
    }
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    bool extract(Teuchos::RCP<V> & vec, const size_t & index, const ScalarT & currtime, int & timeindex) {
      Teuchos::TimeMonitor localtimer(*soln_storage_extract_timer_);
      bool found = false;
      
      if (std::abs(currtime)>1.0e-16) { // ok to use relative tolerance
        for (size_t j=0; j<times_[index].size(); j++) {
          if (std::abs(times_[index][j] - currtime)/currtime < time_rel_TOL_) {
            vec = data_[index][j];
            timeindex = j;
            found = true;
          }
        }
      }
      else {
        for (size_t j=0; j<times_[index].size(); j++) {
          if (std::abs(times_[index][j] - currtime) < time_rel_TOL_) {
            vec = data_[index][j];
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
      Teuchos::TimeMonitor localtimer(*soln_storage_extract_timer_);
      bool found = false;
      for (size_t j=0; j<times_[index].size(); j++) {
        if (std::abs(times_[index][j] - currtime)/currtime < time_rel_TOL_) {
          found = true;
          if (j>0) {
            vec = data_[index][j-1];
            prevtime = times_[index][j-1];
          }
          else {
            vec = data_[index][j];
            prevtime = times_[index][j];
          }
        }
      }
      return found;
    }
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    bool extractNext(Teuchos::RCP<V> & vec, const size_t & index, const ScalarT & currtime, ScalarT & nexttime) {
      Teuchos::TimeMonitor localtimer(*soln_storage_extract_timer_);
      bool found = false;
      for (size_t j=0; j<times_[index].size(); j++) {
        if (std::abs(times_[index][j] - currtime)/currtime < time_rel_TOL_) {
          found = true;
          if (j<(times_[index].size()-1)) {
            vec = data_[index][j+1];
            nexttime = times_[index][j+1];
          }
          else {
            vec = data_[index][j];
            nexttime = times_[index][j];
          }
        }
      }
      return found;
    }
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    bool extractLast(Teuchos::RCP<V> & vec, const size_t & index, ScalarT & lasttime) {
      Teuchos::TimeMonitor localtimer(*soln_storage_extract_timer_);
      bool found = true;
      vec = data_[index][times_[index].size()-1];
      lasttime = times_[index][times_[index].size()-1];
      return found;
    }
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    vector<vector<Teuchos::RCP<V> > > extractAllData() {
      return data_;
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    vector<vector<ScalarT> > extractAllTimes() {
      return times_;
    }

    size_t getTotalTimes(const int & index) {
      return times_[index].size();
    }

    ScalarT getSpecificTime(const int & index, const int & time_index) {
      return times_[index][time_index];
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void store(const Teuchos::RCP<V> & newvec, const ScalarT & currtime, const size_t & index) {
      
      Teuchos::TimeMonitor localtimer(*soln_storage_store_timer_);
      
      // Deep copy of data
      Teuchos::RCP<V> vecstore = copyData(newvec);
      
      
      if (times_.size() <= index) {
        vector<ScalarT> newtime = {currtime};
        vector<Teuchos::RCP<V> > newdata = {vecstore};
        times_.push_back(newtime);
        data_.push_back(newdata);
      }
      else {
        size_t timeindex;
        bool foundtime = false;
        for (size_t j=0; j<times_[index].size(); j++) {
          if (std::abs(currtime)>1.0e-16) { // ok to use relative tolerance
            if (std::abs(times_[index][j] - currtime)/currtime < time_rel_TOL_) {
              foundtime = true;
              timeindex = j;
            }
          }
          else { // use as absolute tolerance
            if (std::abs(times_[index][j] - currtime) < time_rel_TOL_) {
              foundtime = true;
              timeindex = j;
            }
          }
        }
        if (foundtime) {
          data_[index][timeindex] = vecstore;
        }
        else {
          data_[index].push_back(vecstore);
          times_[index].push_back(currtime);
        }
      }
      
    }
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    
    Teuchos::RCP<V> copyData(const Teuchos::RCP<V> & src) {
      
      Teuchos::RCP<V> dest = Teuchos::rcp( new V(src->getMap(),1));
      
      auto src_kv = src->template getLocalView<V_device>(Tpetra::Access::ReadWrite);
      auto dest_kv = dest->template getLocalView<V_device>(Tpetra::Access::ReadWrite);
      Kokkos::deep_copy(dest_kv, src_kv);
      
      return dest;
    }
    
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void trainDNN() {
      Teuchos::TimeMonitor localtimer(*soln_storage_train_DNN_timer_);
      
    }
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void useDNN() {
      Teuchos::TimeMonitor localtimer(*soln_storage_use_DNN_timer_);
      
    }
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
  private: 

    LO max_storage_;
    ScalarT time_rel_TOL_;
    
    vector<vector<ScalarT> > times_;
    vector<vector<Teuchos::RCP<V> > > data_;
    
    // Additional data needed for ML
    vector<vector<Kokkos::View<ScalarT*,V_device> > > inputs_;
    vector<vector<Kokkos::View<AD*,V_device> > > outputs_;
    
    // Add data structures for PyTorch
    
    // Timers
    Teuchos::RCP<Teuchos::Time> soln_storage_store_timer_ = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SolutionStorage::store");
    Teuchos::RCP<Teuchos::Time> soln_storage_extract_timer_ = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SolutionStorage::extract");
    Teuchos::RCP<Teuchos::Time> soln_storage_train_DNN_timer_ = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SolutionStorage::trainDNN");
    Teuchos::RCP<Teuchos::Time> soln_storage_use_DNN_timer_ = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SolutionStorage::useDNN");
    
    
    
  };
  
}

#endif
