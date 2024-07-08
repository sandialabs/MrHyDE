/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
 ************************************************************************/

#ifndef MRHYDE_OPTVEC_HPP
#define MRHYDE_OPTVEC_HPP

#include "trilinos.hpp"
#include "preferences.hpp"

#include "ROL_StdVector.hpp"
#include "ROL_TpetraMultiVector.hpp"

class MrHyDE_OptVector : public ROL::Vector<ScalarT> {
  
private:
  
  std::vector<ROL::Ptr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> > > field_vec; // vector for dynamics
  std::vector<ROL::Ptr<ROL::StdVector<ScalarT> > > scalar_vec;
  
  const int mpirank;
  bool have_scalar, have_field, have_dynamic_scalar, have_dynamic_field;
  double dyn_dt = 1.0;
  
  mutable std::vector<ROL::Ptr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> > > dual_field_vec;
  mutable std::vector<ROL::Ptr<ROL::StdVector<ScalarT> > > dual_scalar_vec;
  mutable ROL::Ptr<MrHyDE_OptVector> dual_vec;
  mutable bool isDualInitialized;
  
public:
  
  ///////////////////////////////////////////////////
  // Constructors for MrHyDE_OptVector
  ///////////////////////////////////////////////////
  
  MrHyDE_OptVector(const std::vector<ROL::Ptr<Tpetra::MultiVector<ScalarT,LO,GO,SolverNode> > > & f_vec,
                   const std::vector<ROL::Ptr<std::vector<ScalarT> > > & s_vec,
                   const double & dt,
                   const int mpirank_ = 0)
  : mpirank(mpirank_), dyn_dt(dt), isDualInitialized(false) {
    
    if (s_vec.size() == 0) {
      //scalar_vec = ROL::nullPtr;
      have_scalar = false;
      have_dynamic_scalar = false;
    }
    else {
      for (size_t k=0; k<s_vec.size(); ++k) {
        scalar_vec.push_back(ROL::makePtr<ROL::StdVector<ScalarT>>(s_vec[k]));
      }
      have_scalar = true;
      if (s_vec.size() > 1) {
        have_dynamic_scalar = true;
      }
      else {
        have_dynamic_scalar = false;
      }
    }
    
    have_field = true;
    if (f_vec.size() == 0) {
      have_field = false;
    }
    for (size_t k=0; k<f_vec.size(); ++k) {
      field_vec.push_back(ROL::makePtr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode>>(f_vec[k]));
      dual_field_vec.push_back(ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(field_vec[k]->dual().clone()));
    }
    
    if (f_vec.size() > 1) {
      have_dynamic_field = true;
    }
    else {
      have_dynamic_field = false;
    }
    
    if (have_scalar) {
      for (size_t k=0; k<s_vec.size(); ++k) {
        dual_scalar_vec.push_back(ROL::dynamicPtrCast<ROL::StdVector<ScalarT> >(scalar_vec[k]->dual().clone()));
      }
    }
    
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  MrHyDE_OptVector(const std::vector<ROL::Ptr<Tpetra::MultiVector<ScalarT,LO,GO,SolverNode> > > & f_vec,
                   const ROL::Ptr<std::vector<ScalarT> > & s_vec,
                   const double & dt,
                   const int mpirank_ = 0)
  : mpirank(mpirank_), dyn_dt(dt), isDualInitialized(false) {
    have_dynamic_scalar = false;
    if (s_vec == ROL::nullPtr) {
      have_scalar = false;
    }
    else {
      scalar_vec.push_back(ROL::makePtr<ROL::StdVector<ScalarT>>(s_vec));
      have_scalar = true;
    }
    have_field = true;
    if (f_vec.size() == 0) {
      have_field = false;
    }
    if (f_vec.size() > 1) {
      have_dynamic_field = true;
    }
    else {
      have_dynamic_field = false;
    }
      
    for (size_t k=0; k<f_vec.size(); ++k) {
      field_vec.push_back(ROL::makePtr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode>>(f_vec[k]));
      dual_field_vec.push_back(ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(field_vec[k]->dual().clone()));
    }
    if (have_scalar) {
      dual_scalar_vec.push_back(ROL::dynamicPtrCast<ROL::StdVector<ScalarT> >(scalar_vec[0]->dual().clone()));
    }
    
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  MrHyDE_OptVector(const ROL::Ptr<Tpetra::MultiVector<ScalarT,LO,GO,SolverNode> > & f_vec,
                   const std::vector<ROL::Ptr<std::vector<ScalarT> > > & s_vec,
                   const double & dt,
                   const int mpirank_ = 0)
  : mpirank(mpirank_), dyn_dt(dt), isDualInitialized(false) {
    if (s_vec.size() == 0) {
      //scalar_vec = ROL::nullPtr;
      have_scalar = false;
      have_dynamic_scalar = false;
    }
    else {
      for (size_t k=0; k<s_vec.size(); ++k) {
        scalar_vec.push_back(ROL::makePtr<ROL::StdVector<ScalarT>>(s_vec[k]));
      }
      have_scalar = true;
      if (s_vec.size() > 1) {
        have_dynamic_scalar = true;
      }
      else {
        have_dynamic_scalar = false;
      }
    }
    
    have_field = true;
    field_vec.push_back(ROL::makePtr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode>>(f_vec));
    dual_field_vec.push_back(ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(field_vec[0]->dual().clone()));
    
    have_dynamic_field = false;
    
    if (have_scalar) {
      for (size_t k=0; k<s_vec.size(); ++k) {
        dual_scalar_vec.push_back(ROL::dynamicPtrCast<ROL::StdVector<ScalarT> >(scalar_vec[k]->dual().clone()));
      }
    }
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  MrHyDE_OptVector(const ROL::Ptr<Tpetra::MultiVector<ScalarT,LO,GO,SolverNode> > & f_vec,
                   const ROL::Ptr<std::vector<ScalarT> > & s_vec,
                   const int mpirank_ = 0)
  : mpirank(mpirank_), isDualInitialized(false) {
    
    scalar_vec.push_back(ROL::makePtr<ROL::StdVector<ScalarT>>(s_vec));
    field_vec.push_back(ROL::makePtr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(f_vec));
    
    have_dynamic_scalar = false;
    have_dynamic_field = false;
    
    if (s_vec->size() > 0) {
      have_scalar = true;
    }
    else {//if (s_vec == ROL::nullPtr) {
      have_scalar = false;
    }
    
    have_field = true;
    
    dual_field_vec.push_back(ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(field_vec[0]->dual().clone()));
    dual_scalar_vec.push_back(ROL::dynamicPtrCast<ROL::StdVector<ScalarT> >(scalar_vec[0]->dual().clone()));
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  MrHyDE_OptVector(const std::vector<ROL::Ptr<Tpetra::MultiVector<ScalarT,LO,GO,SolverNode> > > & f_vec,
                   const double & dt)
  : scalar_vec(ROL::nullPtr), mpirank(0), dyn_dt(dt), isDualInitialized(false) {
    
    have_scalar = false;
    have_field = true;
    have_dynamic_scalar = false;
    
    for (size_t k=0; k<f_vec.size(); ++k) {
      field_vec.push_back(ROL::makePtr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(f_vec[k]));
      dual_field_vec.push_back(ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(field_vec[k]->dual().clone()));
    }
    have_dynamic_field = false;
    if (f_vec.size() > 1) {
      have_dynamic_field = true;
    }
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  MrHyDE_OptVector(const ROL::Ptr<std::vector<ScalarT> > & s_vec,
                   const int & mpirank_ = 0)
  : field_vec(ROL::nullPtr), mpirank(mpirank_), isDualInitialized(false) {
    
    have_scalar = true;
    have_field = false;
    have_dynamic_scalar = false;
    have_dynamic_field = false;
    
    scalar_vec.push_back(ROL::makePtr<ROL::StdVector<ScalarT>>(s_vec));
    dual_scalar_vec.push_back(ROL::dynamicPtrCast<ROL::StdVector<ScalarT> >(scalar_vec[0]->dual().clone()));
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  MrHyDE_OptVector()
  : mpirank(0), isDualInitialized(false) {
    have_scalar = false;
    have_field = false;
    have_dynamic_scalar = false;
    have_dynamic_field = false;
  }
  
  ///////////////////////////////////////////////////
  
  ///////////////////////////////////////////////////
  // Constructors for cloning
  ///////////////////////////////////////////////////
  
  MrHyDE_OptVector(const std::vector<ROL::Ptr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> > > & f_vec,
                   const std::vector<ROL::Ptr<ROL::StdVector<ScalarT> > > & s_vec,
                   const double & dt,
                   const int mpirank_ = 0)
  : field_vec(f_vec), scalar_vec(s_vec), mpirank(mpirank_), dyn_dt(dt), isDualInitialized(false) {
    
    have_scalar = true;
    if (s_vec[0]->getVector()->size() == 0) {
      have_scalar = false;
    }
    if (s_vec.size() > 1) {
      have_dynamic_scalar = true;
    }
    else {
      have_dynamic_scalar = false;
    }
    
    have_field = true;
    if (f_vec.size() == 0) {
      have_field = false;
    }
    if (f_vec.size() > 1) {
      have_dynamic_field = true;
    }
    else {
      have_dynamic_field = false;
    }
    
    for (size_t k=0; k<f_vec.size(); ++k) {
      dual_field_vec.push_back(ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(field_vec[k]->dual().clone()));
    }
    
    for (size_t k=0; k<s_vec.size(); ++k) {
      dual_scalar_vec.push_back(ROL::dynamicPtrCast<ROL::StdVector<ScalarT> >(scalar_vec[k]->dual().clone()));
    }
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////

  MrHyDE_OptVector(const std::vector<ROL::Ptr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> > > & f_vec,
                   const ROL::Ptr<ROL::StdVector<ScalarT> > & s_vec,
                   const double & dt,
                   const int mpirank_ = 0)
  : field_vec(f_vec), mpirank(mpirank_), dyn_dt(dt), isDualInitialized(false) {
    
    scalar_vec.push_back(s_vec);//ROL::makePtr<ROL::StdVector<ScalarT>>(s_vec));
    have_scalar = true;
    if (s_vec->getVector()->size() == 0) {
      have_scalar = false;
    }
    have_dynamic_scalar = false;
    
    have_field = true;
    if (f_vec.size() == 0) {
      have_field = false;
    }
    if (f_vec.size() > 1) {
      have_dynamic_field = true;
    }
    else {
      have_dynamic_field = false;
    }
    
    for (size_t k=0; k<f_vec.size(); ++k) {
      dual_field_vec.push_back(ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(field_vec[k]->dual().clone()));
    }
    dual_scalar_vec.push_back(ROL::dynamicPtrCast<ROL::StdVector<ScalarT> >(scalar_vec[0]->dual().clone()));
  }

  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////

  MrHyDE_OptVector(const ROL::Ptr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> > & f_vec,
                   const std::vector<ROL::Ptr<ROL::StdVector<ScalarT> > > & s_vec,
                   const double & dt,
                   const int mpirank_ = 0)
  : scalar_vec(s_vec), mpirank(mpirank_), dyn_dt(dt), isDualInitialized(false) {
  
    have_scalar = true;
    if (s_vec[0]->getVector()->size() == 0) {
      have_scalar = false;
    }
    if (s_vec.size() > 1) {
      have_dynamic_scalar = true;
    }
    else {
      have_dynamic_scalar = false;
    }
  
    have_field = true;
    have_dynamic_field = false;
  
    field_vec.push_back(f_vec);//ROL::makePtr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(f_vec));
    
    dual_field_vec.push_back(ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(field_vec[0]->dual().clone()));
  
    for (size_t k=0; k<s_vec.size(); ++k) {
      dual_scalar_vec.push_back(ROL::dynamicPtrCast<ROL::StdVector<ScalarT> >(scalar_vec[k]->dual().clone()));
    }
  }

  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////

  MrHyDE_OptVector(const ROL::Ptr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> > & f_vec,
                   const ROL::Ptr<ROL::StdVector<ScalarT> > & s_vec,
                   const int mpirank_ = 0)
  : mpirank(mpirank_), isDualInitialized(false) {
    
    have_scalar = true;
    if (s_vec->getVector()->size() == 0) {
      have_scalar = false;
    }
    have_dynamic_scalar = false;
    
    have_field = true;
    have_dynamic_field = false;
    
    scalar_vec.push_back(s_vec);//ROL::makePtr<ROL::StdVector<ScalarT>>(s_vec));
    field_vec.push_back(f_vec);//ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(f_vec));
    dual_field_vec.push_back(ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(field_vec[0]->dual().clone()));
    dual_scalar_vec.push_back(ROL::dynamicPtrCast<ROL::StdVector<ScalarT> >(scalar_vec[0]->dual().clone()));
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  MrHyDE_OptVector(const std::vector<ROL::Ptr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> > > & f_vec,
                   const double & dt,
                   const int & mpirank_ = 0)
  : field_vec(f_vec), mpirank(mpirank_), dyn_dt(dt), isDualInitialized(false) {
    
    have_scalar = false;
    have_field = true;
    scalar_vec.push_back(ROL::nullPtr);
    
    if (f_vec.size() > 1) {
      have_dynamic_field = true;
    }
    else {
      have_dynamic_field = false;
    }
    have_dynamic_scalar = false;
    
    for (size_t k=0; k<f_vec.size(); ++k) {
      //field_vec.push_back(ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(f_vec[k]->clone()));
      dual_field_vec.push_back(ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(f_vec[k]->dual().clone()));
    }
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  MrHyDE_OptVector(const ROL::Ptr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> > & f_vec,
                   const int & mpirank_ = 0)
  : mpirank(mpirank_), isDualInitialized(false) {
    
    have_scalar = false;
    scalar_vec.push_back(ROL::nullPtr);
    have_dynamic_scalar = false;
    
    have_field = true;
    
    field_vec.push_back(f_vec);
    have_dynamic_field = false;
    
    for (size_t k=0; k<field_vec.size(); ++k) {
      dual_field_vec.push_back(ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(field_vec[k]->dual().clone()));
    }
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  MrHyDE_OptVector(const ROL::Ptr<ROL::StdVector<ScalarT> > & s_vec,
                   const int & mpirank_ = 0)
  : mpirank(mpirank_), isDualInitialized(false) {
    
    scalar_vec.push_back(s_vec);
    have_scalar = true;
    have_dynamic_scalar = false;
    have_field = false;
    have_dynamic_field = false;
    
    dual_scalar_vec.push_back(ROL::dynamicPtrCast<ROL::StdVector<ScalarT> >(scalar_vec[0]->dual().clone()));
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  MrHyDE_OptVector(const std::vector<ROL::Ptr<ROL::StdVector<ScalarT> > > & s_vec,
                   const double & dt,
                   const int & mpirank_ = 0)
  : scalar_vec(s_vec), mpirank(mpirank_), dyn_dt(dt), isDualInitialized(false) {
    
    have_scalar = true;
    have_dynamic_scalar = false;
    if (s_vec.size() > 1) {
      have_dynamic_scalar = true;
    }
    have_field = false;
    have_dynamic_field = false;
    
    for (size_t k=0; k<s_vec.size(); ++k) {
      dual_scalar_vec.push_back(ROL::dynamicPtrCast<ROL::StdVector<ScalarT> >(scalar_vec[k]->dual().clone()));
    }
    
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  ///////////////////////////////////////////////////
  // Virtual functions from ROL::Vector
  ///////////////////////////////////////////////////
  
  void set( const ROL::Vector<ScalarT> &x ) {
    
    const MrHyDE_OptVector &xs = dynamic_cast<const MrHyDE_OptVector&>(x);
    if (field_vec.size() > 0) {
      auto xs_f = xs.getField();
      for (size_t i=0; i<field_vec.size(); ++i) {
        if ( field_vec[i] != ROL::nullPtr ) {
          field_vec[i]->set(*(xs_f[i]));
        }
      }
    }
    
    if (scalar_vec.size() > 0) {
      auto xs_s = xs.getParameter();
      for (size_t i=0; i<scalar_vec.size(); ++i) {
        if ( scalar_vec[i] != ROL::nullPtr ) {
          scalar_vec[i]->set(*(xs_s[i]));
        }
      }
    }
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  void plus( const ROL::Vector<ScalarT> &x ) {
    
    const MrHyDE_OptVector &xs = dynamic_cast<const MrHyDE_OptVector&>(x);
    if (field_vec.size() > 0) {
      auto xs_f = xs.getField();
      for (size_t i=0; i<field_vec.size(); ++i) {
        if ( field_vec[i] != ROL::nullPtr ) {
          field_vec[i]->plus(*(xs_f[i]));
        }
      }
    }
    
    if (scalar_vec.size() > 0) {
      auto xs_s = xs.getParameter();
      for (size_t i=0; i<scalar_vec.size(); ++i) {
        if ( scalar_vec[i] != ROL::nullPtr ) {
          scalar_vec[i]->plus(*(xs_s[i]));
        }
      }
    }
    
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  void scale( const ScalarT alpha ) {
    
    if (field_vec.size() > 0) {
      for (size_t i=0; i<field_vec.size(); ++i) {
        if ( field_vec[i] != ROL::nullPtr ) {
          field_vec[i]->scale(alpha);
        }
      }
    }
    
    if (scalar_vec.size() > 0) {
      for (size_t i=0; i<scalar_vec.size(); ++i) {
        if ( scalar_vec[i] != ROL::nullPtr ) {
          scalar_vec[i]->scale(alpha);
        }
      }
    }
    
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  void zero() {
    
    if (field_vec.size() > 0) {
      for (size_t i=0; i<field_vec.size(); ++i) {
        if ( field_vec[i] != ROL::nullPtr ) {
          field_vec[i]->zero();
        }
      }
    }
    
    if (scalar_vec.size() > 0) {
      for (size_t i=0; i<scalar_vec.size(); ++i) {
        if ( scalar_vec[i] != ROL::nullPtr ) {
          scalar_vec[i]->zero();
        }
      }
    }
    
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  void putScalar(const ScalarT & alpha) {
    
    if (field_vec.size() > 0) {
      for (size_t i=0; i<field_vec.size(); ++i) {
        if ( field_vec[i] != ROL::nullPtr ) {
          field_vec[i]->getVector()->putScalar(alpha);
        }
      }
    }
    
    if (scalar_vec.size() > 0) {
      for (size_t i=0; i<scalar_vec.size(); ++i) {
        if ( scalar_vec[i] != ROL::nullPtr ) {
          scalar_vec[i]->getVector()->assign(scalar_vec[i]->dimension(),alpha);
        }
      }
    }
    
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  void axpy( const ScalarT alpha, const ROL::Vector<ScalarT> &x ) {
    
    const MrHyDE_OptVector &xs = dynamic_cast<const MrHyDE_OptVector&>(x);
    if (field_vec.size() > 0) {
      auto xs_f = xs.getField();
      for (size_t i=0; i<field_vec.size(); ++i) {
        if ( field_vec[i] != ROL::nullPtr ) {
          field_vec[i]->axpy(alpha,*(xs_f[i]));
        }
      }
    }
    
    if (scalar_vec.size() > 0) {
      auto xs_s = xs.getParameter();
      for (size_t i=0; i<scalar_vec.size(); ++i) {
        if ( scalar_vec[i] != ROL::nullPtr ) {
          scalar_vec[i]->axpy(alpha,*(xs_s[i]));
        }
      }
    }
    
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  ScalarT dot( const ROL::Vector<ScalarT> &x ) const {
    
    const MrHyDE_OptVector &xs = dynamic_cast<const MrHyDE_OptVector&>(x);
    ScalarT val(0);
    if (field_vec.size() > 0) {
      auto xs_f = xs.getField();
      for (size_t i=0; i<field_vec.size(); ++i) {
        if ( field_vec[i] != ROL::nullPtr ) {
          val += field_vec[i]->dot(*(xs_f[i]));
        }
      }
    }
    
    if (scalar_vec.size() > 0) {
      auto xs_s = xs.getParameter();
      for (size_t i=0; i<scalar_vec.size(); ++i) {
        if ( scalar_vec[i] != ROL::nullPtr ) {
          val += scalar_vec[i]->dot(*(xs_s[i]));
        }
      }
    }
    
    return val;
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  ScalarT norm() const {
    
    ScalarT val(0);
    if (field_vec.size() > 0) {
      for (size_t i=0; i<field_vec.size(); ++i) {
        if ( field_vec[i] != ROL::nullPtr ) {
          ScalarT norm1 = field_vec[i]->norm();
          val += norm1*norm1;
        }
      }
    }
    
    if (scalar_vec.size() > 0) {
      for (size_t i=0; i<scalar_vec.size(); ++i) {
        if ( scalar_vec[i] != ROL::nullPtr ) {
          ScalarT norm1 = scalar_vec[i]->norm();
          val += norm1*norm1;
        }
      }
    }
    
    return std::sqrt(val);
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  ROL::Ptr<ROL::Vector<ScalarT> > clone(void) const {
    
    ROL::Ptr<ROL::Vector<ScalarT> > clonevec;
    if ( !have_scalar ) {
      std::vector<ROL::Ptr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> > > fvecs;
      
      for (size_t i=0; i<field_vec.size(); ++i) {
        fvecs.push_back(ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(field_vec[i]->clone()));
      }
      clonevec = ROL::makePtr<MrHyDE_OptVector>(fvecs, dyn_dt, mpirank);
      
    }
    else if ( !have_field) {
      std::vector<ROL::Ptr<ROL::StdVector<ScalarT> > > svecs;
      for (size_t i=0; i<scalar_vec.size(); ++i) {
        svecs.push_back(ROL::dynamicPtrCast<ROL::StdVector<ScalarT> >(scalar_vec[i]->clone()));
      }
      clonevec = ROL::makePtr<MrHyDE_OptVector>(svecs, dyn_dt, mpirank);
    }
    else {
      std::vector<ROL::Ptr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> > > fvecs;
      for (size_t i=0; i<field_vec.size(); ++i) {
        fvecs.push_back(ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(field_vec[i]->clone()));
      }
      std::vector<ROL::Ptr<ROL::StdVector<ScalarT> > > svecs;
      for (size_t i=0; i<scalar_vec.size(); ++i) {
        svecs.push_back(ROL::dynamicPtrCast<ROL::StdVector<ScalarT> >(scalar_vec[i]->clone()));
      }
      
      clonevec = ROL::makePtr<MrHyDE_OptVector>(fvecs, svecs, dyn_dt, mpirank);
    }
    return clonevec;
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  const ROL::Vector<ScalarT> & dual(void) const {
    
    if ( !isDualInitialized ) {
      if ( !have_field) {
        dual_vec = ROL::makePtr<MrHyDE_OptVector>(dual_scalar_vec, dyn_dt);
      }
      else if ( !have_scalar ) {
        dual_vec = ROL::makePtr<MrHyDE_OptVector>(dual_field_vec, dyn_dt);
      }
      else {
        dual_vec = ROL::makePtr<MrHyDE_OptVector>(dual_field_vec, dual_scalar_vec, dyn_dt);
      }
      isDualInitialized = true;
    }
    for (size_t i=0; i<field_vec.size(); ++i) {
      if ( field_vec[i] != ROL::nullPtr ) {
        dual_field_vec[i]->set(field_vec[i]->dual());
      }
    }
    for (size_t i=0; i<scalar_vec.size(); ++i) {
      if ( scalar_vec[i] != ROL::nullPtr ) {
        dual_scalar_vec[i]->set(scalar_vec[i]->dual());
      }
    }
    return *dual_vec;
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  ScalarT apply(const ROL::Vector<ScalarT> &x) const {
    const MrHyDE_OptVector &xs = dynamic_cast<const MrHyDE_OptVector&>(x);
    ScalarT val(0);
    auto xs_f = xs.getField();
    for (size_t i=0; i<field_vec.size(); ++i) {
      if ( field_vec[i] != ROL::nullPtr ) {
        val += field_vec[i]->apply(*(xs_f[i]));
      }
    }
    auto xs_s = xs.getParameter();
    for (size_t i=0; i<scalar_vec.size(); ++i) {
      if ( scalar_vec[i] != ROL::nullPtr ) {
        val += scalar_vec[i]->apply(*(xs_s[i]));
      }
    }
    return val;
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  ROL::Ptr<ROL::Vector<ScalarT> > basis( const int i )  const {
    ROL::Ptr<ROL::Vector<ScalarT> > e;
    std::cout << "Basis got called" << std::endl;
    
    /*
     if ( field_vec != ROL::nullPtr && scalar_vec != ROL::nullPtr ) {
     int n1 = field_vec->dimension();
     ROL::Ptr<ROL::Vector<ScalarT> > e1, e2;
     if ( i < n1 ) {
     e1 = field_vec->basis(i);
     e2 = scalar_vec->clone(); e2->zero();
     }
     else {
     e1 = field_vec->clone(); e1->zero();
     e2 = scalar_vec->basis(i-n1);
     }
     e = ROL::makePtr<MrHyDE_OptVector>(
     ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT> >(e1),
     ROL::dynamicPtrCast<ROL::StdVector<ScalarT> >(e2));
     }
     if ( field_vec != ROL::nullPtr && scalar_vec == ROL::nullPtr ) {
     int n1 = field_vec->dimension();
     ROL::Ptr<ROL::Vector<ScalarT> > e1;
     if ( i < n1 ) {
     e1 = field_vec->basis(i);
     }
     else {
     e1->zero();
     }
     e = ROL::makePtr<MrHyDE_OptVector>(
     ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT> >(e1));
     }
     if ( field_vec == ROL::nullPtr && scalar_vec != ROL::nullPtr ) {
     int n2 = scalar_vec->dimension();
     ROL::Ptr<ROL::Vector<ScalarT> > e2;
     if ( i < n2 ) {
     e2 = scalar_vec->basis(i);
     }
     else {
     e2->zero();
     }
     e = ROL::makePtr<MrHyDE_OptVector>(
     ROL::dynamicPtrCast<ROL::StdVector<ScalarT> >(e2));
     }*/
    return e;
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  void applyUnary( const ROL::Elementwise::UnaryFunction<ScalarT> &f ) {
    for (size_t i=0; i<field_vec.size(); ++i) {
      if ( field_vec[i] != ROL::nullPtr ) {
        field_vec[i]->applyUnary(f);
      }
    }
    for (size_t i=0; i<scalar_vec.size(); ++i) {
      if ( scalar_vec[i] != ROL::nullPtr ) {
        scalar_vec[i]->applyUnary(f);
      }
    }
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  void applyBinary( const ROL::Elementwise::BinaryFunction<ScalarT> &f, const ROL::Vector<ScalarT> &x ) {
    const MrHyDE_OptVector &xs = dynamic_cast<const MrHyDE_OptVector&>(x);
    auto xs_f = xs.getField();
    for (size_t i=0; i<field_vec.size(); ++i) {
      if ( field_vec[i] != ROL::nullPtr ) {
        field_vec[i]->applyBinary(f,*(xs_f[i]));
      }
    }
    
    auto xs_s = xs.getParameter();
    for (size_t i=0; i<scalar_vec.size(); ++i) {
      if ( scalar_vec[i] != ROL::nullPtr ) {
        scalar_vec[i]->applyBinary(f,*xs_s[i]);
      }
    }
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  ScalarT reduce( const ROL::Elementwise::ReductionOp<ScalarT> &r ) const {
    ScalarT result = r.initialValue();
    for (size_t i=0; i<field_vec.size(); ++i) {
      if ( field_vec[i] != ROL::nullPtr ) {
        r.reduce(field_vec[i]->reduce(r),result);
      }
    }
    for (size_t i=0; i<scalar_vec.size(); ++i) {
      if ( scalar_vec[i] != ROL::nullPtr ) {
        r.reduce(scalar_vec[i]->reduce(r),result);
      }
    }
    return result;
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  int dimension() const {
    int dim(0);
    for (size_t i=0; i<field_vec.size(); ++i) {
      if ( field_vec[i] != ROL::nullPtr ) {
        dim += field_vec[i]->dimension();
      }
    }
    for (size_t i=0; i<scalar_vec.size(); ++i) {
      if ( scalar_vec[i] != ROL::nullPtr ) {
        dim += scalar_vec[i]->dimension();
      }
    }
    return dim;
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  void randomize(const ScalarT l = 0.0, const ScalarT u = 1.0) {
    for (size_t i=0; i<field_vec.size(); ++i) {
      if (field_vec[i] != ROL::nullPtr) {
        field_vec[i]->randomize(l,u);
      }
    }
    for (size_t i=0; i<scalar_vec.size(); ++i) {
      if (scalar_vec[i] != ROL::nullPtr) {
        scalar_vec[i]->randomize(l,u);
      }
    }
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  void print(std::ostream &outStream) const {
    if (have_field) {
      for (size_t i=0; i<field_vec.size(); ++i) {
        if (field_vec[i] != ROL::nullPtr) {
          field_vec[i]->print(outStream);
        }
      }
    }
    if (have_scalar) {
      for (size_t i=0; i<scalar_vec.size(); ++i) {
        if (mpirank == 0 && scalar_vec[i] != ROL::nullPtr) {
          scalar_vec[i]->print(outStream);
        }
      }
    }
  }
  
  ///////////////////////////////////////////////////
  // Extra functions
  ///////////////////////////////////////////////////
  
  std::vector<ROL::Ptr<ROL::TpetraMultiVector<ScalarT> > > getField(void) const {
    return field_vec;
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  std::vector<ROL::Ptr<ROL::StdVector<ScalarT> > > getParameter(void) const {
    return scalar_vec;
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  std::vector<ROL::Ptr<ROL::TpetraMultiVector<ScalarT> > > getField(void) {
    return field_vec;
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  std::vector<ROL::Ptr<ROL::StdVector<ScalarT> > > getParameter(void) {
    return scalar_vec;
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  void setField(const ROL::Vector<ScalarT>& vec) {
    for (size_t i=0; i<field_vec.size(); ++i) {
      if ( field_vec[i] != ROL::nullPtr ) {
        field_vec[i]->set(vec);
      }
    }
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  void setParameter(const ROL::Vector<ScalarT>& vec) {
    for (size_t i=0; i<scalar_vec.size(); ++i) {
      if ( scalar_vec[i] != ROL::nullPtr ) {
        scalar_vec[i]->set(vec);
      }
    }
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////

  void setParameter(const std::vector<ROL::Vector<ScalarT> >& vec) {
    for (size_t i=0; i<scalar_vec.size(); ++i) {
      if ( scalar_vec[i] != ROL::nullPtr ) {
        scalar_vec[i]->set(vec[i]);
      }
    }
  }

  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  bool haveDynamicField() {
    return have_dynamic_field;
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////

  bool haveDynamicScalar() {
    return have_dynamic_scalar;
  }

  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  bool haveScalar() {
    return have_scalar;
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  bool haveField() {
    return have_field;
  }
  
}; // class MrHyDE_OptVector

#endif

