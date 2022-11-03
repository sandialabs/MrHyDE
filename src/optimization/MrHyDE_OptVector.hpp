/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
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
  ROL::Ptr<ROL::StdVector<ScalarT> > scalar_vec;

  const int mpirank;
  bool have_scalar = true, have_field = false;

  mutable vector<ROL::Ptr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> > > dual_field_vec;
  mutable ROL::Ptr<ROL::StdVector<ScalarT> > dual_scalar_vec;
  mutable ROL::Ptr<MrHyDE_OptVector> dual_vec;
  mutable bool isDualInitialized;

public:
  
  ///////////////////////////////////////////////////
  // Constructors for MrHyDE
  ///////////////////////////////////////////////////
  
  MrHyDE_OptVector(const vector<ROL::Ptr<Tpetra::MultiVector<ScalarT,LO,GO,SolverNode> > > & f_vec,
                   const ROL::Ptr<std::vector<ScalarT> > & s_vec,
                   const int mpirank_ = 0) 
    : mpirank(mpirank_), isDualInitialized(false) {
    if (s_vec == ROL::nullPtr) {
      scalar_vec = ROL::nullPtr;
      have_scalar = false;
    }
    else {
      scalar_vec = ROL::makePtr<ROL::StdVector<ScalarT>>(s_vec);
      have_scalar = true;
    }
    have_field = true;
    if (f_vec.size() == 0) {
      have_field = false;
    }
    for (size_t k=0; k<f_vec.size(); ++k) {
      field_vec.push_back(ROL::makePtr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode>>(f_vec[k]));
      dual_field_vec.push_back(ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(field_vec[k]->dual().clone()));
    }
    if (have_scalar) {
      dual_scalar_vec = ROL::dynamicPtrCast<ROL::StdVector<ScalarT> >(scalar_vec->dual().clone());
    }
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  MrHyDE_OptVector(const ROL::Ptr<Tpetra::MultiVector<ScalarT,LO,GO,SolverNode> > & f_vec,
                   const ROL::Ptr<std::vector<ScalarT> > & s_vec,
                   const int mpirank_ = 0) 
    : mpirank(mpirank_), isDualInitialized(false) {

    scalar_vec = ROL::makePtr<ROL::StdVector<ScalarT>>(s_vec);
    field_vec.push_back(ROL::makePtr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(f_vec));
    
    have_scalar = true;
    if (s_vec->size() == 0) {
      have_scalar = false;
    }
    
    have_field = true;
    
    dual_field_vec.push_back(ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(field_vec[0]->dual().clone()));
    dual_scalar_vec = ROL::dynamicPtrCast<ROL::StdVector<ScalarT> >(scalar_vec->dual().clone());
  }

  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  MrHyDE_OptVector(const vector<ROL::Ptr<Tpetra::MultiVector<ScalarT,LO,GO,SolverNode> > > & f_vec)
    : scalar_vec(ROL::nullPtr), mpirank(0), isDualInitialized(false) {
    
    have_scalar = false;
    have_field = true;
    
    for (size_t k=0; k<f_vec.size(); ++k) {
      field_vec.push_back(ROL::makePtr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(f_vec[k]));
      dual_field_vec.push_back(ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(field_vec[k]->dual().clone()));
    }
  }

  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  MrHyDE_OptVector(const ROL::Ptr<std::vector<ScalarT> > & s_vec,
                   const int & mpirank_ = 0)
    : field_vec(ROL::nullPtr), mpirank(mpirank_), isDualInitialized(false) {
    
    have_scalar = true;
    have_field = false;

    scalar_vec = ROL::makePtr<ROL::StdVector<ScalarT>>(s_vec);
    dual_scalar_vec = ROL::dynamicPtrCast<ROL::StdVector<ScalarT> >(scalar_vec->dual().clone());
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  MrHyDE_OptVector()
    : field_vec(ROL::nullPtr), scalar_vec(ROL::nullPtr),  mpirank(0), isDualInitialized(false) {
    have_scalar = false;
    have_field = false;
  }
  
  ///////////////////////////////////////////////////
  
  ///////////////////////////////////////////////////
  // Constructors for cloning
  ///////////////////////////////////////////////////
  
  MrHyDE_OptVector(const vector<ROL::Ptr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> > > & f_vec,
                   const ROL::Ptr<ROL::StdVector<ScalarT> > & s_vec,
                   const int mpirank_ = 0) 
    : field_vec(f_vec), scalar_vec(s_vec), mpirank(mpirank_), isDualInitialized(false) {

    have_scalar = true;
    if (s_vec->getVector()->size() == 0) {
      have_scalar = false;
    }
    
    have_field = true;
    if (f_vec.size() == 0) {
      have_field = false;
    }
    
    for (size_t k=0; k<f_vec.size(); ++k) {
      dual_field_vec.push_back(ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(field_vec[k]->dual().clone()));
    }
    dual_scalar_vec = ROL::dynamicPtrCast<ROL::StdVector<ScalarT> >(scalar_vec->dual().clone());
  }

  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  MrHyDE_OptVector(const ROL::Ptr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> > & f_vec,
                   const ROL::Ptr<ROL::StdVector<ScalarT> > & s_vec,
                   const int mpirank_ = 0) 
    : scalar_vec(s_vec), mpirank(mpirank_), isDualInitialized(false) {

    have_scalar = true;
    if (s_vec->getVector()->size() == 0) {
      have_scalar = false;
    }
    
    have_field = true;
    
    field_vec.push_back(ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(f_vec));
    dual_field_vec.push_back(ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(field_vec[0]->dual().clone()));
    dual_scalar_vec = ROL::dynamicPtrCast<ROL::StdVector<ScalarT> >(scalar_vec->dual().clone());
  }

  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  MrHyDE_OptVector(const vector<ROL::Ptr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> > > & f_vec,
                   const int & mpirank_ = 0)
    : field_vec(f_vec), scalar_vec(ROL::nullPtr), mpirank(mpirank_), isDualInitialized(false) {

    have_scalar = false;
    have_field = true;
    
    for (size_t k=0; k<f_vec.size(); ++k) {
      //field_vec.push_back(ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(f_vec[k]->clone()));
      dual_field_vec.push_back(ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(f_vec[k]->dual().clone()));
    }
  }

  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  MrHyDE_OptVector(const ROL::Ptr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> > & f_vec,
                   const int & mpirank_ = 0)
    : scalar_vec(ROL::nullPtr), mpirank(mpirank_), isDualInitialized(false) {

    have_scalar = false;
    have_field = true;
    
    field_vec.push_back(f_vec);

    for (size_t k=0; k<field_vec.size(); ++k) {
      dual_field_vec.push_back(ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(field_vec[k]->dual().clone()));
    }
  }

  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  MrHyDE_OptVector(const ROL::Ptr<ROL::StdVector<ScalarT> > & s_vec,
                   const int & mpirank_ = 0)
    : scalar_vec(s_vec), mpirank(mpirank_), isDualInitialized(false) {
    
    have_scalar = true;
    have_field = false;
    
    dual_scalar_vec = ROL::dynamicPtrCast<ROL::StdVector<ScalarT> >(scalar_vec->dual().clone());
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  ///////////////////////////////////////////////////
  // Virtual functions from ROL::Vector
  ///////////////////////////////////////////////////
  
  void set( const ROL::Vector<ScalarT> &x ) {
    //cout << "called set" << endl;
    const MrHyDE_OptVector &xs = dynamic_cast<const MrHyDE_OptVector&>(x);
    if (field_vec.size() > 0) {
      auto xs_f = xs.getField();
      for (size_t i=0; i<field_vec.size(); ++i) {
        if ( field_vec[i] != ROL::nullPtr ) {
          field_vec[i]->set(*(xs_f[i]));
        }
      }
    }
    if ( scalar_vec != ROL::nullPtr ) {
      scalar_vec->set(*(xs.getParameter()));
    }
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  void plus( const ROL::Vector<ScalarT> &x ) {
    //cout << "called plus" << endl;
    const MrHyDE_OptVector &xs = dynamic_cast<const MrHyDE_OptVector&>(x);
    if (field_vec.size() > 0) {
      auto xs_f = xs.getField();
      for (size_t i=0; i<field_vec.size(); ++i) {
        if ( field_vec[i] != ROL::nullPtr ) {
          field_vec[i]->plus(*(xs_f[i]));
        }
      }
    }
    
    if ( scalar_vec != ROL::nullPtr ) {
      scalar_vec->plus(*(xs.getParameter()));
    }
  }

  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  void scale( const ScalarT alpha ) {
    //cout << "called scale: " << alpha << endl;
    if (field_vec.size() > 0) {
      for (size_t i=0; i<field_vec.size(); ++i) {
        if ( field_vec[i] != ROL::nullPtr ) {
          field_vec[i]->scale(alpha);
        }
      }
    }
    if ( scalar_vec != ROL::nullPtr ) {
      scalar_vec->scale(alpha);
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
    if ( scalar_vec != ROL::nullPtr ) {
      scalar_vec->zero();
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
    if ( scalar_vec != ROL::nullPtr ) {
      scalar_vec->getVector()->assign(scalar_vec->dimension(),alpha);
    }
  }

  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  void axpy( const ScalarT alpha, const ROL::Vector<ScalarT> &x ) {
    //cout << "called axpy alpha = " << alpha << endl;
    const MrHyDE_OptVector &xs = dynamic_cast<const MrHyDE_OptVector&>(x);
    if (field_vec.size() > 0) {
      auto xs_f = xs.getField();
      for (size_t i=0; i<field_vec.size(); ++i) {
        if ( field_vec[i] != ROL::nullPtr ) {
          field_vec[i]->axpy(alpha,*(xs_f[i]));
        }
      }
    }
    if ( scalar_vec != ROL::nullPtr ) {
      scalar_vec->axpy(alpha,*(xs.getParameter()));
    }
  }

  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////

  ScalarT dot( const ROL::Vector<ScalarT> &x ) const {
    //cout << "called dot" << endl;
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
    if ( scalar_vec != ROL::nullPtr ) {
      val += scalar_vec->dot(*(xs.getParameter()));
    }
    return val;
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////

  ScalarT norm() const {
    //cout << "called norm" << endl;
    ScalarT val(0);
    if (field_vec.size() > 0) {
      for (size_t i=0; i<field_vec.size(); ++i) {
        if ( field_vec[i] != ROL::nullPtr ) {
          ScalarT norm1 = field_vec[i]->norm();
          val += norm1*norm1;
        }
      }
    }
    if ( scalar_vec != ROL::nullPtr ) {
      ScalarT norm2 = scalar_vec->norm();
      val += norm2*norm2;
    }
    return std::sqrt(val);
  } 

  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////

  ROL::Ptr<ROL::Vector<ScalarT> > clone(void) const {
    //cout << "called clone" << endl;
    if ( !have_scalar ) {
      vector<ROL::Ptr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> > > fvecs;
      
      for (size_t i=0; i<field_vec.size(); ++i) {
        fvecs.push_back(ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(field_vec[i]->clone()));
      }
      return ROL::makePtr<MrHyDE_OptVector>(fvecs, mpirank);

      //vector<ROL::Vector<ScalarT>> fvecs;
      //for (size_t i=0; i<field_vec.size(); ++i) {
      //  fvecs.push_back(field_vec[i]->clone());
      //}
      //return ROL::makePtr<MrHyDE_OptVector>(fvecs, mpirank); 
      /*
      typedef Vector<Real>                  V;
      typedef ROL::Ptr<V>                   Vp;
      typedef PartitionedVector<Real>       PV;

      Vp clone() const {
      std::vector<Vp> clonevec;
      for( size_type i=0; i<vecs_.size(); ++i ) {
        clonevec.push_back(vecs_[i]->clone());
      }
      return ROL::makePtr<PV>(clonevec);
      */
    

    }
    if ( !have_field) {
      return ROL::makePtr<MrHyDE_OptVector>(
             ROL::dynamicPtrCast<ROL::StdVector<ScalarT> >(scalar_vec->clone()), mpirank);
    }
    vector<ROL::Ptr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> > > fvecs;
    for (size_t i=0; i<field_vec.size(); ++i) {
      fvecs.push_back(ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(field_vec[i]->clone()));
    }
    return ROL::makePtr<MrHyDE_OptVector>(fvecs,
                                          ROL::dynamicPtrCast<ROL::StdVector<ScalarT> >(scalar_vec->clone()),
                                          mpirank);
  }

  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////

  const ROL::Vector<ScalarT> & dual(void) const {
    //cout << "called dual" << endl;
    if ( !isDualInitialized ) {
      if ( !have_field) {
        //if ( field_vec == ROL::nullPtr ) {
        dual_vec = ROL::makePtr<MrHyDE_OptVector>(dual_scalar_vec);
      }
      else if ( !have_scalar ) {
        dual_vec = ROL::makePtr<MrHyDE_OptVector>(dual_field_vec);
      }
      else {
        dual_vec = ROL::makePtr<MrHyDE_OptVector>(dual_field_vec,dual_scalar_vec);
      }
      isDualInitialized = true;
    }
    for (size_t i=0; i<field_vec.size(); ++i) {
      if ( field_vec[i] != ROL::nullPtr ) {
        dual_field_vec[i]->set(field_vec[i]->dual());
      }
    }
    if ( scalar_vec != ROL::nullPtr ) {
      dual_scalar_vec->set(scalar_vec->dual());
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
      if ( field_vec[i] != ROL::nullPtr ) val += field_vec[i]->apply(*(xs_f[i]));
    }
    if ( scalar_vec != ROL::nullPtr ) val += scalar_vec->apply(*(xs.getParameter()));
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
    if ( scalar_vec != ROL::nullPtr ) {
      scalar_vec->applyUnary(f);
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
    if ( scalar_vec != ROL::nullPtr ) {
      scalar_vec->applyBinary(f,*xs.getParameter());
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
    if ( scalar_vec != ROL::nullPtr ) {
      r.reduce(scalar_vec->reduce(r),result);
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
    if ( scalar_vec != ROL::nullPtr ) {
      dim += scalar_vec->dimension();
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
    if (scalar_vec != ROL::nullPtr) {
      scalar_vec->randomize(l,u);
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
      if (mpirank == 0) {
        scalar_vec->print(outStream);
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
  
  ROL::Ptr<const ROL::StdVector<ScalarT> > getParameter(void) const { 
    return scalar_vec; 
  }

  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  vector<ROL::Ptr<ROL::TpetraMultiVector<ScalarT> > > getField(void) { 
    return field_vec;
  }

  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  ROL::Ptr<ROL::StdVector<ScalarT> > getParameter(void) { 
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
    scalar_vec->set(vec); 
  }
  
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  bool isDynamic() {
    bool isdy = false;
    if (have_field && field_vec.size() > 1) {
      isdy = true;
    }
    return isdy;
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

