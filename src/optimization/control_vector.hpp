/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef CONTROLVECTOR_HPP
#define CONTROLVECTOR_HPP

#include "ROL_Vector.hpp"
#include "Epetra_MultiVector.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_Map.h"
#include "Epetra_LocalMap.h"
#include "Epetra_IntVector.h"
#include "Epetra_Import.h"

template<class Real>
class ControlVector : public ROL::Vector<Real> {
private:
  
  Teuchos::RCP<Epetra_MultiVector>  epetra_vec_;
  Teuchos::RCP<Epetra_CrsMatrix>    M_;

public:

    virtual ~ControlVector() {}

  ControlVector( const Teuchos::RCP<Epetra_MultiVector> & epetra_vec, 
                 const Teuchos::RCP<Epetra_CrsMatrix>   & M ) : epetra_vec_(epetra_vec), M_(M) { }

  void plus( const ROL::Vector<Real> & x ) {
    ControlVector &ex = Teuchos::dyn_cast<ControlVector>(const_cast <ROL::Vector<Real>&>(x));
    Teuchos::RCP<const Epetra_MultiVector> xvalptr = ex.getVector();
    epetra_vec_->Update( 1.0, *xvalptr, 1.0 );
  }
  
  void scale( Real alpha )  {
    epetra_vec_->Scale( (double)alpha );
  }

  Real dot( const ROL::Vector<Real> & x ) const {
    double val[1];
    Epetra_MultiVector tmp(*(this->epetra_vec_)); 
    tmp.PutScalar(0.0);
    (this->M_)->Multiply(false,*(this->epetra_vec_),tmp);
    ControlVector &ex = Teuchos::dyn_cast<ControlVector>(const_cast <ROL::Vector<Real>&>(x));
    Teuchos::RCP<const Epetra_MultiVector> xvalptr = ex.getVector();
    tmp.Dot( *xvalptr, val );
    return (Real)val[0];
  }

  Real norm() const {
    double val[1];
    Epetra_MultiVector tmp(*(this->epetra_vec_)); 
    tmp.PutScalar(0.0);
    (this->M_)->Multiply(false,*(this->epetra_vec_),tmp);
    tmp.Dot( *(this->epetra_vec_), val );
    return (Real)sqrt(val[0]);
  }

  Teuchos::RCP<ROL::Vector<Real> > clone() const{
    return Teuchos::rcp(new ControlVector(
             Teuchos::rcp(new Epetra_MultiVector(epetra_vec_->Map(),epetra_vec_->NumVectors(),false)),
               this->M_ ));
  }

  void axpy( const Real alpha, const ROL::Vector<Real> &x ) {
    ControlVector &ex = Teuchos::dyn_cast<ControlVector>(const_cast <ROL::Vector<Real>&>(x));
    Teuchos::RCP<const Epetra_MultiVector> xvalptr = ex.getVector();
    epetra_vec_->Update( alpha, *xvalptr, 1.0 );
  }

  virtual void zero() {
    epetra_vec_->PutScalar(0.0);
  }

  virtual void set( const ROL::Vector<Real> &x ) {
    ControlVector &ex = Teuchos::dyn_cast<ControlVector>(const_cast <ROL::Vector<Real>&>(x));
    Teuchos::RCP<const Epetra_MultiVector> xvalptr = ex.getVector();
    epetra_vec_->Scale(1.0,*xvalptr);
  }

  Teuchos::RCP<const Epetra_MultiVector> getVector() const {
    return this->epetra_vec_;
  }

  Teuchos::RCP<ROL::Vector<Real> > basis( const int i ) const {
    Teuchos::RCP<ControlVector> e = 
      Teuchos::rcp( new ControlVector( Teuchos::rcp(
        new Epetra_MultiVector(epetra_vec_->Map(),epetra_vec_->NumVectors(),true)), this->M_ ));
    const Epetra_BlockMap & domainMap = const_cast <Epetra_BlockMap &> (
      const_cast <Epetra_MultiVector &> ((*e->getVector())).Map());

    // Build IntVector of GIDs on all processors.
    const Epetra_Comm & comm = domainMap.Comm();
    int numMyElements = domainMap.NumMyElements();
    Epetra_BlockMap allGidsMap(-1, numMyElements, 1, 0, comm);
    Epetra_IntVector allGids(allGidsMap);
    for (int j=0; j<numMyElements; j++) {allGids[j] = domainMap.GID(j);}

    // Import my GIDs into an all-inclusive map. 
    int numGlobalElements = domainMap.NumGlobalElements();
    Epetra_LocalMap allGidsOnRootMap(numGlobalElements, 0, comm);
    Epetra_Import importer(allGidsOnRootMap, allGidsMap);
    Epetra_IntVector allGidsOnRoot(allGidsOnRootMap);
    allGidsOnRoot.Import(allGids, importer, Insert);
    Epetra_Map rootDomainMap(-1, allGidsOnRoot.MyLength(), allGidsOnRoot.Values(), domainMap.IndexBase(), comm);

    for (int j = 0; j < this->dimension(); j++) {
      // Put 1's in slots
      int curGlobalCol = rootDomainMap.GID(i); // Should return same value on all processors
      if (domainMap.MyGID(curGlobalCol)){
        int curCol = domainMap.LID(curGlobalCol);
        (const_cast <Epetra_MultiVector &> (*e->getVector()))[0][curCol]= 1.0;
      }
    }

    return e;
  }

  int dimension() const {return epetra_vec_->GlobalLength();}
};

#endif
