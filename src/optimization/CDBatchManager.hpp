/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef CDBATCHMANAGER_HPP
#define CDBATCHMANAGER_HPP

#include "ROL_EpetraBatchManager.hpp"
#include "control_vector.hpp"

template<class Real> 
class CDBatchManager : public ROL::EpetraBatchManager<Real> {
public:
  CDBatchManager(Teuchos::RCP<Epetra_Comm> &comm) : ROL::EpetraBatchManager<Real>(comm) {}
  void sumAll(ROL::Vector<Real> &input, ROL::Vector<Real> &output) {
/*
     Teuchos::RCP<Epetra_MultiVector> input_ptr = Teuchos::rcp_const_cast<Epetra_MultiVector>(
                                       (Teuchos::dyn_cast<ControlVector<Real> >(input)).getVector());
     Teuchos::RCP<Epetra_MultiVector> output_ptr = Teuchos::rcp_const_cast<Epetra_MultiVector>(
    			 	                    (Teuchos::dyn_cast<ControlVector<Real> >(output)).getVector());
            //dynamic casting fails...

    //this was previously the version uncommented, but StdVector doesn't have "Values" and "MyLength" so wouldn't compile...
    //Teuchos::RCP<ROL::StdVector<Real> > input_ptr = Teuchos::rcp_const_cast<ROL::StdVector<Real> >(
    //                                  (Teuchos::dyn_cast<ControlVector<Real> >(input)).getVector());
    //Teuchos::RCP<ROL::StdVector<Real> > output_ptr = Teuchos::rcp_const_cast<ROL::StdVector<Real> >(
	//		 	                      (Teuchos::dyn_cast<ControlVector<Real> >(output)).getVector());
	
    ROL::EpetraBatchManager<Real>::sumAll(input_ptr->Values(),output_ptr->Values(),input_ptr->MyLength());
*/
    Teuchos::RCP<std::vector<Real> > invec =
        (Teuchos::dyn_cast<ROL::StdVector<Real> >(const_cast<ROL::Vector<Real> &>(input))).getVector();
    Teuchos::RCP<std::vector<Real> > outvec =
        (Teuchos::dyn_cast<ROL::StdVector<Real> >(const_cast<ROL::Vector<Real> &>(output))).getVector();
    ROL::EpetraBatchManager<Real>::sumAll(&(*invec)[0],&(*outvec)[0],invec->size());
  }
  
};

#endif
