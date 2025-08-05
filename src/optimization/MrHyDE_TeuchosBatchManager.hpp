/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.

 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
 ************************************************************************/

#ifndef MRHYDETEUCHOSBATCHMANAGER_HPP
#define MRHYDETEUCHOSBATCHMANAGER_HPP

#include "ROL_TeuchosBatchManager.hpp"

namespace ROL {

template<class Real, class Ordinal>
class MrHyDETeuchosBatchManager : public TeuchosBatchManager<Real,Ordinal> {
public:
  MrHyDETeuchosBatchManager(const ROL::Ptr<const Teuchos::Comm<int>> &comm)
    : TeuchosBatchManager<Real,Ordinal>(comm) {}

  using ROL::TeuchosBatchManager<Real,Ordinal>::sumAll;
  void sumAll(Vector<Real> &input, Vector<Real> &output) {
    MrHyDE_OptVector input_mrhyde_vec = dynamic_cast<MrHyDE_OptVector&>(input);
    MrHyDE_OptVector output_mrhyde_vec = dynamic_cast<MrHyDE_OptVector&>(output);

    // Sum all field components across processors
    std::vector<ROL::Ptr<ROL::TpetraMultiVector<Real,LO,GO,SolverNode> > > input_field_ptr 
      = input_mrhyde_vec.getField();
    std::vector<ROL::Ptr<ROL::TpetraMultiVector<Real,LO,GO,SolverNode> > > output_field_ptr
      = output_mrhyde_vec.getField();

    if ( input_field_ptr.size() > 0 ) {
      ROL::Ptr<Tpetra::MultiVector<Real,LO,GO,SolverNode>> input_field  = input_field_ptr[0]->getVector();
      ROL::Ptr<Tpetra::MultiVector<Real,LO,GO,SolverNode>> output_field = output_field_ptr[0]->getVector();
      size_t input_length  = input_field->getLocalLength();
      size_t output_length = output_field->getLocalLength();
      TEUCHOS_TEST_FOR_EXCEPTION(input_length != output_length, std::invalid_argument,
        ">>> (BatchManager::sumAll): Field dimension mismatch!");

      size_t input_nvec  = input_field->getNumVectors();
      size_t output_nvec = output_field->getNumVectors();
      TEUCHOS_TEST_FOR_EXCEPTION(input_nvec != output_nvec, std::invalid_argument,
        ">>> (BatchManager::sumAll): Field dimension mismatch!");

      for (size_t i = 0; i < input_nvec; ++i) {
        ROL::TeuchosBatchManager<Real,Ordinal>::sumAll((input_field->getDataNonConst(i)).getRawPtr(),
                                                  (output_field->getDataNonConst(i)).getRawPtr(),
                                                  input_length);
      }
    }
    // Sum all parameter components across processors
    std::vector<ROL::Ptr<ROL::StdVector<Real> > > input_param_ptr
      = input_mrhyde_vec.getParameter();
    std::vector<ROL::Ptr<ROL::StdVector<Real> > > output_param_ptr
      = output_mrhyde_vec.getParameter();

    if ( input_param_ptr.size() > 0 ) {
      ROL::Ptr<std::vector<Real>> input_param  = input_param_ptr[0]->getVector();
      ROL::Ptr<std::vector<Real>> output_param = output_param_ptr[0]->getVector();
      size_t input_size  = static_cast<size_t>(input_param->size());
      size_t output_size = static_cast<size_t>(output_param->size());
      TEUCHOS_TEST_FOR_EXCEPTION(input_size != output_size, std::invalid_argument,
        ">>> (BatchManager::SumAll): Parameter dimension mismatch!");

      ROL::TeuchosBatchManager<Real,Ordinal>::sumAll(&input_param->front(),
                                                &output_param->front(),
                                                input_size);
    }
  }
};

}

#endif
