/***********************************************************************
The reduced Hessian as a Tpetra::Operator, so Trilinos consumers can probe it:
MatrixMarket::Writer::writeOperator forms the matrix, Anasazi takes its spectrum.

 Questions? Contact Alexey Voronin
 ************************************************************************/

#ifndef MRHYDE_HESSIAN_OPERATOR_HPP
#define MRHYDE_HESSIAN_OPERATOR_HPP

#include "trilinos.hpp"
#include "preferences.hpp"
#include "MrHyDE_OptVector.hpp"
#include "param_riesz/MrHyDE_ParamRiesz.hpp"
#include "ROL_Objective.hpp"

class MrHyDE_HessianOperator : public Tpetra::Operator<ScalarT,LO,GO,SolverNode> {

  using LA_Map         = Tpetra::Map<LO,GO,SolverNode>;
  using LA_MultiVector = Tpetra::MultiVector<ScalarT,LO,GO,SolverNode>;
  using vector_RCP     = Teuchos::RCP<LA_MultiVector>;

public:

  // Control space is slab-major: gid = slab*n + parameter gid, local index = slab*nloc + i.
  MrHyDE_HessianOperator(const Teuchos::RCP<ROL::Objective<ScalarT> > & obj,
                         const Teuchos::RCP<ROL::Vector<ScalarT> > & x,
                         const Teuchos::RCP<const LA_Map> & param_map,
                         const ScalarT & fd_step,
                         const Teuchos::RCP<MrHyDE_ParamRiesz> & riesz)
  : obj_(obj), x_(x), param_map_(param_map), fd_step_(fd_step), riesz_(riesz) {

    v_  = x->clone();
    hv_ = x->clone();
    vf_  = Teuchos::dyn_cast<MrHyDE_OptVector>(*v_).getField();
    hvf_ = Teuchos::dyn_cast<MrHyDE_OptVector>(*hv_).getField();

    const size_t nslabs = vf_.size();
    const GO n = static_cast<GO>(param_map->getGlobalNumElements());
    nloc_ = param_map->getLocalNumElements();

    std::vector<GO> gids;
    gids.reserve(nslabs*nloc_);
    for (size_t s=0; s<nslabs; ++s) {
      for (size_t i=0; i<nloc_; ++i) {
        gids.push_back(static_cast<GO>(s)*n + param_map->getGlobalElement(static_cast<LO>(i)));
      }
    }
    map_ = Teuchos::rcp(new LA_Map(static_cast<Tpetra::global_size_t>(nslabs)*n,
                                   Teuchos::ArrayView<const GO>(gids),0,param_map->getComm()));

    // apply() overwrites every field entry, so only the scalar block needs zeroing once.
    v_->zero();

    if (!riesz_.is_null()) { w_ = Teuchos::rcp(new LA_MultiVector(param_map,1)); }
  }

  Teuchos::RCP<const LA_Map> getDomainMap() const override { return map_; }
  Teuchos::RCP<const LA_Map> getRangeMap() const override { return map_; }

  // P^-1 H is self-adjoint only in the P inner product, so there is no Euclidean transpose.
  bool hasTransposeApply() const override { return riesz_.is_null(); }

  void apply(const LA_MultiVector & X, LA_MultiVector & Y,
             Teuchos::ETransp mode = Teuchos::NO_TRANS,
             ScalarT alpha = Teuchos::ScalarTraits<ScalarT>::one(),
             ScalarT beta = Teuchos::ScalarTraits<ScalarT>::zero()) const override {

    TEUCHOS_TEST_FOR_EXCEPTION(mode != Teuchos::NO_TRANS && !riesz_.is_null(),
                               std::runtime_error,
                               "Error: MrHyDE_HessianOperator has no transpose in the Riesz metric.");

    const size_t nslabs = vf_.size();

    for (size_t c=0; c<X.getNumVectors(); ++c) {

      for (size_t s=0; s<nslabs; ++s) {
        vf_[s]->getVector()->assign(*X.getVector(c)->offsetView(param_map_,s*nloc_));
      }

      ScalarT tol = fd_step_;
      obj_->hessVec(*hv_,*v_,*x_,tol);

      for (size_t s=0; s<nslabs; ++s) {
        vector_RCP h = hvf_[s]->getVector();
        if (!riesz_.is_null()) { riesz_->applyInverse(h,w_); h = w_; }
        Y.getVectorNonConst(c)->offsetViewNonConst(param_map_,s*nloc_)->update(alpha,*h,beta);
      }
    }
  }

private:

  Teuchos::RCP<ROL::Objective<ScalarT> > obj_;
  Teuchos::RCP<ROL::Vector<ScalarT> > x_, v_, hv_;
  std::vector<ROL::Ptr<ROL::TpetraMultiVector<ScalarT> > > vf_, hvf_;
  vector_RCP w_;
  Teuchos::RCP<const LA_Map> map_, param_map_;
  size_t nloc_;
  ScalarT fd_step_;
  Teuchos::RCP<MrHyDE_ParamRiesz> riesz_;

};

#endif
