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

namespace MrHyDE {
  
  class rectPeriodicMatcher{
    
  private:
    ScalarT tol_;
    
  public:
    rectPeriodicMatcher(): tol_(1.e-8) {};
    rectPeriodicMatcher(const ScalarT & tol): tol_(tol) {};
    
    bool operator()(const Teuchos::Tuple<ScalarT,3> & a,
                    const Teuchos::Tuple<ScalarT,3> & b) const {
      return ((std::fabs(a[1]-b[1])<tol_) || (std::fabs(a[0]-b[0])<tol_));
    }
    
    void setTol(ScalarT const & tol){
      tol_ = tol;
    }
    
    std::string getString() const { 
      std::stringstream ss;
      ss << "...not sure what this is for...";
      return ss.str();
    }
  };
  
}
