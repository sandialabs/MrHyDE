/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

class rectPeriodicMatcher{

private:
    double tol_;
    
public:
    rectPeriodicMatcher(): tol_(1.e-8) {};
    rectPeriodicMatcher(const double & tol): tol_(tol) {};
    
    bool operator()(const Teuchos::Tuple<double,3> & a,
                    const Teuchos::Tuple<double,3> & b) const { 
        return ((std::fabs(a[1]-b[1])<tol_) || (std::fabs(a[0]-b[0])<tol_));
    }
    
    void setTol(double const & tol){
        tol_ = tol;
    }

    std::string getString() const { 
        std::stringstream ss;
        ss << "...not sure what this is for...";
        return ss.str();
    }
};
