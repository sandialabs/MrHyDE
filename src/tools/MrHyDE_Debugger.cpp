/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "MrHyDE_Debugger.hpp"
#include "kokkosTools.hpp"

using namespace MrHyDE;

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

MrHyDE_Debugger::MrHyDE_Debugger(const int & debug_level, const Teuchos::RCP<MpiComm> & comm) :
debug_level_(debug_level), comm_(comm) {
  // Nothing else to do
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

void MrHyDE_Debugger::print(const std::string & message){
  if (debug_level_ > 0 && comm_->getRank() == 0) {
    std::cout << message << std::endl;
  }
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

void MrHyDE_Debugger::print(const int & threshhold, const std::string & message){
  if (debug_level_ > threshhold && comm_->getRank() == 0) {
    std::cout << message << std::endl;
  }
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

template<class T>
void MrHyDE_Debugger::print(T data, const std::string & message){
  if (debug_level_ > 2 && comm_->getRank() == 0) {
    KokkosTools::print(data, message);
  }
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

template<class T>
void MrHyDE_Debugger::print(const int & threshhold, T data, const std::string & message){
  if (debug_level_ > threshhold && comm_->getRank() == 0) {
    KokkosTools::print(data, message);
  }
}
