/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "msphasefield.hpp"
#include <random>
#include <math.h>
#include <time.h>
using namespace MrHyDE;

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

template<class EvalT>
msphasefield<EvalT>::msphasefield(Teuchos::ParameterList & settings, const int & dimension_,
                           const Teuchos::RCP<MpiComm> & Comm_)
  : PhysicsBase<EvalT>(settings, dimension_),
  Comm(Comm_)
{
  
  spaceDim = dimension_;
  numphases = settings.get<int>("number_phases",1);
  numdisks = settings.get<int>("numdisks",3);
  disksize = settings.get<ScalarT>("disksize",10.0);
  initialType = settings.get<string>("initialType","default");
  uniform = settings.get<bool>("uniform",true);
  variableMobility = settings.get<bool>("variableMobility",false);
  //systematic = settings->sublist("Physics").get<bool>("systematic",true);
  
  string name = "phi";
  for (int i=1;i<=numphases;i++) {
    string vartemp = name + std::to_string(i);
    myvars.push_back(vartemp);
  }
  for(int i=1;i<=numphases;i++) {
    mybasistypes.push_back("HGRAD");
  }
  
  // generation of disks for initial condition
  
  //ScalarT tolerance = 2*disksize+5.0;  // 2 times the disk radius
  ScalarT xpos = 0.0;
  ScalarT ypos = 0.0;
  //ScalarT zpos = 0.0;
  
  
  if(initialType=="systematic") {       // extend to 3d
    xmax = settings.get<ScalarT>("xmax",2);
    xmin = settings.get<ScalarT>("xmin",2);
    ymax = settings.get<ScalarT>("ymax",2);
    ymin = settings.get<ScalarT>("ymin",2);
    
    // works only for power of 2
    int deldisks = pow(numdisks,0.5);
    ScalarT delx = xmax/deldisks;
    ScalarT dely = ymax/deldisks;
    ScalarT intvx = xmax/(deldisks*2);
    ScalarT intvy = ymax/(deldisks*2);
    
    for (int i=0; i<deldisks; i++) {
      for (int j=0; j<deldisks; j++) {
        xpos = intvx + delx*i;
        ypos = intvy + dely*j;
        disk.push_back(xpos);
        disk.push_back(ypos);
        //	  cout << "i = " << i << "  j  " << j << "  xpos = " << xpos << "  ypos = " << ypos << endl;
      }
    }
  } else if(uniform) {    // generate random position
    //	std::random_device rd;
    //	std::mt19937 gen(rd());
    std::mt19937 gen(time(0));
    //std::default_random_engine gen;
    std::uniform_real_distribution<ScalarT> dis(9, 89);
    xpos = dis(gen);
    ypos = dis(gen);
    disk.push_back(xpos);
    disk.push_back(ypos);
    if (spaceDim > 2) {
      //zpos = dis(gen);
      //	cout << "proc  " << Comm->MyPID() << " xpos " << xpos << " ypos " << ypos << " zpos " << zpos << endl;
    } else {
      //	cout << "proc  " << Comm->MyPID() << " xpos " << xpos << " ypos " << ypos << endl;
    }
  } else {
    //	srand(1);
    srand(time(NULL));
    xpos = rand() % 100;
    ypos = rand() % 100;
    disk.push_back(xpos);
    disk.push_back(ypos);
    if (spaceDim > 2) {
      //zpos = rand() % 100;
    }
    // pushdback to a vector, maybe single vector with x1,y1,x2, y2, etc
  }
  //    disk.push_back(xpos);
  //  disk.push_back(ypos);
  // if (spaceDim > 2)
  //   disk.push_back(zpos);
  //  int count = 0;
  // int cntrandom = 0;
  // while(count < numdisks-1) {
  //   // generate next random position
  //   if(uniform) {     //uniform
  // 	std::random_device rd;
  // 	//	  std::mt19937 gen(rd());
  // 	std::mt19937 gen(cntrandom);
  // 	//std::default_random_engine gen;
  // 	std::uniform_real_distribution<> dis(9, 89);
  // 	xpos = dis(gen);
  // 	ypos = dis(gen);
  // 	if (spaceDim > 2)
  // 	  zpos = dis(gen);
  // 	cntrandom++;
  //   } else {
  // 	srand(time(NULL));
  // 	xpos = rand() % 100;
  // 	ypos = rand() % 100;
  // 	if (spaceDim > 2)
  // 	  zpos = rand() % 100;
  //   }
  // compare to all previous entries and if distance is smaller than tolerance, reject
  // and repeat
  // bool test = false;
  
  // if (spaceDim == 2) {
  // 	for (int j=0; j < disk.size() ; j=j+2) {
  // 	  ScalarT tempdist = (xpos-disk[j])*(xpos-disk[j])+(ypos-disk[j+1])*(ypos-disk[j+1]);
  // 	  ScalarT distance = pow(tempdist,0.5);
  // 	  if(distance > tolerance) {
  // 	    test = true;
  // 	    break;
  // 	  }
  // 	}
  // 	// if(test ==true) {
  // 	//   cout << "proc  " << Comm->MyPID() << " xpos " << xpos << " ypos " << ypos << endl;
  // 	//   disk.push_back(xpos);
  // 	//   disk.push_back(ypos);
  // 	//   count++;
  // 	// }
  //   }
  //   if (spaceDim == 3) {
  // 	for (int j=0; j < disk.size() ; j=j+3) {
  // 	  ScalarT tempdist = (xpos-disk[j])*(xpos-disk[j])+(ypos-disk[j+1])*(ypos-disk[j+1])
  // 	    + (zpos-disk[j+2])*(zpos-disk[j+2]);
  // 	  ScalarT distance = pow(tempdist,0.5);
  // 	  if(distance > tolerance) {
  // 	    test = true;
  // 	    break;
  // 	  }
  // 	}
  // 	if(test ==true) {
  // 	  cout << "proc  " << Comm->MyPID() << " xpos " << xpos << " ypos " << ypos
  // 	       << " zpos " << zpos << endl;
  // 	  disk.push_back(xpos);
  // 	  disk.push_back(ypos);
  // 	  disk.push_back(zpos);
  // 	  count++;
  // 	}
  //   }
  // }
  
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void msphasefield<EvalT>::defineFunctions(Teuchos::ParameterList & fs,
                                   Teuchos::RCP<FunctionManager<EvalT> > & functionManager_) {
  
  functionManager = functionManager_;
  
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void msphasefield<EvalT>::volumeResidual() {
  
  // NOTES:
  // 1. basis and basis_grad already include the integration weights
  
  int resindex;
  int spaceDim = wkset->dimension;
  
  //int numCubPoints = wkset->ip.extent(1);
  int phi_basis = wkset->usebasis[phi_num[0]];
  //int numBasis = wkset->basis[phi_basis].extent(1);
  
  // FCAD local_resid(numphases, numBasis);
  
  //ScalarT diff_FAD = diff;
  //ScalarT x = 0.0;
  //ScalarT y = 0.0;
  //ScalarT z = 0.0;
  //    ScalarT current_time = wkset->time;
  
  ScalarT v = 0.0, dvdx = 0.0, dvdy = 0.0, dvdz = 0.0;
  
  std::vector<EvalT>  phi;
  //    std::vector<EvalT>  phiInterface;
  std::vector<EvalT>  dphidx;
  std::vector<EvalT>  dphidy;
  std::vector<EvalT>  dphidz;
  std::vector<EvalT>  phi_dot;
  EvalT  sumphi = 0.0;
  
  vector<View_EvalT2> sol, sol_dot, dsol_dx, dsol_dy, dsol_dz;
  
  for (size_t k=0; k<myvars.size(); k++) {
    sol.push_back(wkset->getSolutionField(myvars[k]));
    sol_dot.push_back(wkset->getSolutionField(myvars[k]+"_t"));
    dsol_dx.push_back(wkset->getSolutionField("grad("+myvars[k]+")[x]"));
    if (spaceDim > 1) {
      dsol_dy.push_back(wkset->getSolutionField("grad("+myvars[k]+")[y]"));
    }
    if (spaceDim > 2) {
      dsol_dz.push_back(wkset->getSolutionField("grad("+myvars[k]+")[z]"));
    }
  }
  
  auto basis = wkset->basis[phi_basis];
  auto basis_grad = wkset->basis_grad[phi_basis];
  auto offsets = wkset->offsets;
  //auto ip = wkset->ip;
  auto res = wkset->res;
  auto wts = wkset->wts;
  
  for (size_type e=0; e<basis.extent(0); e++) {
    for(size_type k=0; k<basis.extent(2); k++ ) {
      //x = ip(e,k,0);
      
      sumphi = 0.0;
      for(int j=0; j<numphases; j++) {
        phi.push_back(sol[j](e,k));
        phi_dot.push_back(sol_dot[j](e,k));
        dphidx.push_back(dsol_dx[j](e,k));
        
        if (spaceDim > 1) {
          //y = ip(e,k,1);
          dphidy.push_back(dsol_dy[j](e,k));
        }
        if (spaceDim > 2) {
          //z = ip(e,k,2);
          dphidz.push_back(dsol_dz[j](e,k));
        }
        sumphi +=  phi[j]*phi[j];
      }
      
      EvalT Lnum = 0.0;
      EvalT Lden = 0.0;
      EvalT mobility = 0.0;
      //      std::mt19937 gen(time(0));
      //      std::uniform_real_distribution<ScalarT> dis(-0.1, 0.1);
      if(variableMobility) {
        for(int i=0; i<numphases; i++) {
          for(int j=0; j<numphases; j++) {
            //	    ScalarT Lij = dis(gen);
            Lnum += L[i*numphases+j] * phi[i] * phi[i] * phi[j] * phi[j];
            Lden +=           phi[i] * phi[i] * phi[j] * phi[j];
          }
        }
//#ifndef MrHyDE_NO_AD
//        ScalarT mcheck = Lden;//.val();
//#else
//        ScalarT mcheck = Lden;
//#endif
        if (Lden < 1E-8) {
          mobility = 0.01;
        } else {
          mobility = Lnum/Lden;
          //	  cout << mobility.val() << endl;
        }
      }
      
      for( size_type i=0; i<basis.extent(1); i++ ) {
        v = basis(e,i,k,0);
        dvdx = basis_grad(e,i,k,0);
        dvdy = basis_grad(e,i,k,1);
        if (spaceDim == 3)
          dvdz = basis_grad(e,i,k,2);
        
        for(int j=0; j<numphases; j++) {
          resindex = offsets(phi_num[j],i);
          //          local_resid(j,i) += L[0]*(16.0*A[0]*phi[j]*(-phi[j]+ sumphi)*v +
          //                                       diff_FAD[0]*diff_FAD[0]*(dphidx[j]*dvdx +
          //                                                      dphidy[j]*dvdy ));
          
          if (spaceDim == 2) {
            //	    if(variableMobility) L[0] = Lvar[j];
            if(variableMobility) {
              res(e,resindex) += (mobility*(16.0*A[0]*phi[j]*(-phi[j]+ sumphi)*v +
                                           diff_FAD[0]*diff_FAD[0]*(dphidx[j]*dvdx + dphidy[j]*dvdy )))*wts(e,k);
            }else {
              res(e,resindex) += (L[0]*(16.0*A[0]*phi[j]*(-phi[j]+ sumphi)*v +
                                       diff_FAD[0]*diff_FAD[0]*(dphidx[j]*dvdx + dphidy[j]*dvdy )))*wts(e,k);
            }
            //	      wkset->res(resindex) += L[0]*(4.0*A[0]*(-phi[j]+ phi[j]*phi[j]*phi[j])*v +
            //					   diff_FAD[0]*diff_FAD[0]*(dphidx[j]*dvdx + dphidy[j]*dvdy ));
            
          }
          if (spaceDim == 3) {
            res(e,resindex) += (L[0]*(4.0*A[0]*phi[j]*(-phi[j]+ sumphi)*v +
                                     diff_FAD[0]*diff_FAD[0]*(dphidx[j]*dvdx + dphidy[j]*dvdy + dphidz[j]*dvdz )))*wts(e,k);
          }
          res(e,resindex) += phi_dot[j]*v*wts(e,k);
        }
      }
    }
  }
  
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void msphasefield<EvalT>::boundaryResidual() {
  
  //TMW: NOT BEEN UPDATED TO NXTGEN
  
  // NOTES:
  // 1. basis and basis_grad already include the integration weights
  
  //    int numCubPoints = ip.extent(1);
  // int phi_basis = usebasis[phi_num];
  // int numBasis = basis[phi_basis].extent(1);
  
  //    FCAD local_resid(numphases, numBasis);
  
  // Set the parameters
  // ScalarT x = 0.0;
  // ScalarT y = 0.0;
  // ScalarT z = 0.0;
  
  // ScalarT v, source, robin_alpha;
  // AD phi, dphidx, dphidy, dphidz;
  // std::string pname = myparams[0];
  // std::vector<AD > diff_FAD = getParameters(params, pname);
  
  // ScalarT dvdx;
  
  // for (size_t ee=0; ee<numCC; ee++) {
  //    for( int i=0; i<numBasis; i++ ) {
  //       for( int k=0; k<numSideCubPoints; k++ ) {
  //          v = basis(ee,i,k);
  //          x = ip(ee,k,0);
  //          phi = local_soln(ee,phi_num,k);
  //          dphidx = local_solngrad(ee,phi_num,k,0);
  //          dvdx = basis_grad(ee,i,k,0);
  
  //          if (spaceDim > 1) {
  //             y = ip(ee,k,1);
  //             dphidy = local_solngrad(ee,phi_num,k,1);
  //          }
  //          if (spaceDim > 2) {
  //             z = ip(ee,k,2);
  //             dphidz = local_solngrad(ee,phi_num,k,2);
  //          }
  //          source = this->boundarySource(x, y, z, current_time, side_name);
  //          robin_alpha = this->robinAlpha(x, y, z, current_time, side_name);
  
  //          local_resid(ee,i) += -diff_FAD[0]*dphidx*normals(ee,k,0)*v;
  //          if (spaceDim > 1) {
  //             local_resid(ee,i) += -diff_FAD[0]*dphidy*normals(ee,k,1)*v;
  //          }
  //          if (spaceDim > 2) {
  //             local_resid(ee,i) += -diff_FAD[0]*dphidz*normals(ee,k,2)*v;
  //          }
  
  //       }
  //    }
  // }
  
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void msphasefield<EvalT>::edgeResidual() {
  
}

// ========================================================================================
// The boundary/edge flux
// ========================================================================================

template<class EvalT>
void msphasefield<EvalT>::computeFlux() {
  
  /*
  ScalarT x = 0.0;
  ScalarT y = 0.0;
  ScalarT z = 0.0;
  
  for (size_t e=0; e<flux.extent(0); e++) {
    for (size_t i=0; i<wkset->ip_side.extent(1); i++) {
      x = wkset->ip_side(e,i,0);
      if (spaceDim > 1)
        y = wkset->ip_side(e,i,1);
      if (spaceDim > 2)
        z = wkset->ip_side(e,i,2);
      
      
      AD diff = DiffusionCoeff(x,y,z);
      AD penalty = 10.0*diff/wkset->h(e);
      wkset->flux(e,phi_num[0],i) += diff*wkset->local_soln_grad_side(e,phi_num[0],i,0)*wkset->normals(e,i,0)
      + penalty*(wkset->local_aux_side(e,phi_num[0],i,0)-wkset->local_soln_side(e,phi_num[0],i,0));
      if (spaceDim > 1)
        wkset->flux(e,phi_num[0],i) += diff*wkset->local_soln_grad_side(e,phi_num[0],i,1)*wkset->normals(e,i,1);
      if (spaceDim > 2)
        wkset->flux(e,phi_num[0],i) += diff*wkset->local_soln_grad_side(e,phi_num[0],i,2)*wkset->normals(e,i,2);
    }
  }
   */
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void msphasefield<EvalT>::setWorkset(Teuchos::RCP<Workset<EvalT> > & wkset_) {

  wkset = wkset_;
  vector<string> varlist = wkset->varlist;
  for (size_t i=0; i<varlist.size(); i++) {
    for (int j=1; j<numphases+1; j++) {
      //	std::string name = "phi";
      // string vartemp = name + std::to_string(j);
      string varphasetemp = "phi" + std::to_string(j);
      if (varlist[i] == varphasetemp)
        phi_num.push_back(i);
    }
  }
}

// ========================================================================================
/* return the source term (to be multiplied by test_function) */
// ========================================================================================

template<class EvalT>
EvalT msphasefield<EvalT>::SourceTerm(const ScalarT & x, const ScalarT & y, const ScalarT & z,
                            const std::vector<EvalT > & tsource) const {
  
  int spaceDim = wkset->dimension;
  if(spaceDim == 1) {
    return tsource[0]*12*PI*PI*sin(2*PI*x);
  }else if (spaceDim == 2) {
    //return (tsource[0] + 2.0*tsource[1] + 4.0*tsource[2])*8*PI*PI*sin(2*PI*x)*sin(2*PI*y);
    return 8*PI*PI*sin(2*PI*x)*sin(2*PI*y);
  } else {
    return tsource[0]*12*PI*PI*sin(2*PI*x)*sin(2*PI*y)*sin(2*PI*z);
  }
}

// ========================================================================================
/* return the source term (to be multiplied by test_function) */
// ========================================================================================

template<class EvalT>
ScalarT msphasefield<EvalT>::boundarySource(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & t,
                                     const string & side) const {
  
  ScalarT val = 0.0;
  
  if (side == "top")
    val = 2*PI*sin(2*PI*x)*cos(2*PI*y);
  
  if (side == "bottom")
    val = -2*PI*sin(2*PI*x)*cos(2*PI*y);
  
  return val;
}

// ========================================================================================
/* return the diffusivity coefficient */
// ========================================================================================

template<class EvalT>
EvalT msphasefield<EvalT>::DiffusionCoeff(const ScalarT & x, const ScalarT & y, const ScalarT & z) const {
  EvalT diff = 0.0;
  diff = diff_FAD[0];
  return diff;
}

// ========================================================================================
/* return the source term (to be multiplied by test_function) */
// ========================================================================================

template<class EvalT>
ScalarT msphasefield<EvalT>::robinAlpha(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & t,
                                 const string & side) const {
  return 0.0;
}

// ========================================================================================
// TMW: this is deprecated
// ========================================================================================

template<class EvalT>
void msphasefield<EvalT>::updateParameters(const vector<Teuchos::RCP<vector<EvalT> > > & params,
                                    const vector<string> & paramnames) {
  
  for (size_t p=0; p<paramnames.size(); p++) {
    if (paramnames[p] == "thermal_diff")
      diff_FAD = *(params[p]);
    else if (paramnames[p] == "L")
      L = *(params[p]);
    else if (paramnames[p] == "A")
      A = *(params[p]);
    
    //else
    //  cout << "Parameter not used: " << paramnames[p] << endl;
  }
}


//////////////////////////////////////////////////////////////
// Explicit template instantiations
//////////////////////////////////////////////////////////////

template class MrHyDE::msphasefield<ScalarT>;

#ifndef MrHyDE_NO_AD
// Custom AD type
template class MrHyDE::msphasefield<AD>;

// Standard built-in types
template class MrHyDE::msphasefield<AD2>;
template class MrHyDE::msphasefield<AD4>;
template class MrHyDE::msphasefield<AD8>;
template class MrHyDE::msphasefield<AD16>;
template class MrHyDE::msphasefield<AD18>;
template class MrHyDE::msphasefield<AD24>;
template class MrHyDE::msphasefield<AD32>;
#endif
