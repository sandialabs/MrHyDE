/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "msphasefield.hpp"
#include <random>
#include <math.h>
#include <time.h>

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

msphasefield::msphasefield(Teuchos::RCP<Teuchos::ParameterList> & settings,
                           const Teuchos::RCP<MpiComm> & Comm_) :
Comm(Comm_) {
  
  spaceDim = settings->sublist("Mesh").get<int>("dim",2);
  numphases = settings->sublist("Physics").get<int>("number_phases",1);
  numdisks = settings->sublist("Physics").get<int>("numdisks",3);
  disksize = settings->sublist("Physics").get<ScalarT>("disksize",10.0);
  initialType = settings->sublist("Physics").get<string>("initialType","default");
  uniform = settings->sublist("Physics").get<bool>("uniform",true);
  variableMobility = settings->sublist("Physics").get<bool>("variableMobility",false);
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
  
  ScalarT tolerance = 2*disksize+5.0;  // 2 times the disk radius
  ScalarT xpos;
  ScalarT ypos;
  ScalarT zpos;
  
  
  if(initialType=="systematic") {       // extend to 3d
    xmax = settings->sublist("Mesh").get<ScalarT>("xmax",2);
    xmin = settings->sublist("Mesh").get<ScalarT>("xmin",2);
    ymax = settings->sublist("Mesh").get<ScalarT>("ymax",2);
    ymin = settings->sublist("Mesh").get<ScalarT>("ymin",2);
    
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
      zpos = dis(gen);
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
    if (spaceDim > 2)
      zpos = rand() % 100;
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

void msphasefield::defineFunctions(Teuchos::RCP<Teuchos::ParameterList> & settings,
                                   Teuchos::RCP<FunctionManager> & functionManager_) {
  
  functionManager = functionManager_;
  
}

// ========================================================================================
// ========================================================================================

void msphasefield::volumeResidual() {
  
  // NOTES:
  // 1. basis and basis_grad already include the integration weights
  
  int resindex;
  
  int numCubPoints = wkset->ip.extent(1);
  int phi_basis = wkset->usebasis[phi_num[0]];
  int numBasis = wkset->basis[phi_basis].extent(1);
  
  // FCAD local_resid(numphases, numBasis);
  
  //ScalarT diff_FAD = diff;
  ScalarT x = 0.0;
  ScalarT y = 0.0;
  ScalarT z = 0.0;
  //    ScalarT current_time = wkset->time;
  
  ScalarT v, dvdx, dvdy, dvdz;
  
  std::vector<AD>  phi;
  //    std::vector<AD>  phiInterface;
  std::vector<AD>  dphidx;
  std::vector<AD>  dphidy;
  std::vector<AD>  dphidz;
  std::vector<AD>  phi_dot;
  AD  sumphi;
  
  sol = wkset->local_soln;
  sol_dot = wkset->local_soln_dot;
  sol_grad = wkset->local_soln_grad;
  
  basis = wkset->basis[phi_basis];
  basis_grad = wkset->basis_grad[phi_basis];
  offsets = wkset->offsets;
  DRV ip = wkset->ip;
  res = wkset->res;
  wts = wkset->wts;
  
  for (size_t e=0; e<res.extent(0); e++) {
    for( int k=0; k<ip.extent(1); k++ ) {
      x = ip(e,k,0);
      
      sumphi = 0.0;
      for(int j=0; j<numphases; j++) {
        phi.push_back(sol(e,phi_num[j],k,0));
        phi_dot.push_back(sol_dot(e,phi_num[j],k,0));
        dphidx.push_back(sol_grad(e,phi_num[j],k,0));
        
        if (spaceDim > 1) {
          y = ip(e,k,1);
          dphidy.push_back(sol_grad(e,phi_num[j],k,1));
        }
        if (spaceDim > 2) {
          z = ip(e,k,2);
          dphidz.push_back(sol_grad(e,phi_num[j],k,2));
        }
        sumphi +=  phi[j]*phi[j];
      }
      
      
      AD Lnum;
      AD Lden;
      AD mobility;
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
        if(Lden < 1E-8) {
          mobility = 0.01;
        } else {
          mobility = Lnum/Lden;
          //	  cout << mobility.val() << endl;
        }
      }
      
      for( int i=0; i<basis.extent(1); i++ ) {
        v = basis(e,i,k);
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

void msphasefield::boundaryResidual() {
  
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

void msphasefield::edgeResidual() {
  
}

// ========================================================================================
// The boundary/edge flux
// ========================================================================================

void msphasefield::computeFlux() {
  
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
      + penalty*(wkset->local_aux_side(e,phi_num[0],i)-wkset->local_soln_side(e,phi_num[0],i,0));
      if (spaceDim > 1)
        wkset->flux(e,phi_num[0],i) += diff*wkset->local_soln_grad_side(e,phi_num[0],i,1)*wkset->normals(e,i,1);
      if (spaceDim > 2)
        wkset->flux(e,phi_num[0],i) += diff*wkset->local_soln_grad_side(e,phi_num[0],i,2)*wkset->normals(e,i,2);
    }
  }
}

// ========================================================================================
// ========================================================================================

void msphasefield::setVars(std::vector<string> & varlist) {
  for (size_t i=0; i<varlist.size(); i++) {
    for (size_t j=1; j<numphases+1; j++) {
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

AD msphasefield::SourceTerm(const ScalarT & x, const ScalarT & y, const ScalarT & z,
                            const std::vector<AD > & tsource) const {
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

ScalarT msphasefield::boundarySource(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & t,
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

AD msphasefield::DiffusionCoeff(const ScalarT & x, const ScalarT & y, const ScalarT & z) const {
  AD diff = 0.0;
  diff = diff_FAD[0];
  return diff;
}

// ========================================================================================
/* return the source term (to be multiplied by test_function) */
// ========================================================================================

ScalarT msphasefield::robinAlpha(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & t,
                                 const string & side) const {
  return 0.0;
}

// ========================================================================================
// TMW: this is deprecated
// ========================================================================================

void msphasefield::updateParameters(const vector<Teuchos::RCP<vector<AD> > > & params,
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
