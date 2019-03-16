/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef MSPHASEFIELD_H
#define MSPHASEFIELD_H

#include "physics_base.hpp"
#include <random>
#include <math.h>
#include <time.h>

static void msphasefieldHelp() {
  cout << "********** Help and Documentation for the Multi-species Phase Field Physics Module **********" << endl << endl;
  cout << "Model:" << endl << endl;
  cout << "User defined functions: " << endl << endl;
}

class msphasefield : public physicsbase {
public:
  
  msphasefield() {} ;
  
  ~msphasefield() {};
  
  // ========================================================================================
  /* Constructor to set up the problem */
  // ========================================================================================
  
  msphasefield(Teuchos::RCP<Teuchos::ParameterList> & settings,
               const Teuchos::RCP<LA_MpiComm> & Comm_, const int & numip_,
               const size_t & numip_side_, const int & numElem_,
               Teuchos::RCP<FunctionInterface> & functionManager_,
               const size_t & blocknum_) :
  Comm(Comm_), numip(numip_), numip_side(numip_side_), numElem(numElem_),
  functionManager(functionManager_), blocknum(blocknum_) {
    
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
    
    if (settings->sublist("Solver").get<string>("solver","steady-state") == "transient")
      isTD = true;
    else
      isTD = false;
    
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
    
    multiscale = settings->isSublist("Subgrid");
    analysis_type = settings->sublist("Analysis").get<string>("analysis type","forward");
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void volumeResidual() {
    
    // NOTES:
    // 1. basis and basis_grad already include the integration weights
    
    int resindex;
    
    int numCubPoints = wkset->ip.dimension(1);
    int phi_basis = wkset->usebasis[phi_num[0]];
    int numBasis = wkset->basis[phi_basis].dimension(1);
    
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
    
    for (size_t e=0; e<numElem; e++) {
      for( int k=0; k<ip.dimension(1); k++ ) {
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
        
        for( int i=0; i<basis.dimension(1); i++ ) {
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
                res(e,resindex) += mobility*(16.0*A[0]*phi[j]*(-phi[j]+ sumphi)*v +
                                                  diff_FAD[0]*diff_FAD[0]*(dphidx[j]*dvdx + dphidy[j]*dvdy ));
              }else {
                res(e,resindex) += L[0]*(16.0*A[0]*phi[j]*(-phi[j]+ sumphi)*v +
                                              diff_FAD[0]*diff_FAD[0]*(dphidx[j]*dvdx + dphidy[j]*dvdy ));
              }
              //	      wkset->res(resindex) += L[0]*(4.0*A[0]*(-phi[j]+ phi[j]*phi[j]*phi[j])*v +
              //					   diff_FAD[0]*diff_FAD[0]*(dphidx[j]*dvdx + dphidy[j]*dvdy ));
              
            }
            if (spaceDim == 3) {
              res(e,resindex) += L[0]*(4.0*A[0]*phi[j]*(-phi[j]+ sumphi)*v +
                                            diff_FAD[0]*diff_FAD[0]*(dphidx[j]*dvdx + dphidy[j]*dvdy + dphidz[j]*dvdz ));
            }
            res(e,resindex) += phi_dot[j]*v;
          }
        }
      }
    }
    
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void boundaryResidual() {
    
    //TMW: NOT BEEN UPDATED TO NXTGEN
    
    // NOTES:
    // 1. basis and basis_grad already include the integration weights
    
    //    int numCubPoints = ip.dimension(1);
    // int phi_basis = usebasis[phi_num];
    // int numBasis = basis[phi_basis].dimension(1);
    
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
  
  void edgeResidual() {
    
  }
  
  // ========================================================================================
  // The boundary/edge flux
  // ========================================================================================
  
  void computeFlux() {
    
    ScalarT x = 0.0;
    ScalarT y = 0.0;
    ScalarT z = 0.0;
    
    for (size_t e=0; e<numElem; e++) {
      for (size_t i=0; i<wkset->ip_side.dimension(1); i++) {
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
  
  AD getDirichletValue(const string & var, const ScalarT & x, const ScalarT & y, const ScalarT & z,
                       const ScalarT & t, const string & gside, const bool & useadjoint) const {
    AD val = 0.0;
    return val;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  ScalarT getInitialValue(const string & var, const ScalarT & x, const ScalarT & y, const ScalarT & z,
                         const bool & useadjoint) const {
    ScalarT val = 0.0;
    //    bool test = false;
    //bool test1 = false;
    //bool random = false;
    
    if(initialType == "test") {
      //      the settings of these radii require that the
      //      delta x and y values are set to 0 to 100.
      ScalarT r0 = 0.0;
      ScalarT r1 = 12.5;
      ScalarT r2 = 0.0;
      ScalarT r3 = 12.5;
      ScalarT r4 = 0.0;
      ScalarT r5 = 12.5;
      
      r0 = pow((x-37.5)*(x-37.5) + (y-50.0)*(y-50.0),0.5);
      r2 = pow((x-61.5)*(x-61.5) + (y-50.0)*(y-50.0),0.5);
      r4 = pow((x-50.0)*(x-50.0) + (y-75.0)*(y-75.0),0.5);
      
      //test three disks and three order parameters
      if(r0<r1) {
        if(var == "phi1")
          val = 1.0;
      } else if (r2<r3) {
        if(var == "phi2")
          val = 1.0;
      } else if (r4<r5) {
        if(var == "phi3")
          val = 1.0;
      } else {
        val = 0.0;
      }
    } else if (initialType == "systematic") {
      //bvbw this is not efficient and may want to consider just passing a pre-defined phi assigment
      // vector<ScalarT> posx;
      // vector<ScalarT> posy;
      ScalarT ri;
      ScalarT epsilon = 0.1;   // epsilon is an arbritrary fraction to make sure the radii are small enough
      ScalarT rad = xmax/(numdisks*2) - epsilon* xmax/numdisks;
      int count=0;
      for(int i=0;i<numdisks;i++) {  //assumes diks==numphases
        // posx.push_back(disk[count]);
        // posy.push_back(disk[count+1]);
        ri = pow((x-disk[count])*(x-disk[count]) + (y-disk[count+1])*(y-disk[count+1]),0.5);
        std::string name = "phi";
        int tmpi = i+1;
        string vartemp = name + std::to_string(tmpi);
        
        if ( ri < rad && var == vartemp ) {
          return val = 1.0;
          //	  cout << "numdisk = " << i << " ri = " << ri << " rad = " << rad << " var = " << var << endl;
        } else {
          val = 0.0;
        }
        count += 2;
      }
      
      //      the settings of these radii require that the
      //      delta x and y values are set to 0 to 100.
      // ScalarT r0 = 0.0;
      // ScalarT r1 = 12.5;
      // ScalarT r2 = 0.0;
      // ScalarT r3 = 12.5;
      // ScalarT r4 = 0.0;
      // ScalarT r5 = 12.5;
      
      // r0 = pow((x-37.5)*(x-37.5) + (y-50.0)*(y-50.0),0.5);
      // r2 = pow((x-61.5)*(x-61.5) + (y-50.0)*(y-50.0),0.5);
      // r4 = pow((x-50.0)*(x-50.0) + (y-75.0)*(y-75.0),0.5);
      
      // //test three disks and three order parameters
      // if(r0<r1) {
      //   if(var == "phi1")
      //     val = 1.0;
      // } else if (r2<r3) {
      //   if(var == "phi2")
      //     val = 1.0;
      // } else if (r4<r5) {
      //   if(var == "phi3")
      //     val = 1.0;
      // } else {
      //   val = 0.0;
      // }
    } else if (initialType == "singledisk") {
      ScalarT r0 = 0.0;
      ScalarT r1 = 30.0;
      
      r0 = pow((x-50.0)*(x-50.0) + (y-50.0)*(y-50.0),0.5);
      if(r0>r1) {
        val = -1.0;
      } else {
        val =1.0;
      }
    } else if (initialType == "twodisk") {  //two disks, two phis
      
      ScalarT r0 = 0.0;
      ScalarT r1 = 12.5;
      ScalarT r2 = 0.0;
      ScalarT r3 = 12.5;
      
      r0 = pow((x-30.0)*(x-30.0) + (y-30.0)*(y-30.0),0.5);
      r2 = pow((x-70.0)*(x-70.0) + (y-70.0)*(y-70.0),0.5);
      
      if(r0<r1) {
        if(var == "phi1")
          val = 1.0;
      } else if (r2<r3) {
        if(var == "phi2")
          val = 1.0;
      } else {
        val = 0.0;
      }
    } else if (initialType == "test1") {
      //      Original code for two disks and three order parameters
      ScalarT r0 = 0.0;
      ScalarT r1 = 12.5;
      ScalarT r2 = 0.0;
      ScalarT r3 = 12.5;
      
      r0 = pow((x-37.5)*(x-37.5) + (y-50.0)*(y-50.0),0.5);
      r2 = pow((x-61.5)*(x-61.5) + (y-50.0)*(y-50.0),0.5);
      
      if(r0<r1) {
        if(var == "phi1" || var == "phi3")
          val = 0.0;
        if(var == "phi2")
          val = 1.0;
      } else if (r2<r3) {
        if(var == "phi1" || var == "phi2")
          val = 0.0;
        if(var == "phi3")
          val = 1.0;
      } else {
        if(var == "phi2" || var == "phi2")
          val = 0.0;
        if(var == "phi1")
          val = 1.0;
      }
    } else if (initialType == "test2") {  // two phis, one disk
      
      ScalarT r0 = 0.0;
      ScalarT r1 = 25.0;
      
      r0 = pow((x-50.0)*(x-50.0) + (y-50.0)*(y-50.0),0.5);
      
      if(r0<r1) {
        if(var == "phi1")
          val = 1.0;
      } else {
        val = 0.0;
      }
    } else if (initialType == "random") {
      //   srand(time(NULL)); //initialize random seed
      //      val =  (ScalarT)rand()/(ScalarT)RAND_MAX
      if(uniform) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(1, 2);
        val = dis(gen);
      } else {
        srand(time(NULL));
        ScalarT xpos = rand() % 100;
        ScalarT ypos = rand() % 100;
      }
    } else if (initialType == "default") {
      ScalarT rad;
      val = 0.0;
      if (spaceDim==2) {
        for (int i=0; i < numdisks ; i++) {
          //          cout << "proc # " << Comm->MyPID()<<  " x " << x << " y " << y << endl;
          rad = pow((x-disk[2*i])*(x-disk[2*i]) + (y-disk[2*i+1])*(y-disk[2*i+1]),0.5);
          //bvbw debug
          //        cout << "proc # " << Comm->MyPID()<<  " x " << x << " y "
          //             << y << " rad " << rad << endl;
          if (rad < disksize) {
            for (int j=0; j< numphases; j++) {
              std::string name = "phi";
              int tmpj = j+1;
              string vartemp = name + std::to_string(tmpj);
              
              if (var == vartemp && i==j ){
                val = 1.0;
              }
            }
          }
        }
      }
      if (spaceDim==3) {
        for (int i=0; i < numdisks ; i++) {
          //          cout << "disknumber " << i << "proc # " << Comm->MyPID()<<  " x " << x << " y " << y << " z " << z << endl;
          rad = pow((x-disk[3*i])*(x-disk[3*i]) + (y-disk[3*i+1])*(y-disk[3*i+1])
                    + (z-disk[3*i+2])*(z-disk[3*i+2]),0.5);
          //bvbw debug
          //        cout << "proc # " << Comm->MyPID()<<  " x " << x << " y "
          //             << y << " rad " << rad << endl;
          if (rad < disksize) {
            for (int j=0; j< numphases; j++) {
              std::string name = "phi";
              int tmpj = j+1;
              string vartemp = name + std::to_string(tmpj);
              
              if (var == vartemp && i==j ){
                val = 1.0;
              }
            }
          }
        }
      }
    } else {
      std::cout << "no initialization type specified" << std::endl;
    }
    return val;
  }
  
  // ========================================================================================
  // Get the initial value
  // ========================================================================================
  
  Kokkos::View<ScalarT**,AssemblyDevice> getInitial(const DRV & ip, const string & var,
                                                     const ScalarT & time, const bool & isAdjoint) const {
      
    int numip = ip.dimension(1);
    Kokkos::View<ScalarT**,AssemblyDevice> initial("initial",numElem,numip);
    
    string dummy ("dummy");
    ScalarT x = 0.0;
    ScalarT y = 0.0;
    ScalarT z = 0.0;
    
    for (size_t e=0; e<numElem; e++) {
      for (int i=0; i<numip; i++) {
        x = ip(e,i,0);
        if (spaceDim > 1)
          y = ip(e,i,1);
        if (spaceDim > 2)
          z = ip(e,i,2);
        initial(e,i) = getInitialValue(var, x, y, z, isAdjoint);
      }
    }
    // original implementation
    // if(testdisks) {
    // for (int i=0; i < numdisks ; i++) {
    //   for(int j=0; j<numip; j++) {
    // 	x = ip(0,j,0);
    // 	y = ip(0,j,1);
    
    // 	  //	cout << "proc # " << Comm->MyPID()<<  " x " << x << " y " << y << endl;
    // 	rad = pow((x-disk[2*i])*(x-disk[2*i]) + (y-disk[2*i+1])*(y-disk[2*i+1]),0.5);
    // 	//bvbw debug
    // 	//        cout << "proc # " << Comm->MyPID()<<  " x " << x << " y "
    // 	//             << y << " rad " << rad << endl;
    // 	if (rad < disksize) {
    // 	  int tmpj = 0;
    // 	  for (int k=0; k< numphases; k++) {
    // 	    std::string name = "phi";
    // 	    tmpj += 1;
    // 	    string vartemp = name + std::to_string(tmpj);
    
    // 	    if (varlist[k] == vartemp && i==k ){
    // 	      initial(k,j) = 1.0;
    // 	      //bvbw debug
    // 	      std::cout << initial << std::endl;
    // 	    }
    // 	  }
    // 	}
    //   }
    // }
    
    return initial;
  }
  
  // ========================================================================================
  // error calculation
  // ========================================================================================
  
  ScalarT trueSolution(const string & var, const ScalarT & x, const ScalarT & y, const ScalarT & z,
                      const ScalarT & time) const {
    ScalarT e = sin(2*PI*x);
    if (spaceDim > 1)
      e *= sin(2*PI*y);
    if (spaceDim > 2)
      e *= sin(2*PI*z);
    
    return e;
  }
  
  // ========================================================================================
  // response calculation
  // ========================================================================================
  
  Kokkos::View<AD***,AssemblyDevice> response(Kokkos::View<AD****,AssemblyDevice> local_soln,
                                              Kokkos::View<AD****,AssemblyDevice> local_soln_grad,
                                              Kokkos::View<AD***,AssemblyDevice> local_psoln,
                                              Kokkos::View<AD****,AssemblyDevice> local_psoln_grad,
                                              const DRV & ip, const ScalarT & time) {
    int numip = ip.dimension(1);
    Kokkos::View<AD***,AssemblyDevice> resp("response",numElem,1,numip);
    
    for (size_t e=0; e<numElem; e++) {
      for (int j=0; j<numip; j++) {
        resp(e,0,j) = local_soln(e,phi_num[j],j,0);
      }
    }
    
    return resp;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  Kokkos::View<AD***,AssemblyDevice> target(const DRV & ip, const ScalarT & time) {
    int numip = ip.dimension(1);
    Kokkos::View<AD***,AssemblyDevice> targ("target",numElem,1,numip);
    for (size_t e=0; e<numElem; e++) {
      for (int j=0; j<numip; j++) {
        targ(e,0,j) = 1.0;
      }
    }
    return targ;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void setVars(std::vector<string> & varlist_) {
    varlist = varlist_;
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
  
  AD SourceTerm(const ScalarT & x, const ScalarT & y, const ScalarT & z,
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
  
  ScalarT boundarySource(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & t,
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
  
  AD DiffusionCoeff(const ScalarT & x, const ScalarT & y, const ScalarT & z) const {
    AD diff = 0.0;
    diff = diff_FAD[0];
    return diff;
  }
  
  // ========================================================================================
  /* return the source term (to be multiplied by test_function) */
  // ========================================================================================
  
  ScalarT robinAlpha(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & t,
                    const string & side) const {
    return 0.0;
  }
  
  
  std::vector<string> extraFieldNames() const {
    std::vector<string> ef;
    ef.push_back("interface");
    return ef;
  }
  
  vector<Kokkos::View<ScalarT***,AssemblyDevice> > extraFields(const DRV & ip, const ScalarT & time) const {
    std::vector<Kokkos::View<ScalarT***,AssemblyDevice> > ef;
    std::vector<AD>  phi;
    AD dval;
    AD phiInterface;
    int numip = ip.dimension(1);
    
    Kokkos::View<ScalarT***,AssemblyDevice> dvals("dvals",numElem,1,numip);
    for (size_t e=0; e<numElem; e++) {
      for (size_t i=0; i<numip; i++) {
        for(int j=0; j<numphases; j++){
          //	phi.push_back(wkset->local_soln(phi_num[j],i));
          phi.push_back(i);
          phiInterface += phi[i]*phi[i] *(1.0 - phi[i])*(1.0-phi[i]);
        }
        //      dval = DiffusionCoeff(x,y,z);
        dvals(e,0,i) = phiInterface.val();
      }
    }
    ef.push_back(dvals);
    return ef;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  vector<Kokkos::View<ScalarT***,AssemblyDevice> > extraCellFields(const DRV ip, const ScalarT & time) const {
    std::vector<Kokkos::View<ScalarT***,AssemblyDevice> > ef;
    std::vector<AD>  phi;
    AD dval;
    AD phiInterface;
    int numip = ip.dimension(1);
    
    Kokkos::View<ScalarT***,AssemblyDevice> dvals("dvals",numElem,1,numip);
    for (size_t e=0; e<numElem; e++) {
      for (size_t i=0; i<numip; i++) {
        for(int j=0; j<numphases; j++){
          //	phi.push_back(wkset->local_soln(phi_num[j],i));
          phi.push_back(i);
          phiInterface += phi[i]*phi[i] *(1.0 - phi[i])*(1.0-phi[i]);
        }
        //      dval = DiffusionCoeff(x,y,z);
        dvals(e,0,i) = phiInterface.val();
      }
    }
    ef.push_back(dvals);
    return ef;
  }
  
  
  // ========================================================================================
  // TMW: this is deprecated
  // ========================================================================================
  
  void updateParameters(const vector<Teuchos::RCP<vector<AD> > > & params,
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
  
  // ========================================================================================
  // ========================================================================================
  
private:
  
  size_t numip, numip_side, numElem, blocknum;
  
  Teuchos::RCP<FunctionInterface> functionManager;
  
  Teuchos::RCP<LA_MpiComm> Comm;      
  std::vector<AD> diff_FAD, L, A;   
  int spaceDim, numParams, numResponses, numphases, numdisks;
  vector<string> varlist;
  std::vector<int> phi_num;
  ScalarT diff, alpha;
  ScalarT disksize;
  ScalarT xmax, xmin, ymax, ymin;
  bool isTD;
  bool uniform, systematic, variableMobility;
  std::vector<ScalarT> disk;
  string initialType; 
  string analysis_type; //to know when parameter is a sample that needs to be transformed
  bool multiscale;
  
  Kokkos::View<AD****,AssemblyDevice> sol, sol_dot, sol_grad;
  Kokkos::View<AD**,AssemblyDevice> res;
  Kokkos::View<int**,AssemblyDevice> offsets;
  Kokkos::View<int****,AssemblyDevice> sideinfo;
  DRV basis, basis_grad;
  
  Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::msphasefield::volumeResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::msphasefield::volumeResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::msphasefield::boundaryResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::msphasefield::boundaryResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MILO::msphasefield::computeFlux() - function evaluation");
  Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MILO::msphasefield::computeFlux() - evaluation of flux");
  
};

#endif
