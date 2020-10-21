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

#ifndef PHASESOLIDIFICATION_H
#define PHASESOLIDIFICATION_H

#include "physics_base.hpp"
#include <random>
#include <math.h>
#include <time.h>

namespace MrHyDE {
  
  class phasesolidification : public physicsbase {
  public:
    
    phasesolidification() {} ;
    
    ~phasesolidification() {};
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    phasesolidification(Teuchos::RCP<Teuchos::ParameterList> & settings,
                        const Teuchos::RCP<Epetra_MpiComm> & Comm_, const int & numip_,
                        const size_t & numip_side_) :
    Comm(Comm_), numip(numip_), numip_side(numip_side_) {
      
      spaceDim = settings->sublist("Mesh").get<int>("dim",2);
      numphases = settings->sublist("Physics").get<int>("number_phases",1);
      numdisks = settings->sublist("Physics").get<int>("numdisks",3);
      disksize = settings->sublist("Physics").get<ScalarT>("disksize",10.0);
      uniform = settings->sublist("Physics").get<bool>("uniform",true);
      
      for (size_t i=0; i<numphases; i++) {
        e_num.push_back(i);
      }
      
      PI = 3.141592653589793238463;
      
      if (settings->sublist("Solver").get<int>("solver",0) == 1)
        isTD = true;
      else
        isTD = false;
      
      // generation of disks for initial condition
      
      ScalarT tolerance = 2*disksize+5.0;  // 2 times the disk radius
      ScalarT xpos;
      ScalarT ypos;
      ScalarT zpos;
      
      // generate random position
      /*
       if(uniform) {
       //	std::random_device rd;
       //	std::mt19937 gen(rd());
       std::mt19937 gen(time(0));
       //std::default_random_engine gen;
       std::uniform_real_distribution<ScalarT> dis(9, 89);
       xpos = dis(gen);
       ypos = dis(gen);
       if (spaceDim > 2) {
       zpos = dis(gen);
       cout << "proc  " << Comm->MyPID() << " xpos " << xpos << " ypos " << ypos << " zpos " << zpos << endl;
       } else {
       cout << "proc  " << Comm->MyPID() << " xpos " << xpos << " ypos " << ypos << endl;
       }
       } else {
       //	srand(1);
       srand(time(NULL));
       xpos = rand() % 100;
       ypos = rand() % 100;
       if (spaceDim > 2)
       zpos = rand() % 100;
       // pushdback to a vector, maybe single vector with x1,y1,x2, y2, etc
       }
       disk.push_back(xpos);
       disk.push_back(ypos);
       if (spaceDim > 2)
       disk.push_back(zpos);
       int count = 0;
       int cntrandom = 0;
       while(count < numdisks-1) {
       // generate next random position
       if(uniform) {     //uniform
       std::random_device rd;
       //	  std::mt19937 gen(rd());
       std::mt19937 gen(cntrandom);
       //std::default_random_engine gen;
       std::uniform_real_distribution<> dis(9, 89);
       xpos = dis(gen);
       ypos = dis(gen);
       if (spaceDim > 2)
       zpos = dis(gen);
       cntrandom++;
       } else {
       srand(time(NULL));
       xpos = rand() % 100;
       ypos = rand() % 100;
       if (spaceDim > 2)
       zpos = rand() % 100;
       }
       // compare to all previous entries and if distance is smaller than tolerance, reject
       // and repeat
       bool test = false;
       
       if (spaceDim == 2) {
       for (int j=0; j < disk.size() ; j=j+2) {
       ScalarT tempdist = (xpos-disk[j])*(xpos-disk[j])+(ypos-disk[j+1])*(ypos-disk[j+1]);
       ScalarT distance = pow(tempdist,0.5);
       if(distance > tolerance) {
       test = true;
       break;
       }
       }
       if(test ==true) {
       cout << "proc  " << Comm->MyPID() << " xpos " << xpos << " ypos " << ypos << endl;
       disk.push_back(xpos);
       disk.push_back(ypos);
       count++;
       }
       }
       if (spaceDim == 3) {
       for (int j=0; j < disk.size() ; j=j+3) {
       ScalarT tempdist = (xpos-disk[j])*(xpos-disk[j])+(ypos-disk[j+1])*(ypos-disk[j+1])
       + (zpos-disk[j+2])*(zpos-disk[j+2]);
       ScalarT distance = pow(tempdist,0.5);
       if(distance > tolerance) {
       test = true;
       break;
       }
       }
       if(test ==true) {
       cout << "proc  " << Comm->MyPID() << " xpos " << xpos << " ypos " << ypos
       << " zpos " << zpos << endl;
       disk.push_back(xpos);
       disk.push_back(ypos);
       disk.push_back(zpos);
       count++;
       }
       }
       }
       */
    }
    
    // ========================================================================================
    // ========================================================================================
    
    void volumeResidual() {
      
      // NOTES:
      // 1. basis and basis_grad already include the integration weights
      
      int numCubPoints = wkset->ip.extent(1);
      int e_basis = wkset->usebasis[e_num[0]];
      int numBasis = wkset->basis[e_basis].extent(1);
      
      //    FCAD local_resid(numphases, numBasis);
      
      //ScalarT diff_FAD = diff;
      ScalarT x = 0.0;
      ScalarT y = 0.0;
      ScalarT z = 0.0;
      ScalarT current_time = wkset->time;
      
      ScalarT v, dvdx, dvdy, dvdz;
      
      for( int i=0; i<numBasis; i++ ) {
        for( int nPt=0; nPt<numCubPoints; nPt++ ) {
          
          std::vector<AD>  phi;
          std::vector<AD>  dphidx;
          std::vector<AD>  dphidy;
          std::vector<AD>  dphidz;
          std::vector<AD>  phi_dot;
          AD  sumphi;
          
          v = wkset->basis[e_basis](0,i,nPt);
          dvdx = wkset->basis_grad[e_basis](0,i,nPt,0);
          x = wkset->ip(0,nPt,0);
          
          sumphi = 0.0;
          for(int j=0; j<numphases; j++) {
            phi.push_back(wkset->local_soln(e_num[j],nPt,0));
            phi_dot.push_back(wkset->local_soln_dot(e_num[j],nPt,0));
            dphidx.push_back(wkset->local_soln_grad(e_num[j],nPt,0));
            
            if (spaceDim > 1) {
              y = wkset->ip(0,nPt,1);
              dphidy.push_back(wkset->local_soln_grad(e_num[j],nPt,1));
              
              dvdy = wkset->basis_grad[e_basis](0,i,nPt,1);
            }
            if (spaceDim > 2) {
              z = wkset->ip(0,nPt,2);
              dphidz.push_back(wkset->local_soln_grad(e_num[j],nPt,2));
              dvdz = wkset->basis_grad[e_basis](0,i,nPt,2);
            }
            sumphi = sumphi + phi[j]*phi[j];
          }
          
          
          for(int j=0; j<numphases; j++) {
            
            wkset->res(j,i) += L[0]*(16.0*A[0]*phi[j]*(-phi[j]+ sumphi)*v +
                                     diff_FAD[0]*diff_FAD[0]*(dphidx[j]*dvdx +
                                                              dphidy[j]*dvdy ));
            if (spaceDim >2 )
              wkset->res(j,i) += L[0]*(diff_FAD[0]*diff_FAD[0]*(dphidz[j]*dvdz +
                                                                dphidz[j]*dvdz ));
            if (wkset->isTransient)
              wkset->res(j,i) += phi_dot[j]*v;
          }
        }
      }
    }
    
    // ========================================================================================
    // ========================================================================================
    
    void boundaryResidual() {
      
      // NOTES:
      // 1. basis and basis_grad already include the integration weights
      
      // int numBasis = basis.extent(2);
      // int numSideCubPoints = ip.extent(1);
      
      // FCAD local_resid(numphases, numBasis);
      
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
      //       for( int nPt=0; nPt<numSideCubPoints; nPt++ ) {
      //          v = basis(ee,i,nPt);
      //          x = ip(ee,nPt,0);
      //          phi = local_soln(ee,e_num,nPt);
      //          dphidx = local_solngrad(ee,e_num,nPt,0);
      //          dvdx = basis_grad(ee,i,nPt,0);
      
      //          if (spaceDim > 1) {
      //             y = ip(ee,nPt,1);
      //             dphidy = local_solngrad(ee,e_num,nPt,1);
      //          }
      //          if (spaceDim > 2) {
      //             z = ip(ee,nPt,2);
      //             dphidz = local_solngrad(ee,e_num,nPt,2);
      //          }
      //          source = this->boundarySource(x, y, z, current_time, side_name);
      //          robin_alpha = this->robinAlpha(x, y, z, current_time, side_name);
      
      //          local_resid(ee,i) += -diff_FAD[0]*dphidx*normals(ee,nPt,0)*v;
      //          if (spaceDim > 1) {
      //             local_resid(ee,i) += -diff_FAD[0]*dphidy*normals(ee,nPt,1)*v;
      //          }
      //          if (spaceDim > 2) {
      //             local_resid(ee,i) += -diff_FAD[0]*dphidz*normals(ee,nPt,2)*v;
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
      ScalarT current_time = wkset->time;
      
      for (size_t i=0; i<wkset->ip_side.extent(1); i++) {
        x = wkset->ip_side(0,i,0);
        if (spaceDim > 1)
          y = wkset->ip_side(0,i,1);
        if (spaceDim > 2)
          z = wkset->ip_side(0,i,2);
        
        
        AD diff = DiffusionCoeff(x,y,z);
        AD penalty = 10.0*diff/wkset->h;
        wkset->flux(e_num[0],i) += diff*wkset->local_soln_grad_side(e_num[0],i,0)*wkset->normals(0,i,0) + penalty*(wkset->local_aux_side(e_num[0],i)-wkset->local_soln_side(e_num[0],i,0));
        if (spaceDim > 1)
          wkset->flux(e_num[0],i) += diff*wkset->local_soln_grad_side(e_num[0],i,1)*wkset->normals(0,i,1);
        if (spaceDim > 2)
          wkset->flux(e_num[0],i) += diff*wkset->local_soln_grad_side(e_num[0],i,2)*wkset->normals(0,i,2);
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
      bool test = false;
      bool test1 = false;
      bool random = false;
      
      if(test) {
        //      the settings of these radii require that the delta x and y values are set to 0 to 100.
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
      } else if (test1) {
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
      } else if (random) {
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
      } else {
        ScalarT rad;
        val = 0.0;
        if (spaceDim==2) {
          for (int i=0; i < numdisks ; i++) {
            cout << "proc # " << Comm->MyPID()<<  " x " << x << " y " << y << endl;
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
            cout << "disknumber " << i << "proc # " << Comm->MyPID()<<  " x " << x << " y " << y << " z " << z << endl;
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
      }
      return val;
    }
    
    // ========================================================================================
    // Get the initial value
    // ========================================================================================
    
    FC getInitial(const DRV & ip, const string & var, const ScalarT & time, const bool & isAdjoint) const {
      int numip = ip.extent(1);
      FC initial(1,numip);
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
    
    FCAD response(const FCAD & local_soln,
                  const FCAD & local_soln_grad,
                  const FCAD & local_psoln,
                  const FCAD & local_psoln_grad,
                  const DRV & ip, const ScalarT & time) const {
      int numip = ip.extent(1);
      FCAD resp(1,numip);
      for (int j=0; j<numip; j++) {
        resp(0,j) = local_soln(e_num[j],j,0);
      }
      
      return resp;
    }
    
    // ========================================================================================
    // ========================================================================================
    
    //  FCAD target(const FC & ip, const ScalarT & time) const {
    FCAD target(const FC & ip, const ScalarT & time) {
      int numCC = ip.extent(0);
      int numip = ip.extent(1);
      FCAD targ(numCC,1,numip);
      for (int i=0; i<numCC; i++) {
        for (int j=0; j<numip; j++) {
          targ(i,0,j) = 1.0;
        }
      }
      
      return targ;
    }
    
    // ========================================================================================
    // ========================================================================================
    
    void setVars(std::vector<string> & varlist_) {
      // varlist = varlist_;
      // for (size_t i=0; i<varlist.size(); i++) {
      //   if (varlist[i] == "phi1")
      //     e1_num = i;
      //   if (varlist[i] == "phi2")
      //     e2_num = i;
      //   if (varlist[i] == "phi3")
      //     e3_num = i;
      
      varlist = varlist_;
      //    for (size_t i=0; i<varlist.size(); i++) {
      for (size_t i=0; i<numphases; i++) {
        e_num.push_back(i);
      }
    }
    
    // ========================================================================================
    // ========================================================================================
    
    void setUserDefined(Teuchos::RCP<UserDefined> & udfunc_) {
      udfunc = udfunc_;
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
    
    // ========================================================================================
    // ========================================================================================
    
    int getNumResponses() {
      return numResponses;
    }
    
    // ========================================================================================
    // ========================================================================================
    
    std::vector<string> extraFieldNames() const {
      std::vector<string> ef;
      return ef;
    }
    
    // ========================================================================================
    // ========================================================================================
    
    vector<string> ResponseFieldNames() const {
      std::vector<string> rf;
      return rf;
    }
    
    // ========================================================================================
    // ========================================================================================
    vector<string> extraCellFieldNames() const {
      std::vector<string> ef;
      ef.push_back("grain");
      return ef;
    }
    
    // ========================================================================================
    // ========================================================================================
    
    std::vector<FC > extraFields() const {
      std::vector<FC > ef;
      return ef;
    }
    
    vector<FC> extraFields(const FC & ip, const ScalarT & time) {
      std::vector<FC > ef;
      return ef;
    }
    
    // ========================================================================================
    // ========================================================================================
    
    vector<FC> extraCellFields(const FC & ip, const ScalarT & time) const {
      std::vector<FC > ef;
      return ef;
    }
    
    // ========================================================================================
    // ========================================================================================
    
    void setExtraFields(const size_t & numElem_) {
      numElem = numElem_;
    }
    
    // ========================================================================================
    // ========================================================================================
    
    void updateParameters(const std::vector<std::vector<AD> > & params,
                          const std::vector<string> & paramnames) {
      
      for (size_t p=0; p<paramnames.size(); p++) {
        if (paramnames[p] == "thermal_diff")
          diff_FAD = params[p];
        else if (paramnames[p] == "L")
          L = params[p];
        else if (paramnames[p] == "A")
          A = params[p];
        
        //else
        //cout << "Parameter not used: " << paramnames[p] << endl;
      }
    }
    
    // ========================================================================================
    // ========================================================================================
    
    
    
    FCAD scalarRespFunc(const FCAD & integralResponses,
                        const bool & justDeriv) const {return integralResponses;}
    bool useScalarRespFunc() const {return false;}
    
  private:
    
    Teuchos::RCP<UserDefined> udfunc;
    size_t numip, numip_side;
    
    
    Teuchos::RCP<Epetra_MpiComm> Comm;
    std::vector<AD> diff_FAD, L, A;
    int spaceDim, numElem, numParams, numResponses, numphases, numdisks;
    vector<string> varlist;
    std::vector<int> e_num;
    ScalarT PI;
    ScalarT diff, alpha;
    ScalarT disksize;
    bool isTD;
    bool uniform;
    std::vector<ScalarT> disk;
    string analysis_type; //to know when parameter is a sample that needs to be transformed
    bool multiscale;
  };
  
}

#endif
