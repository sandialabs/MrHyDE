/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef LINEARELAST_H
#define LINEARELAST_H

#include "physics_base.hpp"
#include "CrystalElasticity.hpp"
#include <string>

static void linearelasticityHelp() {
  cout << "********** Help and Documentation for the Linear Elasticity Physics Module **********" << endl << endl;
  cout << "Model:" << endl << endl;
  cout << "User defined functions: " << endl << endl;
}

class linearelasticity : public physicsbase {
public:
  
  linearelasticity() {} ;
  
  ~linearelasticity() {} ;
  
  // ========================================================================================
  /* Constructor to set up the problem */
  // ========================================================================================
  
  linearelasticity(Teuchos::RCP<Teuchos::ParameterList> & settings, const int & numip_,
                   const size_t & numip_side_, const int & numElem_,
                   Teuchos::RCP<FunctionInterface> & functionManager_, const size_t & blocknum_) :
  numip(numip_), numip_side(numip_side_), numElem(numElem_),
  blocknum(blocknum_) {
    
    label = "linearelasticity";
    
    spaceDim = settings->sublist("Mesh").get<int>("dim",2);
    functionManager = functionManager_;
    
    if (spaceDim == 1) {
      myvars = {"dx"};
      mybasistypes = {"HGRAD"};
    }
    else if (spaceDim == 2) {
      myvars = {"dx","dy"};
      mybasistypes = {"HGRAD","HGRAD"};
    }
    else if (spaceDim == 3) {
      myvars = {"dx","dy","dz"};
      mybasistypes = {"HGRAD","HGRAD","HGRAD"};
    }
    
    incplanestress = settings->sublist("Physics").get<bool>("incplanestress",false);
    
    formparam = settings->sublist("Physics").get<ScalarT>("form_param",1.0);
    epen = settings->sublist("Physics").get<ScalarT>("penalty",10.0);
    cell_num = 0;
    
    multiscale = settings->isSublist("Subgrid");
    useLame = settings->sublist("Physics").get<bool>("Use Lame Parameters",true);
    // all these need to be updated to the parameter format
    
    addBiot = settings->sublist("Physics").get<bool>("Biot",false);
    
    // TMW: we might move biot_alpha to udfunc
    // TS: e_ref and alpha_T too?
    biot_alpha = settings->sublist("Physics").get<ScalarT>("Biot alpha",0.0);
    e_ref = settings->sublist("Physics").get<ScalarT>("T_ambient",0.0);
    alpha_T = settings->sublist("Physics").get<ScalarT>("alpha_T",1.0e-6);
    
    x = 0.0;    y = 0.0;    z = 0.0;
    dx = 0.0;    ddx_dx = 0.0;    ddx_dy = 0.0;    ddx_dz = 0.0;
    dy = 0.0;    ddy_dx = 0.0;    ddy_dy = 0.0;    ddy_dz = 0.0;
    dz = 0.0;    ddz_dx = 0.0;    ddz_dy = 0.0;    ddz_dz = 0.0;
    
    dpdx = 0.0;    dpdy = 0.0;    dpdz = 0.0;    eval = 0.0;    delta_e = 0.0;
    plambdax = 0.0;    plambday = 0.0;    plambdaz = 0.0;
    
    crystalelast = Teuchos::rcp(new CrystalElastic(settings, numElem));
    useCE = settings->sublist("Physics").get<bool>("Use Crystal Elasticity",false);
    
    Teuchos::ParameterList fs = settings->sublist("Functions");
    
    functionManager->addFunction("lambda",fs.get<string>("lambda","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("mu",fs.get<string>("mu","0.5"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("source dx",fs.get<string>("source dx","0.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("source dy",fs.get<string>("source dy","0.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("source dz",fs.get<string>("source dz","0.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("lambda",fs.get<string>("lambda","1.0"),numElem,numip_side,"side ip",blocknum);
    functionManager->addFunction("mu",fs.get<string>("mu","0.5"),numElem,numip_side,"side ip",blocknum);
    //functionManager->addFunction("Neumann source dx",fs.get<string>("Neumann source dx","0.0"),numElem,numip_side,"side ip",blocknum);
    //functionManager->addFunction("Neumann source dy",fs.get<string>("Neumann source dy","0.0"),numElem,numip_side,"side ip",blocknum);
    //functionManager->addFunction("Neumann source dz",fs.get<string>("Neumann source dz","0.0"),numElem,numip_side,"side ip",blocknum);
    
  }
  // ========================================================================================
  // ========================================================================================
  
  void volumeResidual() {
    
    eval = 0.0;
    delta_e = 0.0;
    
    time = wkset->time;
    
    {
      Teuchos::TimeMonitor funceval(*volumeResidualFunc);
      source_dx = functionManager->evaluate("source dx","ip",blocknum);
      if (spaceDim > 1) {
        source_dy = functionManager->evaluate("source dy","ip",blocknum);
      }
      if (spaceDim > 2) {
        source_dz = functionManager->evaluate("source dz","ip",blocknum);
      }
      lambda = functionManager->evaluate("lambda","ip",blocknum);
      mu = functionManager->evaluate("mu","ip",blocknum);
    }
    
    this->computeStress(false);
    
    Teuchos::TimeMonitor localtime(*volumeResidualFill);
    
    if (spaceDim == 1) {
      dx_basis = wkset->usebasis[dx_num];
      basis = wkset->basis[dx_basis];
      basis_grad = wkset->basis_grad[dx_basis];
      
      parallel_for(RangePolicy<AssemblyDevice>(0,res.dimension(0)), KOKKOS_LAMBDA (const int e ) {
        for (size_t k=0; k<basis.dimension(2); k++ ) {
          this->setLocalSoln(e,k,false);
          for (int i=0; i<basis.dimension(1); i++ ) {
            v = basis(e,i,k);
            dvdx = basis_grad(e,i,k,0);
            resindex = offsets(dx_num,i);
            res(e,resindex) += stress(e,k,0,0)*dvdx - source_dx(e,k)*v;
          }
        }
      });
    }
    else if (spaceDim == 2) {
      
      // first equation
      dx_basis = wkset->usebasis[dx_num];
      basis = wkset->basis[dx_basis];
      basis_grad = wkset->basis_grad[dx_basis];
      
      parallel_for(RangePolicy<AssemblyDevice>(0,res.dimension(0)), KOKKOS_LAMBDA (const int e ) {
        for (size_t k=0; k<basis.dimension(2); k++ ) {
          //this->setLocalSoln(e,k,false);
          for (int i=0; i<basis.dimension(1); i++ ) {
            v = basis(e,i,k);
            dvdx = basis_grad(e,i,k,0);
            dvdy = basis_grad(e,i,k,1);
            resindex = offsets(dx_num,i);
            
            res(e,resindex) += stress(e,k,0,0)*dvdx + stress(e,k,0,1)*dvdy - source_dx(e,k)*v;
          
          }
        }
      });
      
      
      // second equation
      dy_basis = wkset->usebasis[dy_num];
      basis = wkset->basis[dy_basis];
      basis_grad = wkset->basis_grad[dy_basis];
      
      parallel_for(RangePolicy<AssemblyDevice>(0,res.dimension(0)), KOKKOS_LAMBDA (const int e ) {
        for (size_t k=0; k<basis.dimension(2); k++ ) {
          //this->setLocalSoln(e,k,false);
          
          for (int i=0; i<basis.dimension(1); i++ ) {
            v = basis(e,i,k);
            dvdx = basis_grad(e,i,k,0);
            dvdy = basis_grad(e,i,k,1);
            resindex = offsets(dy_num,i);
            
            res(e,resindex) += stress(e,k,1,0)*dvdx + stress(e,k,1,1)*dvdy - source_dy(e,k)*v;
            
          }
        }
      });
    }
    else if (spaceDim == 3) {
      
      // first equation
      dx_basis = wkset->usebasis[dx_num];
      basis = wkset->basis[dx_basis];
      basis_grad = wkset->basis_grad[dx_basis];
      
      parallel_for(RangePolicy<AssemblyDevice>(0,res.dimension(0)), KOKKOS_LAMBDA (const int e ) {
        for(size_t k=0; k<basis.dimension(2); k++ ) {
          //this->setLocalSoln(e,k,false);
          for( int i=0; i<basis.dimension(1); i++ ) {
            v = basis(e,i,k);
            dvdx = basis_grad(e,i,k,0);
            dvdy = basis_grad(e,i,k,1);
            dvdz = basis_grad(e,i,k,2);
            resindex = offsets(dx_num,i);
            res(e,resindex) += stress(e,k,0,0)*dvdx + stress(e,k,0,1)*dvdy + stress(e,k,0,2)*dvdz - source_dx(e,k)*v;
          }
        }
      });
      
      // second equation
      dy_basis = wkset->usebasis[dy_num];
      basis = wkset->basis[dy_basis];
      basis_grad = wkset->basis_grad[dy_basis];
      
      parallel_for(RangePolicy<AssemblyDevice>(0,res.dimension(0)), KOKKOS_LAMBDA (const int e ) {
        for(size_t k=0; k<basis.dimension(2); k++ ) {
          //this->setLocalSoln(e,k,false);
          for( int i=0; i<basis.dimension(1); i++ ) {
            v = basis(e,i,k);
            dvdx = basis_grad(e,i,k,0);
            dvdy = basis_grad(e,i,k,1);
            dvdz = basis_grad(e,i,k,2);
            resindex = offsets(dy_num,i);
            res(e,resindex) += stress(e,k,1,0)*dvdx + stress(e,k,1,1)*dvdy + stress(e,k,1,2)*dvdz - source_dy(e,k)*v;
          }
        }
      });
      
      // third equation
      dz_basis = wkset->usebasis[dz_num];
      basis = wkset->basis[dz_basis];
      basis_grad = wkset->basis_grad[dz_basis];
      
      parallel_for(RangePolicy<AssemblyDevice>(0,res.dimension(0)), KOKKOS_LAMBDA (const int e ) {
        for(size_t k=0; k<basis.dimension(2); k++ ) {
          //this->setLocalSoln(e,k,false);
          for( int i=0; i<basis.dimension(1); i++ ) {
            v = basis(e,i,k);
            dvdx = basis_grad(e,i,k,0);
            dvdy = basis_grad(e,i,k,1);
            dvdz = basis_grad(e,i,k,2);
            resindex = offsets(dz_num,i);
            res(e,resindex) += stress(e,k,2,0)*dvdx + stress(e,k,2,1)*dvdy + stress(e,k,2,2)*dvdz - source_dz(e,k)*v;
          }
        }
      });
    }
    
    //KokkosTools::print(wkset->res);
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void boundaryResidual() {
    // NOTES:
    // 1. basis and basis_grad already include the integration weights
    
    //sideinfo = wkset->sideinfo;
    Kokkos::View<int**,AssemblyDevice> bcs = wkset->var_bcs;
    
    //AD lambda, mu;
    AD basisVec;
    AD penalty;
    eval = 0.0;
    delta_e = 0.0;
    AD trac = 0.0; // dummy argument unless using discretized traction parameter
    time = wkset->time;
    
    int cside = wkset->currentside;
    string sname = wkset->sidename;
    
    ScalarT sf = formparam;
    if (wkset->isAdjoint) {
      sf = 1.0;
    }
    
    int dy_sidetype = 0;
    int dz_sidetype = 0;
    int dx_sidetype = bcs(dx_num,cside);
    if (spaceDim > 1) {
      dy_sidetype = bcs(dy_num,cside);
    }
    if (spaceDim > 2) {
      dz_sidetype = bcs(dz_num,cside);
    }
    
    if (dx_sidetype > 1 || dy_sidetype > 1 || dz_sidetype > 1) {
      
      {
        Teuchos::TimeMonitor localtime(*boundaryResidualFunc);
        if (dx_sidetype == 2) {
          sourceN_dx = functionManager->evaluate("Neumann dx " + sname,"side ip",blocknum);
        }
        if (dy_sidetype == 2) {
          sourceN_dy = functionManager->evaluate("Neumann dy " + sname,"side ip",blocknum);
        }
        if (dz_sidetype == 2) {
          sourceN_dz = functionManager->evaluate("Neumann dz " + sname,"side ip",blocknum);
        }
        
        lambda_side = functionManager->evaluate("lambda","side ip",blocknum);
        mu_side = functionManager->evaluate("mu","side ip",blocknum);
        
      }
      
      // Since normals get recomputed often, this needs to be reset
      normals = wkset->normals;
      Teuchos::TimeMonitor localtime(*boundaryResidualFill);
      
      this->computeStress(true);
      
      if (spaceDim == 1) {
        dx_basis = wkset->usebasis[dx_num];
        basis = wkset->basis_side[dx_basis];
        basis_grad = wkset->basis_grad_side[dx_basis];
        
        for (int e=0; e<basis.dimension(0); e++) {
          if (dx_sidetype == 2) { // Neumann
            for (size_t k=0; k<basis.dimension(2); k++ ) {
              for (int i=0; i<basis.dimension(1); i++ ) {
                v = basis(e,i,k);
                resindex = offsets(dx_num,i);
                res(e,resindex) += -sourceN_dx(e,k)*v;
              }
            }
          }
          else if (dx_sidetype == 4 || dx_sidetype == 5) {
            for (int k=0; k<basis.dimension(2); k++ ) {
              this->setLocalSoln(e,k,true);
              penalty = epen*(lambda_side(e,k) + 2.0*mu_side(e,k))/wkset->h(e);
              plambdax = 0.0;
              if (dx_sidetype == 5) {
                plambdax = aux_side(e,dx_num,k);
              }
              
              for (int i=0; i<basis.dimension(1); i++ ) {
                v = basis(e,i,k);
                basisVec = computeBasisVec(dx-plambdax, dy-plambday, dz-plambdaz,
                                           mu_side(e,k), lambda_side(e,k), normals,
                                           basis_grad, dx_basis, e, i, k, 0);
                resindex = offsets(dx_num,i);
                res(e,resindex) += (-stress(e,k,0,0)*normals(e,k,0))*v + penalty*(dx-plambdax)*v - sf*basisVec;
              }
            }
          }
        }
      }
      else if (spaceDim == 2) {
        
        // dx equation boundary residual
        dx_basis = wkset->usebasis[dx_num];
        basis = wkset->basis_side[dx_basis];
        basis_grad = wkset->basis_grad_side[dx_basis];
        
        for (int e=0; e<basis.dimension(0); e++) {
          if (dx_sidetype == 2) {
            for (size_t k=0; k<basis.dimension(2); k++ ) {
              for (int i=0; i<basis.dimension(1); i++ ) {
                v = basis(e,i,k);
                resindex = offsets(dx_num,i);
                res(e,resindex) += -sourceN_dx(e,k)*v;
              }
            }
          }
          else if (dx_sidetype == 4 || dx_sidetype == 5) {
            for (int k=0; k<basis.dimension(2); k++ ) {
              this->setLocalSoln(e,k,true);
              penalty = epen*(lambda_side(e,k) + 2.0*mu_side(e,k))/wkset->h(e);
              plambdax = 0.0;
              plambday = 0.0;
              if (dx_sidetype == 5) {
                plambdax = aux_side(e,dx_num,k);
                plambday = aux_side(e,dy_num,k);
              }
              for (int i=0; i<basis.dimension(1); i++ ) {
                v = basis(e,i,k);
                basisVec = computeBasisVec(dx-plambdax, dy-plambday, dz-plambdaz,
                                           mu_side(e,k), lambda_side(e,k), normals, basis_grad,
                                           dx_basis, e, i, k, 0);
                resindex = offsets(dx_num,i);
                res(e,resindex) += (-stress(e,k,0,0)*normals(e,k,0) - stress(e,k,0,1)*normals(e,k,1))*v -
                sf*basisVec + penalty*(dx-plambdax)*v;
                
              }
              
            }
          }
        }
        
        // dy equation boundary residual
        dy_basis = wkset->usebasis[dy_num];
        basis = wkset->basis_side[dy_basis];
        basis_grad = wkset->basis_grad_side[dy_basis];
        
        for (int e=0; e<basis.dimension(0); e++) {
          if (dy_sidetype == 2) {
            
            for (size_t k=0; k<basis.dimension(2); k++ ) {
              for (int i=0; i<basis.dimension(1); i++ ) {
                v = basis(e,i,k);
                resindex = offsets(dy_num,i);
                res(e,resindex) += -sourceN_dy(e,k)*v;
              }
            }
          }
          else if (dy_sidetype == 4 || dy_sidetype == 5) {
            for (int k=0; k<basis.dimension(2); k++ ) {
              this->setLocalSoln(e,k,true);
              penalty = epen*(lambda_side(e,k) + 2.0*mu_side(e,k))/wkset->h(e);
              plambdax = 0.0;
              plambday = 0.0;
              if (dy_sidetype == 5) {
                plambdax = aux_side(e,dx_num,k);
                plambday = aux_side(e,dy_num,k);
              }
              
              for (int i=0; i<basis.dimension(1); i++ ) {
                v = basis(e,i,k);
                basisVec = computeBasisVec(dx-plambdax, dy-plambday, dz-plambdaz,
                                           mu_side(e,k), lambda_side(e,k), normals, basis_grad,
                                           dy_basis, e, i, k, 1);
                
                resindex = offsets(dy_num,i);
                res(e,resindex) += (-stress(e,k,1,0)*normals(e,k,0) - stress(e,k,1,1)*normals(e,k,1))*v -
                sf*basisVec + penalty*(dy-plambday)*v;
              }
              
            }
          }
        }
      }
      
      else if (spaceDim == 3) {
        
        // dx equation boundary residual
        dx_basis = wkset->usebasis[dx_num];
        basis = wkset->basis_side[dx_basis];
        basis_grad = wkset->basis_grad_side[dx_basis];
        
        AD deltax, deltay, deltaz;
        
        for (int e=0; e<basis.dimension(0); e++) {
          if (dx_sidetype == 2) {
            for (size_t k=0; k<basis.dimension(2); k++ ) {
              for (int i=0; i<basis.dimension(1); i++ ) {
                v = basis(e,i,k);
                resindex = offsets(dx_num,i);
                res(e,resindex) += -sourceN_dx(e,k)*v;
              }
            }
          }
          else if (dx_sidetype == 4 || dx_sidetype == 5) {
            for (int k=0; k<basis.dimension(2); k++ ) {
              this->setLocalSoln(e,k,true);
              penalty = epen*(lambda_side(e,k) + 2.0*mu_side(e,k))/wkset->h(e);
              plambdax = 0.0;
              plambday = 0.0;
              plambdaz = 0.0;
              if (dx_sidetype == 5) {
                plambdax = aux_side(e,dx_num,k);
                plambday = aux_side(e,dy_num,k);
                plambdaz = aux_side(e,dz_num,k);
              }
              
              for (int i=0; i<basis.dimension(1); i++ ) {
                v = basis(e,i,k);
                deltax = dx-plambdax;
                deltay = dy-plambday;
                deltaz = dz-plambdaz;
                
                basisVec = lambda_side(e,k)*basis_grad(e,i,k,0)*(deltax*normals(e,k,0) + deltay*normals(e,k,1) + deltaz*normals(e,k,2)) +
                2.0*mu_side(e,k)*basis_grad(e,i,k,0)*deltax*normals(e,k,0) +
                mu_side(e,k)*basis_grad(e,i,k,1)*(deltax*normals(e,k,1) + deltay*normals(e,k,0)) +
                mu_side(e,k)*basis_grad(e,i,k,2)*(deltax*normals(e,k,2) + deltaz*normals(e,k,0));
                
                resindex = offsets(dx_num,i);
                res(e,resindex) += (-stress(e,k,0,0)*normals(e,k,0) - stress(e,k,0,1)*normals(e,k,1) - stress(e,k,0,2)*normals(e,k,2))*v -
                sf*basisVec + penalty*(dx-plambdax)*v;
                
              }
            }
          }
          
        }
        
        // dy equation boundary residual
        dy_basis = wkset->usebasis[dy_num];
        basis = wkset->basis_side[dy_basis];
        basis_grad = wkset->basis_grad_side[dy_basis];
        
        for (int e=0; e<basis.dimension(0); e++) {
          if (dy_sidetype == 2) {
            for (size_t k=0; k<basis.dimension(2); k++ ) {
              for (int i=0; i<basis.dimension(1); i++ ) {
                v = basis(e,i,k);
                resindex = offsets(dy_num,i);
                res(e,resindex) += -sourceN_dy(e,k)*v;
              }
            }
          }
          else if (dy_sidetype == 4 || dy_sidetype == 5) {
            for (int k=0; k<basis.dimension(2); k++ ) {
              this->setLocalSoln(e,k,true);
              penalty = epen*(lambda_side(e,k) + 2.0*mu_side(e,k))/wkset->h(e);
              plambdax = 0.0;
              plambday = 0.0;
              plambdaz = 0.0;
              if (dy_sidetype == 5) {
                plambdax = aux_side(e,dx_num,k);
                plambday = aux_side(e,dy_num,k);
                plambdaz = aux_side(e,dz_num,k);
              }
              
              for (int i=0; i<basis.dimension(1); i++ ) {
                v = basis(e,i,k);
                deltax = dx-plambdax;
                deltay = dy-plambday;
                deltaz = dz-plambdaz;
                
                basisVec = lambda_side(e,k)*basis_grad(e,i,k,1)*(deltax*normals(e,k,0) + deltay*normals(e,k,1) + deltaz*normals(e,k,2)) +
                2.0*mu_side(e,k)*basis_grad(e,i,k,1)*deltay*normals(e,k,1) +
                mu_side(e,k)*basis_grad(e,i,k,0)*(deltay*normals(e,k,0) + deltax*normals(e,k,1)) +
                mu_side(e,k)*basis_grad(e,i,k,2)*(deltay*normals(e,k,2) + deltaz*normals(e,k,1));
                
                resindex = offsets(dy_num,i);
                res(e,resindex) += (-stress(e,k,1,0)*normals(e,k,0) - stress(e,k,1,1)*normals(e,k,1) - stress(e,k,1,2)*normals(e,k,2))*v -
                sf*basisVec + penalty*(dy-plambday)*v;
                
              }
            }
          }
          
        }
        
        // dz equation boundary residual
        dz_basis = wkset->usebasis[dz_num];
        basis = wkset->basis_side[dz_basis];
        basis_grad = wkset->basis_grad_side[dz_basis];
        
        for (int e=0; e<basis.dimension(0); e++) {
          if (dz_sidetype == 2) {
            for (size_t k=0; k<basis.dimension(2); k++ ) {
              for (int i=0; i<basis.dimension(1); i++ ) {
                v = basis(e,i,k);
                resindex = offsets(dz_num,i);
                res(e,resindex) += -sourceN_dz(e,k)*v;
              }
            }
          }
          else if (dz_sidetype == 4 || dz_sidetype == 5) {
            for (int k=0; k<basis.dimension(2); k++ ) {
              this->setLocalSoln(e,k,true);
              penalty = epen*(lambda_side(e,k) + 2.0*mu_side(e,k))/wkset->h(e);
              plambdax = 0.0;
              plambday = 0.0;
              plambdaz = 0.0;
              if (dz_sidetype == 5) {
                plambdax = aux_side(e,dx_num,k);
                plambday = aux_side(e,dy_num,k);
                plambdaz = aux_side(e,dz_num,k);
              }
              
              for (int i=0; i<basis.dimension(1); i++ ) {
                v = basis(e,i,k);
                deltax = dx-plambdax;
                deltay = dy-plambday;
                deltaz = dz-plambdaz;
                
                basisVec = lambda_side(e,k)*basis_grad(e,i,k,2)*(deltax*normals(e,k,0) + deltay*normals(e,k,1) + deltaz*normals(e,k,2)) +
                2.0*mu_side(e,k)*basis_grad(e,i,k,2)*deltaz*normals(e,k,2) +
                mu_side(e,k)*basis_grad(e,i,k,0)*(deltaz*normals(e,k,0) + deltax*normals(e,k,2)) +
                mu_side(e,k)*basis_grad(e,i,k,1)*(deltaz*normals(e,k,1) + deltay*normals(e,k,2));
                
                resindex = offsets(dz_num,i);
                res(e,resindex) += (-stress(e,k,2,0)*normals(e,k,0) - stress(e,k,2,1)*normals(e,k,1) - stress(e,k,2,2)*normals(e,k,2))*v -
                sf*basisVec + penalty*(dz-plambdaz)*v;
              }
            }
          }
          
        }
        
      }
    }
    
  }
  
  
  // ========================================================================================
  // The boundary/edge flux
  // ========================================================================================
  
  void computeFlux() {
    
    eval = 0.0;
    delta_e = 0.0;
    AD penalty;
    ScalarT current_time = wkset->time;
    
    ScalarT sf = 1.0;
    if (wkset->isAdjoint) {
      sf = formparam;
    }
    
    {
      Teuchos::TimeMonitor localtime(*fluxFunc);
      
      lambda_side = functionManager->evaluate("lambda","side ip",blocknum);
      mu_side = functionManager->evaluate("mu","side ip",blocknum);
    }
    
    // Since normals get recomputed often, this needs to be reset
    normals = wkset->normals;
    //flux = wkset->flux;
    //aux_side = wkset->local_aux_side;
    //sol_side = wkset->local_soln_side;
    //sol_grad_side = wkset->local_soln_grad_side;
    //offsets = wkset->offsets;
    
    {
      Teuchos::TimeMonitor localtime(*fluxFill);
      
      this->computeStress(true);
      
      if (spaceDim == 1) {
        for (size_t e=0; e<sol_side.dimension(0); e++) {
          for (size_t i=0; i<sol_side.dimension(2); i++) {
            this->setLocalSoln(e,i,true);
            plambdax = aux_side(e,dx_num,i);
            penalty = epen*(lambda_side(e,i) + 2.0*mu_side(e,i))/wkset->h(e);
            flux(e,dx_num,i) += sf*stress(e,i,0,0)*normals(e,i,0) + penalty*(plambdax-dx);
          }
        }
      }
      else if (spaceDim == 2) {
        for (size_t e=0; e<sol_side.dimension(0); e++) {
          for (size_t i=0; i<sol_side.dimension(2); i++) {
            this->setLocalSoln(e,i,true);
            plambdax = aux_side(e,dx_num,i);
            plambday = aux_side(e,dy_num,i);
            penalty = epen*(lambda_side(e,i) + 2.0*mu_side(e,i))/wkset->h(e);
            flux(e,dx_num,i) += sf*(stress(e,i,0,0)*normals(e,i,0) + stress(e,i,0,1)*normals(e,i,1)) + penalty*(plambdax-dx);
            flux(e,dy_num,i) += sf*(stress(e,i,1,0)*normals(e,i,0) + stress(e,i,1,1)*normals(e,i,1)) + penalty*(plambday-dy);
          }
        }
      }
      else if (spaceDim == 3) {
        for (size_t e=0; e<sol_side.dimension(0); e++) {
          for (size_t i=0; i<sol_side.dimension(2); i++) {
            this->setLocalSoln(e,i,true);
            plambdax = aux_side(e,dx_num,i);
            plambday = aux_side(e,dy_num,i);
            plambdaz = aux_side(e,dz_num,i);
            penalty = epen*(lambda_side(e,i) + 2.0*mu_side(e,i))/wkset->h(e);
            
            flux(e,dx_num,i) += sf*(stress(e,i,0,0)*normals(e,i,0) + stress(e,i,0,1)*normals(e,i,1) + stress(e,i,0,2)*normals(e,i,2)) + penalty*(plambdax-dx);
            flux(e,dy_num,i) += sf*(stress(e,i,1,0)*normals(e,i,0) + stress(e,i,1,1)*normals(e,i,1) + stress(e,i,1,2)*normals(e,i,2)) + penalty*(plambday-dy);
            flux(e,dz_num,i) += sf*(stress(e,i,2,0)*normals(e,i,0) + stress(e,i,2,1)*normals(e,i,1) + stress(e,i,2,2)*normals(e,i,2)) + penalty*(plambdaz-dz);
          }
        }
      }
    }
    //KokkosTools::print(stress);
    //KokkosTools::print(flux);
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void setLocalSoln(const size_t & e, const size_t & ipindex, const bool & onside) {
    Teuchos::TimeMonitor localtime(*setLocalSol);
    
    if (onside) {
      if (spaceDim == 1) {
        dx = sol_side(e,dx_num,ipindex,0);
        ddx_dx = sol_grad_side(e,dx_num,ipindex,0);
      }
      else if (spaceDim == 2) {
        dx = sol_side(e,dx_num,ipindex,0);
        dy = sol_side(e,dy_num,ipindex,0);
        ddx_dx = sol_grad_side(e,dx_num,ipindex,0);
        ddx_dy = sol_grad_side(e,dx_num,ipindex,1);
        ddy_dx = sol_grad_side(e,dy_num,ipindex,0);
        ddy_dy = sol_grad_side(e,dy_num,ipindex,1);
      }
      else if (spaceDim == 3) {
        dx = sol_side(e,dx_num,ipindex,0);
        dy = sol_side(e,dy_num,ipindex,0);
        dz = sol_side(e,dz_num,ipindex,0);
        ddx_dx = sol_grad_side(e,dx_num,ipindex,0);
        ddx_dy = sol_grad_side(e,dx_num,ipindex,1);
        ddx_dz = sol_grad_side(e,dx_num,ipindex,2);
        ddy_dx = sol_grad_side(e,dy_num,ipindex,0);
        ddy_dy = sol_grad_side(e,dy_num,ipindex,1);
        ddy_dz = sol_grad_side(e,dy_num,ipindex,2);
        ddz_dx = sol_grad_side(e,dz_num,ipindex,0);
        ddz_dy = sol_grad_side(e,dz_num,ipindex,1);
        ddz_dz = sol_grad_side(e,dz_num,ipindex,2);
      }
      if (e_num >= 0) {
        eval = sol_side(e,e_num,ipindex,0);
        delta_e = eval-e_ref;
      }
    }
    else {
      if (spaceDim == 1) {
        dx = sol(e,dx_num,ipindex,0);
        ddx_dx = sol_grad(e,dx_num,ipindex,0);
      }
      else if (spaceDim == 2) {
        dx = sol(e,dx_num,ipindex,0);
        dy = sol(e,dy_num,ipindex,0);
        ddx_dx = sol_grad(e,dx_num,ipindex,0);
        ddx_dy = sol_grad(e,dx_num,ipindex,1);
        ddy_dx = sol_grad(e,dy_num,ipindex,0);
        ddy_dy = sol_grad(e,dy_num,ipindex,1);
      }
      else if (spaceDim == 3) {
        dx = sol(e,dx_num,ipindex,0);
        dy = sol(e,dy_num,ipindex,0);
        dz = sol(e,dz_num,ipindex,0);
        ddx_dx = sol_grad(e,dx_num,ipindex,0);
        ddx_dy = sol_grad(e,dx_num,ipindex,1);
        ddx_dz = sol_grad(e,dx_num,ipindex,2);
        ddy_dx = sol_grad(e,dy_num,ipindex,0);
        ddy_dy = sol_grad(e,dy_num,ipindex,1);
        ddy_dz = sol_grad(e,dy_num,ipindex,2);
        ddz_dx = sol_grad(e,dz_num,ipindex,0);
        ddz_dy = sol_grad(e,dz_num,ipindex,1);
        ddz_dz = sol_grad(e,dz_num,ipindex,2);
      }
      if (e_num >= 0) {
        eval = sol(e,e_num,ipindex,0);
        delta_e = eval-e_ref;
      }
      if (p_num >= 0) {
        pval = sol(e,p_num,ipindex,0);
      }
      
      
    }
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void setVars(std::vector<string> & varlist_) {
    varlist = varlist_;
    dx_num = -1;
    dy_num = -1;
    dz_num = -1;
    e_num = -1;
    p_num = -1;
    for (size_t i=0; i<varlist.size(); i++) {
      if (varlist[i] == "dx")
        dx_num = i;
      else if (varlist[i] == "dy")
        dy_num = i;
      else if (varlist[i] == "dz")
        dz_num = i;
      else if (varlist[i] == "e")
        e_num = i;
      else if (varlist[i] == "p")
        p_num = i;
      else if (varlist[i] == "Po")
        p_num = i;
      else if (varlist[i] == "Pw")
        p_num = i;
      
    }
  }
  
  
  // ========================================================================================
  // return the stress
  // ========================================================================================
  
  void computeStress(const bool & onside) {
    
    Teuchos::TimeMonitor localtime(*fillStress);
    
    if (useCE) {
      vector<int> indices = {dx_num, dy_num, dz_num, e_num};
      
      stress = crystalelast->computeStress(wkset, indices, onside);
      
    }
    else {
      FDATA mu_vals = mu;
      FDATA lambda_vals = lambda;
      int nip = numip;
      
      if (onside) {
        mu_vals = mu_side;
        lambda_vals = lambda_side;
        nip = numip_side;
      }
      
      stress = Kokkos::View<AD****>("stress",numElem,nip,3,3);
      
      for (int e=0; e<lambda_vals.dimension(0); e++) {
        for (size_t k=0; k<nip; k++) {
          
          this->setLocalSoln(e,k,onside);
          
          AD lambda_val = lambda_vals(e,k);
          if (incplanestress)
            lambda_val = 2.0*mu_vals(e,k);
          else
            lambda_val = lambda_vals(e,k);
          
          AD mu_val = mu_vals(e,k);
          
          stress(e,k,0,0) = (2.0*mu_val+lambda_val)*ddx_dx + lambda_val*(ddy_dy+ddz_dz);
          stress(e,k,0,1) = mu_val*(ddx_dy+ddy_dx);
          stress(e,k,0,2) = mu_val*(ddx_dz+ddz_dx);
          
          stress(e,k,1,0) = mu_val*(ddx_dy+ddy_dx);
          stress(e,k,1,1) = (2.0*mu_val+lambda_val)*ddy_dy + lambda_val*(ddx_dx+ddz_dz);
          stress(e,k,1,2) = mu_val*(ddy_dz+ddz_dy);
          
          stress(e,k,2,0) = mu_val*(ddx_dz+ddz_dx);
          stress(e,k,2,1) = mu_val*(ddy_dz+ddz_dy);
          stress(e,k,2,2) = (2.0*mu_val+lambda_val)*ddz_dz + lambda_val*(ddx_dx+ddy_dy);
          
          
          if (e_num >= 0) { // if we are running thermoelasticity
            //AD alpha_val = alpha_T;
            stress(e,k,0,0) += -alpha_T*delta_e*(3.0*lambda_val + 2.0*mu_val);
            stress(e,k,1,1) += -alpha_T*delta_e*(3.0*lambda_val + 2.0*mu_val);
            stress(e,k,2,2) += -alpha_T*delta_e*(3.0*lambda_val + 2.0*mu_val);
          }
          
          if (addBiot) {
            stress(e,k,0,0) += -biot_alpha*pval;
            stress(e,k,1,1) += -biot_alpha*pval;
            stress(e,k,2,2) += -biot_alpha*pval;
          }
        }
      }
      //vector<int> indices = {dx_num, dy_num, dz_num, e_num};
      
      //FCAD stress2 = crystalelast->computeStress(wkset, indices, onside);
      
    }
    
  }
  
  // ========================================================================================
  /* return the SIPG / IIPG term for a given node and component at an integration point */
  // ========================================================================================
  
  AD computeBasisVec(const AD dx, const AD dy, const AD dz, const AD mu_val, const AD lambda_val,
                     const DRV normals, DRV basis_grad, const int num_basis,
                     const int & elem, const int inode, const int k, const int component) {
    
    Teuchos::TimeMonitor localtime(*computeBasis);
    
    
    AD basisVec;
    //AD lambda_val = this->MaterialProperty("lambda", x, y, z, t);
    //
    
    if (spaceDim == 1) {
      basisVec = (lambda_val + 2.0*mu_val)*basis_grad(elem,inode,k,0)*dx*normals(elem,k,0);
    }
    else if (spaceDim == 2) {
      if (component == 0) {
        basisVec = lambda_val*basis_grad(elem,inode,k,0)*(dx*normals(elem,k,0) + dy*normals(elem,k,1)) +
        2.0*mu_val*basis_grad(elem,inode,k,0)*dx*normals(elem,k,0) +
        mu_val*basis_grad(elem,inode,k,1)*(dx*normals(elem,k,1) + dy*normals(elem,k,0));
        
      }
      else if (component == 1) {
        basisVec = lambda_val*basis_grad(elem,inode,k,1)*(dx*normals(elem,k,0) + dy*normals(elem,k,1)) +
        2.0*mu_val*basis_grad(elem,inode,k,1)*dy*normals(elem,k,1) +
        mu_val*basis_grad(elem,inode,k,0)*(dy*normals(elem,k,0) + dx*normals(elem,k,1));
      }
    }
    else if (spaceDim == 3) {
      if (component == 0) {
        basisVec = lambda_val*basis_grad(elem,inode,k,0)*(dx*normals(elem,k,0) + dy*normals(elem,k,1) + dz*normals(elem,k,2)) +
        2.0*mu_val*basis_grad(elem,inode,k,0)*dx*normals(elem,k,0) +
        mu_val*basis_grad(elem,inode,k,1)*(dx*normals(elem,k,1) + dy*normals(elem,k,0)) +
        mu_val*basis_grad(elem,inode,k,2)*(dx*normals(elem,k,2) + dz*normals(elem,k,0));
        
      }
      else if (component == 1) {
        basisVec = lambda_val*basis_grad(elem,inode,k,1)*(dx*normals(elem,k,0) + dy*normals(elem,k,1) + dz*normals(elem,k,2)) +
        2.0*mu_val*basis_grad(elem,inode,k,1)*dy*normals(elem,k,1) +
        mu_val*basis_grad(elem,inode,k,0)*(dy*normals(elem,k,0) + dx*normals(elem,k,1)) +
        mu_val*basis_grad(elem,inode,k,2)*(dy*normals(elem,k,2) + dz*normals(elem,k,1));
      }
      else if (component == 2) {
        basisVec = lambda_val*basis_grad(elem,inode,k,2)*(dx*normals(elem,k,0) + dy*normals(elem,k,1) + dz*normals(elem,k,2)) +
        2.0*mu_val*basis_grad(elem,inode,k,2)*dz*normals(elem,k,2) +
        mu_val*basis_grad(elem,inode,k,0)*(dz*normals(elem,k,0) + dx*normals(elem,k,2)) +
        mu_val*basis_grad(elem,inode,k,1)*(dz*normals(elem,k,1) + dy*normals(elem,k,2));
        
      }
    }
    
    
    return basisVec;
  }
  
  
  // ========================================================================================
  // TMW: this has been deprecated ... capability will be replaced
  // ========================================================================================
  
  vector<string> extraCellFieldNames() const {
    
    vector<string> ef;// = udfunc->extraCellFieldNames(label);
    /*
     if (spaceDim == 2) {
     ef.push_back("sig11");
     ef.push_back("sig12");
     ef.push_back("sig22");
     }
     else if (spaceDim == 3) {
     ef.push_back("sig11");
     ef.push_back("sig12");
     ef.push_back("sig13");
     ef.push_back("sig22");
     ef.push_back("sig23");
     ef.push_back("sig33");
     }
     ef.push_back("vm_stress");
     */
    return ef;
  }
  
  // ========================================================================================
  // TMW: this has been deprecated ... capability will be replaced
  // ========================================================================================
  
  vector<Kokkos::View<ScalarT***,AssemblyDevice>> extraCellFields() {
    vector<Kokkos::View<ScalarT***,AssemblyDevice>> ef;// = udfunc->extraCellFields(label,wkset);
    /*
     DRV wts = wkset->wts;
     udfunc->coefficient("lambda",wkset,false,lambda);
     udfunc->coefficient("mu",wkset,false,mu);
     this->computeStress(false);
     
     Kokkos::View<ScalarT***,AssemblyDevice> svals;
     int numElem = wts.dimension(0);
     if (spaceDim == 2) {
     svals = Kokkos::View<ScalarT***,AssemblyDevice>("svals",numElem,3,1);
     }
     else if (spaceDim == 3) {
     svals = Kokkos::View<ScalarT***,AssemblyDevice>("svals",numElem,6,1);
     }
     
     for (int e=0; e<numElem; e++) {
     ScalarT vol = 0.0;
     for (size_t k=0; k<numip; k++) {
     vol += wts(e,k);
     }
     for (int j=0; j<numip; j++) {
     ScalarT cwt = wts(e,j)/vol;
     if (spaceDim == 2) {
     svals(e,0,0) += (stress(e,j,0,0).val())*cwt;
     svals(e,1,0) += (stress(e,j,0,1).val())*cwt;
     svals(e,2,0) += (stress(e,j,1,1).val())*cwt;
     }
     else if (spaceDim == 3) {
     svals(e,0,0) += (stress(e,j,0,0).val())*cwt;
     svals(e,1,0) += (stress(e,j,0,1).val())*cwt;
     svals(e,2,0) += (stress(e,j,0,2).val())*cwt;
     svals(e,3,0) += (stress(e,j,1,1).val())*cwt;
     svals(e,4,0) += (stress(e,j,1,2).val())*cwt;
     svals(e,5,0) += (stress(e,j,2,2).val())*cwt;
     }
     }
     }
     ef.push_back(svals);
     
     
     Kokkos::View<ScalarT***,AssemblyDevice> vms("von mises stress",numElem,1,1);
     
     if (spaceDim == 2) {
     for (int e=0; e<numElem; e++) {
     ScalarT vol = 0.0;
     for (size_t k=0; k<numip; k++) {
     vol += wts(e,k);
     }
     for (int j=0; j<numip; j++) {
     ScalarT cwt = wts(e,j)/vol;
     ScalarT pratio = lambda(e,j).val() / (2.0*(lambda(e,j).val() + mu(e,j).val()));
     ScalarT s11 = stress(e,j,0,0).val();
     ScalarT s22 = stress(e,j,1,1).val();
     ScalarT s33 = pratio*(s11 + s22);
     ScalarT s12 = stress(e,j,0,1).val();
     
     vms(e,0,0) += sqrt(0.5*( pow(s11 - s22,2.0) + pow(s22 - s33,2.0)
     + pow(s33 - s11,2.0) + 6.0*s12*s12))*cwt;
     }
     }
     }
     else if (spaceDim == 3) {
     for (int e=0; e<numElem; e++) {
     ScalarT vol = 0.0;
     for (size_t k=0; k<numip; k++) {
     vol += wts(e,k);
     }
     for (int j=0; j<numip; j++) {
     ScalarT cwt = wts(e,j)/vol;
     vms(e,0,0) += sqrt(0.5*( pow(stress(e,j,0,0).val() - stress(e,j,1,1).val(),2.0) +
     pow(stress(e,j,1,1).val() - stress(e,j,2,2).val(),2.0) +
     pow(stress(e,j,2,2).val() - stress(e,j,0,0).val(),2.0) +
     6.0*(pow(stress(e,j,1,2).val(),2.0) + pow(stress(e,j,2,0).val(),2.0) + pow(stress(e,j,0,1).val(),2.0))))*cwt;
     }
     }
     }
     
     ef.push_back(vms);
     */
    return ef;
    
  }
  
  
  // ========================================================================================
  // TMW: needs to be deprecated
  // ========================================================================================
  
  void updateParameters(const vector<Teuchos::RCP<vector<AD> > > & params,
                        const vector<string> & paramnames) {
    if (useCE) {
      crystalelast->updateParams(wkset);
    }
  }
  
  
private:
  
  size_t numip, numip_side, blocknum;
  
  int spaceDim, numElem, numParams, numResponses;
  vector<string> varlist;
  int dx_num, dy_num, dz_num;
  int e_num, p_num;
  int test, simNum, cell_num;
  string response_type;
  // Parameters
  //    ScalarT lambda, mu;
  
  ScalarT v, dvdx, dvdy, dvdz;
  int resindex, dx_basis, dy_basis, dz_basis;
  ScalarT time;
  ScalarT x,y,z;
  
  // The notation here is a little unfortunate for the derivatives
  // Attempting to make it clearer by using dvar_dx where var = {dx,dy,dz}
  AD dx, ddx_dx, ddx_dy, ddx_dz;
  AD dy, ddy_dx, ddy_dy, ddy_dz;
  AD dz, ddz_dx, ddz_dy, ddz_dz;
  
  AD dpdx, dpdy, dpdz, eval, delta_e, pval;
  AD plambdax, plambday, plambdaz;
  
  //Kokkos::View<AD**,AssemblyDevice> lambda, mu, source_dx, source_dy, source_dz;
  //Kokkos::View<AD**,AssemblyDevice> lambda_side, mu_side, sourceN_dx, sourceN_dy, sourceN_dz;
  
  FDATA lambda, mu, source_dx, source_dy, source_dz;
  FDATA lambda_side, mu_side, sourceN_dx, sourceN_dy, sourceN_dz;
  
  Kokkos::View<AD****,AssemblyDevice> stress;
  
  Kokkos::View<int****,AssemblyDevice> sideinfo;
  
  bool multiscale, useLame, addBiot, useCE;
  bool incplanestress;
  bool disp_response_type;
  ScalarT formparam, biot_alpha, e_ref, alpha_T, epen;
  
  Teuchos::RCP<CrystalElastic> crystalelast;
  
  Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::elasticity::volumeResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::elasticity::volumeResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::elasticity::boundaryResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::elasticity::boundaryResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MILO::elasticity::computeFlux() - function evaluation");
  Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MILO::elasticity::computeFlux() - evaluation of flux");
  Teuchos::RCP<Teuchos::Time> setLocalSol = Teuchos::TimeMonitor::getNewCounter("MILO::elasticity::setLocalSoln()");
  Teuchos::RCP<Teuchos::Time> fillStress = Teuchos::TimeMonitor::getNewCounter("MILO::elasticity::computeStress()");
  Teuchos::RCP<Teuchos::Time> computeBasis = Teuchos::TimeMonitor::getNewCounter("MILO::elasticity::computeBasisVec()");
  
};

#endif
