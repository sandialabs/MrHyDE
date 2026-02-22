/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "maxwell_bianisotropic.hpp"
using namespace MrHyDE;

template<class EvalT>
maxwell_bianisotropic<EvalT>::maxwell_bianisotropic(Teuchos::ParameterList & settings, const int & dimension_)
: PhysicsBase<EvalT>(settings, dimension_)
{
  
  label = "maxwell_bianisotropic";
  
  spaceDim = dimension_;
  if (spaceDim < 3) {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: bianisotropic maxwell only runs in 3D");
  }
  include_Beqn = false;
  include_Eeqn = false;
    
  if (settings.isParameter("active variables")) {
    string active = settings.get<string>("active variables");
    std::size_t foundE = active.find("E");
    if (foundE!=std::string::npos) {
      include_Eeqn = true;
    }
    std::size_t foundB = active.find("B");
    if (foundB!=std::string::npos) {
      include_Beqn = true;
    }
  }
  else {
    include_Beqn = true;
    include_Eeqn = true;
  }
  
  if (include_Eeqn) {
    myvars.push_back("E");
    mybasistypes.push_back("HCURL");
  }
  if (include_Beqn) {
    myvars.push_back("B");
    mybasistypes.push_back("HDIV");
  }
    
  useLeapFrog = settings.get<bool>("use leap frog",false);
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void maxwell_bianisotropic<EvalT>::defineFunctions(Teuchos::ParameterList & fs,
                              Teuchos::RCP<FunctionManager<EvalT> > & functionManager_) {
  
  functionManager = functionManager_;
  
  functionManager->addFunction("current x",fs.get<string>("current x","0.0"),"ip");
  functionManager->addFunction("current y",fs.get<string>("current y","0.0"),"ip");
  functionManager->addFunction("current z",fs.get<string>("current z","0.0"),"ip");
  functionManager->addFunction("mu0",fs.get<string>("permeability","1.0"),"ip");
  
  functionManager->addFunction("eta0",fs.get<string>("eta0","1.0"),"ip");
  functionManager->addFunction("refractive index",fs.get<string>("refractive index","1.0"),"ip");
  functionManager->addFunction("epsilon0",fs.get<string>("permittivity","1.0"),"ip");
  
  functionManager->addFunction("epsr_xx",fs.get<string>("epsr_xx","1.0"),"ip");
  functionManager->addFunction("epsr_xy",fs.get<string>("epsr_xy","0.0"),"ip");
  functionManager->addFunction("epsr_xz",fs.get<string>("epsr_xz","0.0"),"ip");
  functionManager->addFunction("epsr_yx",fs.get<string>("epsr_yx","0.0"),"ip");
  functionManager->addFunction("epsr_yy",fs.get<string>("epsr_yy","1.0"),"ip");
  functionManager->addFunction("epsr_yz",fs.get<string>("epsr_yz","0.0"),"ip");
  functionManager->addFunction("epsr_zx",fs.get<string>("epsr_zx","0.0"),"ip");
  functionManager->addFunction("epsr_zy",fs.get<string>("epsr_zy","0.0"),"ip");
  functionManager->addFunction("epsr_zz",fs.get<string>("epsr_zz","1.0"),"ip");
  
  functionManager->addFunction("invmur_xx",fs.get<string>("invmur_xx","1.0"),"ip");
  functionManager->addFunction("invmur_xy",fs.get<string>("invmur_xy","0.0"),"ip");
  functionManager->addFunction("invmur_xz",fs.get<string>("invmur_xz","0.0"),"ip");
  functionManager->addFunction("invmur_yx",fs.get<string>("invmur_yx","0.0"),"ip");
  functionManager->addFunction("invmur_yy",fs.get<string>("invmur_yy","1.0"),"ip");
  functionManager->addFunction("invmur_yz",fs.get<string>("invmur_yz","0.0"),"ip");
  functionManager->addFunction("invmur_zx",fs.get<string>("invmur_zx","0.0"),"ip");
  functionManager->addFunction("invmur_zy",fs.get<string>("invmur_zy","0.0"),"ip");
  functionManager->addFunction("invmur_zz",fs.get<string>("invmur_zz","1.0"),"ip");
  
  functionManager->addFunction("xir_xx",fs.get<string>("xir_xx","1.0"),"ip");
  functionManager->addFunction("xir_xy",fs.get<string>("xir_xy","0.0"),"ip");
  functionManager->addFunction("xir_xz",fs.get<string>("xir_xz","0.0"),"ip");
  functionManager->addFunction("xir_yx",fs.get<string>("xir_yx","0.0"),"ip");
  functionManager->addFunction("xir_yy",fs.get<string>("xir_yy","1.0"),"ip");
  functionManager->addFunction("xir_yz",fs.get<string>("xir_yz","0.0"),"ip");
  functionManager->addFunction("xir_zx",fs.get<string>("xir_zx","0.0"),"ip");
  functionManager->addFunction("xir_zy",fs.get<string>("xir_zy","0.0"),"ip");
  functionManager->addFunction("xir_zz",fs.get<string>("xir_zz","1.0"),"ip");
  
  functionManager->addFunction("zetar_xx",fs.get<string>("zetar_xx","1.0"),"ip");
  functionManager->addFunction("zetar_xy",fs.get<string>("zetar_xy","0.0"),"ip");
  functionManager->addFunction("zetar_xz",fs.get<string>("zetar_xz","0.0"),"ip");
  functionManager->addFunction("zetar_yx",fs.get<string>("zetar_yx","0.0"),"ip");
  functionManager->addFunction("zetar_yy",fs.get<string>("zetar_yy","1.0"),"ip");
  functionManager->addFunction("zetar_yz",fs.get<string>("zetar_yz","0.0"),"ip");
  functionManager->addFunction("zetar_zx",fs.get<string>("zetar_zx","0.0"),"ip");
  functionManager->addFunction("zetar_zy",fs.get<string>("zetar_zy","0.0"),"ip");
  functionManager->addFunction("zetar_zz",fs.get<string>("zetar_zz","1.0"),"ip");
  
  functionManager->addFunction("sigma_xx",fs.get<string>("sigma_xx","0.0"),"ip");
  functionManager->addFunction("sigma_xy",fs.get<string>("sigma_xy","0.0"),"ip");
  functionManager->addFunction("sigma_xz",fs.get<string>("sigma_xz","0.0"),"ip");
  functionManager->addFunction("sigma_yx",fs.get<string>("sigma_yx","0.0"),"ip");
  functionManager->addFunction("sigma_yy",fs.get<string>("sigma_yy","0.0"),"ip");
  functionManager->addFunction("sigma_yz",fs.get<string>("sigma_yz","0.0"),"ip");
  functionManager->addFunction("sigma_zx",fs.get<string>("sigma_zx","0.0"),"ip");
  functionManager->addFunction("sigma_zy",fs.get<string>("sigma_zy","0.0"),"ip");
  functionManager->addFunction("sigma_zz",fs.get<string>("sigma_zz","0.0"),"ip");
  
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void maxwell_bianisotropic<EvalT>::volumeResidual() {
  
  Teuchos::TimeMonitor resideval(*volumeResidualFill);
  
  int stage = wkset->current_stage;
  
  if (include_Beqn) {
    int B_basis = wkset->usebasis[Bnum];
    
    Vista<EvalT> mu0, epsilon0;
    {
      Teuchos::TimeMonitor funceval(*volumeResidualFunc);
      mu0 = functionManager->evaluate("mu0","ip");
      epsilon0 = functionManager->evaluate("epsilon0","ip");
    }
    
    // (1/c0*dB/dt + curl E,V) = 0
    
    auto off = subview(wkset->offsets, Bnum, ALL());
    auto wts = wkset->wts;
    auto res = wkset->res;
    auto basis = wkset->basis[B_basis];
    auto dBx_dt = wkset->getSolutionField("B_t[x]");
    auto dBy_dt = wkset->getSolutionField("B_t[y]");
    auto dBz_dt = wkset->getSolutionField("B_t[z]");
    
    if (useLeapFrog) {
      if (stage == 0) {
        auto curlE_x = wkset->getSolutionField("curl(E)[x]");
        auto curlE_y = wkset->getSolutionField("curl(E)[y]");
        auto curlE_z = wkset->getSolutionField("curl(E)[z]");
        
        parallel_for("maxwell_bianisotropics B volume resid",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     MRHYDE_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            EvalT c0 = 1.0/std::sqrt(mu0(elem,pt)*epsilon0(elem,pt));
            EvalT f0 = (1.0/c0*dBx_dt(elem,pt) + curlE_x(elem,pt))*wts(elem,pt);
            EvalT f1 = (1.0/c0*dBy_dt(elem,pt) + curlE_y(elem,pt))*wts(elem,pt);
            EvalT f2 = (1.0/c0*dBz_dt(elem,pt) + curlE_z(elem,pt))*wts(elem,pt);
            for (size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += f0*basis(elem,dof,pt,0);
              res(elem,off(dof)) += f1*basis(elem,dof,pt,1);
              res(elem,off(dof)) += f2*basis(elem,dof,pt,2);
            }
          }
        });
      }
      else {
        
        parallel_for("maxwell_bianisotropics B volume resid",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     MRHYDE_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            EvalT c0 = 1.0/std::sqrt(mu0(elem,pt)*epsilon0(elem,pt));
            EvalT f0 = 1.0/c0*dBx_dt(elem,pt)*wts(elem,pt);
            EvalT f1 = 1.0/c0*dBy_dt(elem,pt)*wts(elem,pt);
            EvalT f2 = 1.0/c0*dBz_dt(elem,pt)*wts(elem,pt);
            for (size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += f0*basis(elem,dof,pt,0);
              res(elem,off(dof)) += f1*basis(elem,dof,pt,1);
              res(elem,off(dof)) += f2*basis(elem,dof,pt,2);
            }
          }
        });
        
      }
    }
    else {
      auto curlE_x = wkset->getSolutionField("curl(E)[x]");
      auto curlE_y = wkset->getSolutionField("curl(E)[y]");
      auto curlE_z = wkset->getSolutionField("curl(E)[z]");
      
      parallel_for("maxwell_bianisotropics B volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   MRHYDE_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          EvalT c0 = 1.0/std::sqrt(mu0(elem,pt)*epsilon0(elem,pt));
          EvalT f0 = (1.0/c0*dBx_dt(elem,pt) + curlE_x(elem,pt))*wts(elem,pt);
          EvalT f1 = (1.0/c0*dBy_dt(elem,pt) + curlE_y(elem,pt))*wts(elem,pt);
          EvalT f2 = (1.0/c0*dBz_dt(elem,pt) + curlE_z(elem,pt))*wts(elem,pt);
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += f0*basis(elem,dof,pt,0);
            res(elem,off(dof)) += f1*basis(elem,dof,pt,1);
            res(elem,off(dof)) += f2*basis(elem,dof,pt,2);
          }
        }
      });
      
    }
  }
  
  if (include_Eeqn) {
    // 1/c0*([eps]-[xi]*[invmu]*[zeta])*dE/dt,V) + 1/c0*([xi]*[invmu]*dB/dt,V) - ([invmu]*B - [invmu]*[zeta]*E, curl V) + (eta0*[sigma] E,V) = -(eta0*current,V)
    
    int E_basis = wkset->usebasis[Enum];
    
    Vista<EvalT> mu0, epsilon0, eta0, rindex;
    Vista<EvalT> current_x, current_y, current_z;
    Vista<EvalT> epsr_xx, epsr_xy, epsr_xz, epsr_yx, epsr_yy, epsr_yz, epsr_zx, epsr_zy, epsr_zz;
    Vista<EvalT> invmur_xx, invmur_xy, invmur_xz, invmur_yx, invmur_yy, invmur_yz, invmur_zx, invmur_zy, invmur_zz;
    Vista<EvalT> xir_xx, xir_xy, xir_xz, xir_yx, xir_yy, xir_yz, xir_zx, xir_zy, xir_zz;
    Vista<EvalT> zetar_xx, zetar_xy, zetar_xz, zetar_yx, zetar_yy, zetar_yz, zetar_zx, zetar_zy, zetar_zz;
    Vista<EvalT> sigma_xx, sigma_xy, sigma_xz, sigma_yx, sigma_yy, sigma_yz, sigma_zx, sigma_zy, sigma_zz;
    {
      Teuchos::TimeMonitor funceval(*volumeResidualFunc);
      current_x = functionManager->evaluate("current x","ip");
      current_y = functionManager->evaluate("current y","ip");
      current_z = functionManager->evaluate("current z","ip");
      
      mu0 = functionManager->evaluate("mu0","ip");
      epsilon0 = functionManager->evaluate("epsilon0","ip");
      rindex = functionManager->evaluate("refractive index","ip");
      eta0 = functionManager->evaluate("eta0","ip");
      
      epsr_xx = functionManager->evaluate("invepsr_xx","ip");
      epsr_xy = functionManager->evaluate("invepsr_xy","ip");
      epsr_xz = functionManager->evaluate("invepsr_xz","ip");
      epsr_yx = functionManager->evaluate("invepsr_yx","ip");
      epsr_yy = functionManager->evaluate("invepsr_yy","ip");
      epsr_yz = functionManager->evaluate("invepsr_yz","ip");
      epsr_zx = functionManager->evaluate("invepsr_zx","ip");
      epsr_zy = functionManager->evaluate("invepsr_yz","ip");
      epsr_zz = functionManager->evaluate("invepsr_zz","ip");
      
      invmur_xx = functionManager->evaluate("mur_xx","ip");
      invmur_xy = functionManager->evaluate("mur_xy","ip");
      invmur_xz = functionManager->evaluate("mur_xz","ip");
      invmur_yx = functionManager->evaluate("mur_yx","ip");
      invmur_yy = functionManager->evaluate("mur_yy","ip");
      invmur_yz = functionManager->evaluate("mur_yz","ip");
      invmur_zx = functionManager->evaluate("mur_zx","ip");
      invmur_zy = functionManager->evaluate("mur_zy","ip");
      invmur_zz = functionManager->evaluate("mur_zz","ip");
      
      xir_xx = functionManager->evaluate("xir_xx","ip");
      xir_xy = functionManager->evaluate("xir_xy","ip");
      xir_xz = functionManager->evaluate("xir_xz","ip");
      xir_yx = functionManager->evaluate("xir_yz","ip");
      xir_yy = functionManager->evaluate("xir_yy","ip");
      xir_yz = functionManager->evaluate("xir_yz","ip");
      xir_zx = functionManager->evaluate("xir_zx","ip");
      xir_zy = functionManager->evaluate("xir_zy","ip");
      xir_zz = functionManager->evaluate("xir_zz","ip");
      
      zetar_xx = functionManager->evaluate("zetar_xx","ip");
      zetar_xy = functionManager->evaluate("zetar_xy","ip");
      zetar_xz = functionManager->evaluate("zetar_xz","ip");
      zetar_yx = functionManager->evaluate("zetar_yz","ip");
      zetar_yy = functionManager->evaluate("zetar_yy","ip");
      zetar_yz = functionManager->evaluate("zetar_yz","ip");
      zetar_zx = functionManager->evaluate("zetar_zx","ip");
      zetar_zy = functionManager->evaluate("zetar_zy","ip");
      zetar_zz = functionManager->evaluate("zetar_zz","ip");
      
      sigma_xx = functionManager->evaluate("sigma_xx","ip");
      sigma_xy = functionManager->evaluate("sigma_xy","ip");
      sigma_xz = functionManager->evaluate("sigma_xz","ip");
      sigma_yx = functionManager->evaluate("sigma_yx","ip");
      sigma_yy = functionManager->evaluate("sigma_yy","ip");
      sigma_yz = functionManager->evaluate("sigma_yz","ip");
      sigma_zx = functionManager->evaluate("sigma_zx","ip");
      sigma_zy = functionManager->evaluate("sigma_zy","ip");
      sigma_zz = functionManager->evaluate("sigma_zz","ip");
      
    }
    
    
    if (!useLeapFrog || stage == 1) {
      auto basis = wkset->basis[E_basis];
      auto basis_curl = wkset->basis_curl[E_basis];
      auto dEx_dt = wkset->getSolutionField("E_t[x]");
      auto dEy_dt = wkset->getSolutionField("E_t[y]");
      auto dEz_dt = wkset->getSolutionField("E_t[z]");
      auto dBx_dt = wkset->getSolutionField("B_t[x]");
      auto dBy_dt = wkset->getSolutionField("B_t[y]");
      auto dBz_dt = wkset->getSolutionField("B_t[z]");
      auto Bx = wkset->getSolutionField("B[x]");
      auto By = wkset->getSolutionField("B[y]");
      auto Bz = wkset->getSolutionField("B[z]");
      auto Ex = wkset->getSolutionField("E[x]");
      auto Ey = wkset->getSolutionField("E[y]");
      auto Ez = wkset->getSolutionField("E[z]");
      auto off = subview(wkset->offsets, Enum, ALL());
      auto wts = wkset->wts;
      auto res = wkset->res;
      
      // 1/c0*([eps]-[xi]*[invmu]*[zeta])*dE/dt,V) + 1/c0*([xi]*[invmu]*dB/dt,V) - ([invmu]*B - [invmu]*[zeta]*E, curl V) + (eta0*[sigma] E,V) = - (eta0*current,V)
      
      parallel_for("maxwell_bianisotropics E volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   MRHYDE_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          EvalT c0 = 1.0/std::sqrt(epsilon0(elem,pt)*mu0(elem,pt));
          
          // compute [eps]*dEdt
          EvalT eps_dEdt_x = epsr_xx(elem,pt)*dEx_dt(elem,pt) + epsr_xy(elem,pt)*dEy_dt(elem,pt) + epsr_xz(elem,pt)*dEz_dt(elem,pt);
          EvalT eps_dEdt_y = epsr_yx(elem,pt)*dEx_dt(elem,pt) + epsr_yy(elem,pt)*dEy_dt(elem,pt) + epsr_yz(elem,pt)*dEz_dt(elem,pt);
          EvalT eps_dEdt_z = epsr_zx(elem,pt)*dEx_dt(elem,pt) + epsr_zy(elem,pt)*dEy_dt(elem,pt) + epsr_zz(elem,pt)*dEz_dt(elem,pt);
          
          // compute [xi]*[invmu]*[zeta]*dEdt
          EvalT zeta_dEdt_x = zetar_xx(elem,pt)*dEx_dt(elem,pt) + zetar_xy(elem,pt)*dEy_dt(elem,pt) + zetar_xz(elem,pt)*dEz_dt(elem,pt);
          EvalT zeta_dEdt_y = zetar_yx(elem,pt)*dEx_dt(elem,pt) + zetar_yy(elem,pt)*dEy_dt(elem,pt) + zetar_yz(elem,pt)*dEz_dt(elem,pt);
          EvalT zeta_dEdt_z = zetar_zx(elem,pt)*dEx_dt(elem,pt) + zetar_zy(elem,pt)*dEy_dt(elem,pt) + zetar_zz(elem,pt)*dEz_dt(elem,pt);
          
          EvalT invmu_zeta_dEdt_x = invmur_xx(elem,pt)*zeta_dEdt_x + invmur_xy(elem,pt)*zeta_dEdt_y + invmur_xz(elem,pt)*zeta_dEdt_z;
          EvalT invmu_zeta_dEdt_y = invmur_yx(elem,pt)*zeta_dEdt_x + invmur_yy(elem,pt)*zeta_dEdt_y + invmur_yz(elem,pt)*zeta_dEdt_z;
          EvalT invmu_zeta_dEdt_z = invmur_zx(elem,pt)*zeta_dEdt_x + invmur_zy(elem,pt)*zeta_dEdt_y + invmur_zz(elem,pt)*zeta_dEdt_z;
          
          EvalT xi_invmu_zeta_dEdt_x = xir_xx(elem,pt)*invmu_zeta_dEdt_x + xir_xy(elem,pt)*invmu_zeta_dEdt_y + xir_xz(elem,pt)*invmu_zeta_dEdt_z;
          EvalT xi_invmu_zeta_dEdt_y = xir_yx(elem,pt)*invmu_zeta_dEdt_x + xir_yy(elem,pt)*invmu_zeta_dEdt_y + xir_yz(elem,pt)*invmu_zeta_dEdt_z;
          EvalT xi_invmu_zeta_dEdt_z = xir_zx(elem,pt)*invmu_zeta_dEdt_x + xir_zy(elem,pt)*invmu_zeta_dEdt_y + xir_zz(elem,pt)*invmu_zeta_dEdt_z;
          
          // compute [xi]*[invmu]*dB/dt
          EvalT invmu_dBdt_x = invmur_xx(elem,pt)*dBx_dt(elem,pt) + invmur_xy(elem,pt)*dBy_dt(elem,pt) + invmur_xz(elem,pt)*dBz_dt(elem,pt);
          EvalT invmu_dBdt_y = invmur_yx(elem,pt)*dBx_dt(elem,pt) + invmur_yy(elem,pt)*dBy_dt(elem,pt) + invmur_yz(elem,pt)*dBz_dt(elem,pt);
          EvalT invmu_dBdt_z = invmur_zx(elem,pt)*dBx_dt(elem,pt) + invmur_zy(elem,pt)*dBy_dt(elem,pt) + invmur_zz(elem,pt)*dBz_dt(elem,pt);
          
          EvalT xi_invmu_dBdt_x = xir_xx(elem,pt)*invmu_dBdt_x + xir_xy(elem,pt)*invmu_dBdt_y + xir_xz(elem,pt)*invmu_dBdt_z;
          EvalT xi_invmu_dBdt_y = xir_yx(elem,pt)*invmu_dBdt_x + xir_yy(elem,pt)*invmu_dBdt_y + xir_yz(elem,pt)*invmu_dBdt_z;
          EvalT xi_invmu_dBdt_z = xir_zx(elem,pt)*invmu_dBdt_x + xir_zy(elem,pt)*invmu_dBdt_y + xir_zz(elem,pt)*invmu_dBdt_z;
          
          // compute [sigma]*E
          EvalT sigma_E_x = sigma_xx(elem,pt)*Ex(elem,pt) + sigma_xy(elem,pt)*Ey(elem,pt) + sigma_xz(elem,pt)*Ez(elem,pt);
          EvalT sigma_E_y = sigma_yx(elem,pt)*Ex(elem,pt) + sigma_yy(elem,pt)*Ey(elem,pt) + sigma_yz(elem,pt)*Ez(elem,pt);
          EvalT sigma_E_z = sigma_zx(elem,pt)*Ex(elem,pt) + sigma_zy(elem,pt)*Ey(elem,pt) + sigma_zz(elem,pt)*Ez(elem,pt);
          
          // put terms multiplying the basis together
          EvalT fx = (1/c0*(eps_dEdt_x - xi_invmu_zeta_dEdt_x + xi_invmu_dBdt_x) + eta0(elem,pt)*sigma_E_x + eta0(elem,pt)*current_x(elem,pt))*wts(elem,pt);
          EvalT fy = (1/c0*(eps_dEdt_y - xi_invmu_zeta_dEdt_y + xi_invmu_dBdt_y) + eta0(elem,pt)*sigma_E_y + eta0(elem,pt)*current_y(elem,pt))*wts(elem,pt);
          EvalT fz = (1/c0*(eps_dEdt_z - xi_invmu_zeta_dEdt_z + xi_invmu_dBdt_z) + eta0(elem,pt)*sigma_E_z + eta0(elem,pt)*current_z(elem,pt))*wts(elem,pt);
          
          // compute [invmu]*B
          EvalT invmu_B_x = invmur_xx(elem,pt)*Bx(elem,pt) + invmur_xy(elem,pt)*By(elem,pt) + invmur_xz(elem,pt)*Bz(elem,pt);
          EvalT invmu_B_y = invmur_yx(elem,pt)*Bx(elem,pt) + invmur_yy(elem,pt)*By(elem,pt) + invmur_yz(elem,pt)*Bz(elem,pt);
          EvalT invmu_B_z = invmur_zx(elem,pt)*Bx(elem,pt) + invmur_zy(elem,pt)*By(elem,pt) + invmur_zz(elem,pt)*Bz(elem,pt);
          
          // compute [invmu]*[zeta]*E
          EvalT zeta_E_x = zetar_xx(elem,pt)*Ex(elem,pt) + zetar_xy(elem,pt)*Ey(elem,pt) + zetar_xz(elem,pt)*Ez(elem,pt);
          EvalT zeta_E_y = zetar_yx(elem,pt)*Ex(elem,pt) + zetar_yy(elem,pt)*Ey(elem,pt) + zetar_yz(elem,pt)*Ez(elem,pt);
          EvalT zeta_E_z = zetar_zx(elem,pt)*Ex(elem,pt) + zetar_zy(elem,pt)*Ey(elem,pt) + zetar_zz(elem,pt)*Ez(elem,pt);
          
          EvalT invmu_zeta_E_x = invmur_xx(elem,pt)*zeta_E_x + invmur_xy(elem,pt)*zeta_E_y + invmur_xz(elem,pt)*zeta_E_z;
          EvalT invmu_zeta_E_y = invmur_yx(elem,pt)*zeta_E_x + invmur_yy(elem,pt)*zeta_E_y + invmur_yz(elem,pt)*zeta_E_z;
          EvalT invmu_zeta_E_z = invmur_zx(elem,pt)*zeta_E_x + invmur_zy(elem,pt)*zeta_E_y + invmur_zz(elem,pt)*zeta_E_z;
          
          // put terms multiplying the curl(basis) together
          EvalT gx = (-invmu_B_x + invmu_zeta_E_x)*wts(elem,pt);
          EvalT gy = (-invmu_B_y + invmu_zeta_E_y)*wts(elem,pt);
          EvalT gz = (-invmu_B_z + invmu_zeta_E_z)*wts(elem,pt);
          
          // compute the residual
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += fx*basis(elem,dof,pt,0) + gx*basis_curl(elem,dof,pt,0);
            res(elem,off(dof)) += fy*basis(elem,dof,pt,1) + gy*basis_curl(elem,dof,pt,1);
            res(elem,off(dof)) += fz*basis(elem,dof,pt,2) + gz*basis_curl(elem,dof,pt,2);
          }
        }
      });
    }
    
    
     
  }
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void maxwell_bianisotropic<EvalT>::boundaryResidual() {
  
  auto bcs = wkset->var_bcs;
  
  int cside = wkset->currentside;
  
  auto wts = wkset->wts_side;
  auto res = wkset->res;
  
  
  double gamma = -0.9944;
  if (include_Beqn && bcs(Bnum,cside) == "Neumann") { // Really ABC
    // Contributes -<nxnxE,V> along boundary in B equation
    
    View_Sc2 nx, ny, nz;
    nx = wkset->getScalarField("n[x]");
    ny = wkset->getScalarField("n[y]");
    nz = wkset->getScalarField("n[z]");
    auto Ex = wkset->getSolutionField("E[x]");
    auto Ey = wkset->getSolutionField("E[y]");
    auto Ez = wkset->getSolutionField("E[z]");
    
    auto off = subview(wkset->offsets, Bnum, ALL());
    auto basis = wkset->basis_side[wkset->usebasis[Bnum]];
    
    parallel_for("maxwell_bianisotropic bndry resid ABC",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 MRHYDE_LAMBDA (const int elem ) {
      
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        EvalT nce_x = ny(elem,pt)*Ez(elem,pt) - nz(elem,pt)*Ey(elem,pt);
        EvalT nce_y = nz(elem,pt)*Ex(elem,pt) - nx(elem,pt)*Ez(elem,pt);
        EvalT nce_z = nx(elem,pt)*Ey(elem,pt) - ny(elem,pt)*Ex(elem,pt);
        EvalT c0 = -(1.0+gamma)*(ny(elem,pt)*nce_z - nz(elem,pt)*nce_y)*wts(elem,pt);
        EvalT c1 = -(1.0+gamma)*(nz(elem,pt)*nce_x - nx(elem,pt)*nce_z)*wts(elem,pt);
        EvalT c2 = -(1.0+gamma)*(nx(elem,pt)*nce_y - ny(elem,pt)*nce_x)*wts(elem,pt);
        for (size_type dof=0; dof<basis.extent(1); dof++ ) {
          res(elem,off(dof)) += c0*basis(elem,dof,pt,0) + c1*basis(elem,dof,pt,1) + c2*basis(elem,dof,pt,2);
        }
      }
    });
    
  }
  
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void maxwell_bianisotropic<EvalT>::setWorkset(Teuchos::RCP<Workset<EvalT> > & wkset_) {
  
  wkset = wkset_;
  
  vector<string> varlist = wkset->varlist;
  
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "E")
      Enum = i;
    if (varlist[i] == "B")
      Bnum = i;
    //if (varlist[i] == "E2")
    //  E2num = i;
    //if (varlist[i] == "B2")
    //  B2num = i;
  }
}


//////////////////////////////////////////////////////////////
// Explicit template instantiations
//////////////////////////////////////////////////////////////

template class MrHyDE::maxwell_bianisotropic<ScalarT>;

#ifndef MrHyDE_NO_AD
// Custom AD type
template class MrHyDE::maxwell_bianisotropic<AD>;

// Standard built-in types
template class MrHyDE::maxwell_bianisotropic<AD2>;
template class MrHyDE::maxwell_bianisotropic<AD4>;
template class MrHyDE::maxwell_bianisotropic<AD8>;
template class MrHyDE::maxwell_bianisotropic<AD16>;
template class MrHyDE::maxwell_bianisotropic<AD18>;
template class MrHyDE::maxwell_bianisotropic<AD24>;
template class MrHyDE::maxwell_bianisotropic<AD32>;
#endif
