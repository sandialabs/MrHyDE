/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
************************************************************************/


/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

template<class EvalT>
void PhysicsInterface::volumeResidual(const size_t & set, const size_t block) {

  debugger->print(1, "**** Starting PhysicsInterface volume residual ...");

  if (std::is_same<EvalT, ScalarT>::value) {
    for (size_t i=0; i<modules[set][block].size(); i++) {
      modules[set][block][i]->volumeResidual();
    }
  }
#ifndef MrHyDE_NO_AD
  else if (std::is_same<EvalT, AD>::value) {
    for (size_t i=0; i<modules_AD[set][block].size(); i++) {
      modules_AD[set][block][i]->volumeResidual();
    }
  }
  else if (std::is_same<EvalT, AD2>::value) {
    for (size_t i=0; i<modules_AD2[set][block].size(); i++) {
      modules_AD2[set][block][i]->volumeResidual();
    }
  }
  else if (std::is_same<EvalT, AD4>::value) {
    for (size_t i=0; i<modules_AD4[set][block].size(); i++) {
      modules_AD4[set][block][i]->volumeResidual();
    }
  }
  else if (std::is_same<EvalT, AD8>::value) {
    for (size_t i=0; i<modules_AD8[set][block].size(); i++) {
      modules_AD8[set][block][i]->volumeResidual();
    }
  }
  else if (std::is_same<EvalT, AD16>::value) {
    for (size_t i=0; i<modules_AD16[set][block].size(); i++) {
      modules_AD16[set][block][i]->volumeResidual();
    }
  }
  else if (std::is_same<EvalT, AD18>::value) {
    for (size_t i=0; i<modules_AD18[set][block].size(); i++) {
      modules_AD18[set][block][i]->volumeResidual();
    }
  }
  else if (std::is_same<EvalT, AD24>::value) {
    for (size_t i=0; i<modules_AD24[set][block].size(); i++) {
      modules_AD24[set][block][i]->volumeResidual();
    }
  }
  else if (std::is_same<EvalT, AD32>::value) {
    for (size_t i=0; i<modules_AD32[set][block].size(); i++) {
      modules_AD32[set][block][i]->volumeResidual();
    }
  }
#endif
  debugger->print(1, "**** Finished PhysicsInterface volume residual");

}

template void PhysicsInterface::volumeResidual<ScalarT>(const size_t & set, const size_t block);
#ifndef MrHyDE_NO_AD
template void PhysicsInterface::volumeResidual<AD>(const size_t & set, const size_t block);
template void PhysicsInterface::volumeResidual<AD2>(const size_t & set, const size_t block);
template void PhysicsInterface::volumeResidual<AD4>(const size_t & set, const size_t block);
template void PhysicsInterface::volumeResidual<AD8>(const size_t & set, const size_t block);
template void PhysicsInterface::volumeResidual<AD16>(const size_t & set, const size_t block);
template void PhysicsInterface::volumeResidual<AD18>(const size_t & set, const size_t block);
template void PhysicsInterface::volumeResidual<AD24>(const size_t & set, const size_t block);
template void PhysicsInterface::volumeResidual<AD32>(const size_t & set, const size_t block);
#endif

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

template<class EvalT>
void PhysicsInterface::boundaryResidual(const size_t & set, const size_t block) {
  debugger->print(1, "**** Starting PhysicsInterface boundary residual ...");
  
  if (std::is_same<EvalT, ScalarT>::value) {
    for (size_t i=0; i<modules[set][block].size(); i++) {
      modules[set][block][i]->boundaryResidual();
    }
  }
#ifndef MrHyDE_NO_AD
  else if (std::is_same<EvalT, AD>::value) {
    for (size_t i=0; i<modules_AD[set][block].size(); i++) {
      modules_AD[set][block][i]->boundaryResidual();
    }
  }
  else if (std::is_same<EvalT, AD2>::value) {
    for (size_t i=0; i<modules_AD2[set][block].size(); i++) {
      modules_AD2[set][block][i]->boundaryResidual();
    }
  }
  else if (std::is_same<EvalT, AD4>::value) {
    for (size_t i=0; i<modules_AD4[set][block].size(); i++) {
      modules_AD4[set][block][i]->boundaryResidual();
    }
  }
  else if (std::is_same<EvalT, AD8>::value) {
    for (size_t i=0; i<modules_AD8[set][block].size(); i++) {
      modules_AD8[set][block][i]->boundaryResidual();
    }
  }
  else if (std::is_same<EvalT, AD16>::value) {
    for (size_t i=0; i<modules_AD16[set][block].size(); i++) {
      modules_AD16[set][block][i]->boundaryResidual();
    }
  }
  else if (std::is_same<EvalT, AD18>::value) {
    for (size_t i=0; i<modules_AD18[set][block].size(); i++) {
      modules_AD18[set][block][i]->boundaryResidual();
    }
  }
  else if (std::is_same<EvalT, AD24>::value) {
    for (size_t i=0; i<modules_AD24[set][block].size(); i++) {
      modules_AD24[set][block][i]->boundaryResidual();
    }
  }
  else if (std::is_same<EvalT, AD32>::value) {
    for (size_t i=0; i<modules_AD32[set][block].size(); i++) {
      modules_AD32[set][block][i]->boundaryResidual();
    }
  }
#endif
  debugger->print(1, "**** Finished PhysicsInterface boundary residual");
  
}

template void PhysicsInterface::boundaryResidual<ScalarT>(const size_t & set, const size_t block);
#ifndef MrHyDE_NO_AD
template void PhysicsInterface::boundaryResidual<AD>(const size_t & set, const size_t block);
template void PhysicsInterface::boundaryResidual<AD2>(const size_t & set, const size_t block);
template void PhysicsInterface::boundaryResidual<AD4>(const size_t & set, const size_t block);
template void PhysicsInterface::boundaryResidual<AD8>(const size_t & set, const size_t block);
template void PhysicsInterface::boundaryResidual<AD16>(const size_t & set, const size_t block);
template void PhysicsInterface::boundaryResidual<AD18>(const size_t & set, const size_t block);
template void PhysicsInterface::boundaryResidual<AD24>(const size_t & set, const size_t block);
template void PhysicsInterface::boundaryResidual<AD32>(const size_t & set, const size_t block);
#endif


/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

template<class EvalT>
void PhysicsInterface::faceResidual(const size_t & set, const size_t block) {
  if (std::is_same<EvalT, ScalarT>::value) {
    for (size_t i=0; i<modules[set][block].size(); i++) {
      modules[set][block][i]->faceResidual();
    }
  }
#ifndef MrHyDE_NO_AD
  else if (std::is_same<EvalT, AD>::value) {
    for (size_t i=0; i<modules_AD[set][block].size(); i++) {
      modules_AD[set][block][i]->faceResidual();
    }
  }
  else if (std::is_same<EvalT, AD2>::value) {
    for (size_t i=0; i<modules_AD2[set][block].size(); i++) {
      modules_AD2[set][block][i]->faceResidual();
    }
  }
  else if (std::is_same<EvalT, AD4>::value) {
    for (size_t i=0; i<modules_AD4[set][block].size(); i++) {
      modules_AD4[set][block][i]->faceResidual();
    }
  }
  else if (std::is_same<EvalT, AD8>::value) {
    for (size_t i=0; i<modules_AD8[set][block].size(); i++) {
      modules_AD8[set][block][i]->faceResidual();
    }
  }
  else if (std::is_same<EvalT, AD16>::value) {
    for (size_t i=0; i<modules_AD16[set][block].size(); i++) {
      modules_AD16[set][block][i]->faceResidual();
    }
  }
  else if (std::is_same<EvalT, AD18>::value) {
    for (size_t i=0; i<modules_AD18[set][block].size(); i++) {
      modules_AD18[set][block][i]->faceResidual();
    }
  }
  else if (std::is_same<EvalT, AD24>::value) {
    for (size_t i=0; i<modules_AD24[set][block].size(); i++) {
      modules_AD24[set][block][i]->faceResidual();
    }
  }
  else if (std::is_same<EvalT, AD32>::value) {
    for (size_t i=0; i<modules_AD32[set][block].size(); i++) {
      modules_AD32[set][block][i]->faceResidual();
    }
  }
#endif
}

template void PhysicsInterface::faceResidual<ScalarT>(const size_t & set, const size_t block);
#ifndef MrHyDE_NO_AD
template void PhysicsInterface::faceResidual<AD>(const size_t & set, const size_t block);
template void PhysicsInterface::faceResidual<AD2>(const size_t & set, const size_t block);
template void PhysicsInterface::faceResidual<AD4>(const size_t & set, const size_t block);
template void PhysicsInterface::faceResidual<AD8>(const size_t & set, const size_t block);
template void PhysicsInterface::faceResidual<AD16>(const size_t & set, const size_t block);
template void PhysicsInterface::faceResidual<AD18>(const size_t & set, const size_t block);
template void PhysicsInterface::faceResidual<AD24>(const size_t & set, const size_t block);
template void PhysicsInterface::faceResidual<AD32>(const size_t & set, const size_t block);
#endif
