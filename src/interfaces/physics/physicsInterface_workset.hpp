/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
************************************************************************/


/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

// These cannot be templated (unfortunately)

void PhysicsInterface::setWorkset(vector<Teuchos::RCP<Workset<ScalarT> > > & wkset) {
  for (size_t block = 0; block<wkset.size(); block++) {
    if (wkset[block]->isInitialized) {
      for (size_t set=0; set<modules.size(); set++) {
        wkset[block]->updatePhysicsSet(set);
        for (size_t i=0; i<modules[set][block].size(); i++) {
          modules[set][block][i]->setWorkset(wkset[block]);
        }
      }
    }
  }
}

#ifndef MrHyDE_NO_AD
void PhysicsInterface::setWorkset(vector<Teuchos::RCP<Workset<AD> > > & wkset) {
  for (size_t block = 0; block<wkset.size(); block++) {
    if (wkset[block]->isInitialized) {
      for (size_t set=0; set<modules_AD.size(); set++) {
        wkset[block]->updatePhysicsSet(set);
        for (size_t i=0; i<modules_AD[set][block].size(); i++) {
          modules_AD[set][block][i]->setWorkset(wkset[block]);
        }
      }
    }
  }
}

void PhysicsInterface::setWorkset(vector<Teuchos::RCP<Workset<AD2> > > & wkset) {
  for (size_t block = 0; block<wkset.size(); block++) {
    if (wkset[block]->isInitialized) {
      for (size_t set=0; set<modules_AD2.size(); set++) {
        wkset[block]->updatePhysicsSet(set);
        for (size_t i=0; i<modules_AD2[set][block].size(); i++) {
          modules_AD2[set][block][i]->setWorkset(wkset[block]);
        }
      }
    }
  }
}

void PhysicsInterface::setWorkset(vector<Teuchos::RCP<Workset<AD4> > > & wkset) {
  for (size_t block = 0; block<wkset.size(); block++) {
    if (wkset[block]->isInitialized) {
      for (size_t set=0; set<modules_AD4.size(); set++) {
        wkset[block]->updatePhysicsSet(set);
        for (size_t i=0; i<modules_AD4[set][block].size(); i++) {
          modules_AD4[set][block][i]->setWorkset(wkset[block]);
        }
      }
    }
  }
}

void PhysicsInterface::setWorkset(vector<Teuchos::RCP<Workset<AD8> > > & wkset) {
  for (size_t block = 0; block<wkset.size(); block++) {
    if (wkset[block]->isInitialized) {
      for (size_t set=0; set<modules_AD8.size(); set++) {
        wkset[block]->updatePhysicsSet(set);
        for (size_t i=0; i<modules_AD8[set][block].size(); i++) {
          modules_AD8[set][block][i]->setWorkset(wkset[block]);
        }
      }
    }
  }
}

void PhysicsInterface::setWorkset(vector<Teuchos::RCP<Workset<AD16> > > & wkset) {
  for (size_t block = 0; block<wkset.size(); block++) {
    if (wkset[block]->isInitialized) {
      for (size_t set=0; set<modules_AD16.size(); set++) {
        wkset[block]->updatePhysicsSet(set);
        for (size_t i=0; i<modules_AD16[set][block].size(); i++) {
          modules_AD16[set][block][i]->setWorkset(wkset[block]);
        }
      }
    }
  }
}

void PhysicsInterface::setWorkset(vector<Teuchos::RCP<Workset<AD18> > > & wkset) {
  for (size_t block = 0; block<wkset.size(); block++) {
    if (wkset[block]->isInitialized) {
      for (size_t set=0; set<modules_AD18.size(); set++) {
        wkset[block]->updatePhysicsSet(set);
        for (size_t i=0; i<modules_AD18[set][block].size(); i++) {
          modules_AD18[set][block][i]->setWorkset(wkset[block]);
        }
      }
    }
  }
}

void PhysicsInterface::setWorkset(vector<Teuchos::RCP<Workset<AD24> > > & wkset) {
  for (size_t block = 0; block<wkset.size(); block++) {
    if (wkset[block]->isInitialized) {
      for (size_t set=0; set<modules_AD24.size(); set++) {
        wkset[block]->updatePhysicsSet(set);
        for (size_t i=0; i<modules_AD24[set][block].size(); i++) {
          modules_AD24[set][block][i]->setWorkset(wkset[block]);
        }
      }
    }
  }
}

void PhysicsInterface::setWorkset(vector<Teuchos::RCP<Workset<AD32> > > & wkset) {
  for (size_t block = 0; block<wkset.size(); block++) {
    if (wkset[block]->isInitialized) {
      for (size_t set=0; set<modules_AD32.size(); set++) {
        wkset[block]->updatePhysicsSet(set);
        for (size_t i=0; i<modules_AD32[set][block].size(); i++) {
          modules_AD32[set][block][i]->setWorkset(wkset[block]);
        }
      }
    }
  }
}
#endif
