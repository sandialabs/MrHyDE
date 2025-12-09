/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/



/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

vector<string> PhysicsInterface::breakupList(const string & list, const string & delimiter) {
  // Script to break delimited list into pieces
  string tmplist = list;
  vector<string> terms;
  size_t pos = 0;
  if (tmplist.find(delimiter) == string::npos) {
    terms.push_back(tmplist);
  }
  else {
    string token;
    while ((pos = tmplist.find(delimiter)) != string::npos) {
      token = tmplist.substr(0, pos);
      terms.push_back(token);
      tmplist.erase(0, pos + delimiter.length());
    }
    terms.push_back(tmplist);
  }
  return terms;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

int PhysicsInterface::getvarOwner(const int & set, const int & block, const string & var) {
  int owner = 0;
  for (size_t k=0; k<var_list[set][block].size(); k++) {
    if (var_list[set][block][k] == var) {
      owner = var_owned[set][block][k];
    }
  }
  return owner;
  
}

/////////////////////////////////////////////////////////////////////////////////////////////
//
/////////////////////////////////////////////////////////////////////////////////////////////

bool PhysicsInterface::checkFace(const size_t & set, const size_t & block){
  bool include_face = false;
  for (size_t i=0; i<modules[set][block].size(); i++) {
    bool cuseef = modules[set][block][i]->include_face;
    if (cuseef) {
      include_face = true;
    }
  }
  
  return include_face;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

int PhysicsInterface::getUniqueIndex(const int & set, const int & block, const std::string & var) {
  int index = 0;
  size_t prog = 0;
  for (size_t set=0; set<num_vars.size(); ++set) {
    for (size_t j=0; j<num_vars[set][block]; j++) {
      if (var_list[set][block][j] == var) {
        index = unique_index[block][j+prog];
      }
    }
    prog += num_vars[set][block];
  }
  return index;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
    
void PhysicsInterface::updateFlags(vector<bool> & newflags) {
  for (size_t set=0; set<modules.size(); set++) {
    for (size_t block=0; block<modules[set].size(); block++) {
      for (size_t i=0; i<modules[set][block].size(); i++) {
        modules[set][block][i]->updateFlags(newflags);
      }
    }
  }
#ifndef MrHyDE_NO_AD
  for (size_t set=0; set<modules_AD.size(); set++) {
    for (size_t block=0; block<modules_AD[set].size(); block++) {
      for (size_t i=0; i<modules_AD[set][block].size(); i++) {
        modules_AD[set][block][i]->updateFlags(newflags);
      }
    }
  }
  for (size_t set=0; set<modules_AD2.size(); set++) {
    for (size_t block=0; block<modules_AD2[set].size(); block++) {
      for (size_t i=0; i<modules_AD2[set][block].size(); i++) {
        modules_AD2[set][block][i]->updateFlags(newflags);
      }
    }
  }
  for (size_t set=0; set<modules_AD4.size(); set++) {
    for (size_t block=0; block<modules_AD4[set].size(); block++) {
      for (size_t i=0; i<modules_AD4[set][block].size(); i++) {
        modules_AD4[set][block][i]->updateFlags(newflags);
      }
    }
  }
  for (size_t set=0; set<modules_AD8.size(); set++) {
    for (size_t block=0; block<modules_AD8[set].size(); block++) {
      for (size_t i=0; i<modules_AD8[set][block].size(); i++) {
        modules_AD8[set][block][i]->updateFlags(newflags);
      }
    }
  }
  for (size_t set=0; set<modules_AD16.size(); set++) {
    for (size_t block=0; block<modules_AD16[set].size(); block++) {
      for (size_t i=0; i<modules_AD16[set][block].size(); i++) {
        modules_AD16[set][block][i]->updateFlags(newflags);
      }
    }
  }
  for (size_t set=0; set<modules_AD18.size(); set++) {
    for (size_t block=0; block<modules_AD18[set].size(); block++) {
      for (size_t i=0; i<modules_AD18[set][block].size(); i++) {
        modules_AD18[set][block][i]->updateFlags(newflags);
      }
    }
  }
  for (size_t set=0; set<modules_AD24.size(); set++) {
    for (size_t block=0; block<modules_AD24[set].size(); block++) {
      for (size_t i=0; i<modules_AD24[set][block].size(); i++) {
        modules_AD24[set][block][i]->updateFlags(newflags);
      }
    }
  }
  for (size_t set=0; set<modules_AD32.size(); set++) {
    for (size_t block=0; block<modules_AD32[set].size(); block++) {
      for (size_t i=0; i<modules_AD32[set][block].size(); i++) {
        modules_AD32[set][block][i]->updateFlags(newflags);
      }
    }
  }
#endif
}
    
/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
    
void PhysicsInterface::purgeMemory() {
  // nothing here
}
