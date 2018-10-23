/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_XMLParser.hpp"
#include "Teuchos_YamlParser_decl.hpp"
#include "Teuchos_XMLParameterListCoreHelpers.hpp"
#include "Teuchos_YamlParameterListCoreHelpers.hpp"
#include "Teuchos_Assert.hpp"
#include <fstream>

using namespace Teuchos;
using namespace std;

int main(int argc, char * argv[]) {
  
  TEUCHOS_ASSERT(argc == 2);
  
  string prefix = argv[1];
  std::string input_xml_file_name(prefix+".xml");
  auto pList = getParametersFromXmlFile(input_xml_file_name);
  
  std::string output_yaml_file_name(prefix+".yaml");
  writeParameterListToYamlFile(*pList,output_yaml_file_name);
  
  return 0;
}


