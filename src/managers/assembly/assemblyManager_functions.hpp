/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
************************************************************************/

#include <cctype>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

// ========================================================================================
// ========================================================================================

template<class Node>
void AssemblyManager<Node>::addFunction(const int & block, const string & name, const string & expression, const string & location) {
  function_managers[block]->addFunction(name, expression, location);
#ifndef MrHyDE_NO_AD
  if (type_AD == -1) {
    function_managers_AD[block]->addFunction(name, expression, location);
  }
  else if (type_AD == 2) {
    function_managers_AD2[block]->addFunction(name, expression, location);
  }
  else if (type_AD == 4) {
    function_managers_AD4[block]->addFunction(name, expression, location);
  }
  else if (type_AD == 8) {
    function_managers_AD8[block]->addFunction(name, expression, location);
  }
  else if (type_AD == 16) {
    function_managers_AD16[block]->addFunction(name, expression, location);
  }
  else if (type_AD == 18) {
    function_managers_AD18[block]->addFunction(name, expression, location);
  }
  else if (type_AD == 24) {
    function_managers_AD24[block]->addFunction(name, expression, location);
  }
  else if (type_AD == 32) {
    function_managers_AD32[block]->addFunction(name, expression, location);
  }
#endif
}

// ========================================================================================
// ========================================================================================

template<class Node>
View_Sc2 AssemblyManager<Node>::evaluateFunction(const int & block, const string & name, const string & location) {

  typedef typename Node::execution_space LA_exec;

  auto data = function_managers[block]->evaluate(name, location);
  size_type num_elem = function_managers[block]->num_elem_;
  size_type num_pts = 0;
  if (location == "ip") {
    num_pts = function_managers[block]->num_ip_;
  }
  else if (location == "side ip") {
    num_pts = function_managers[block]->num_ip_side_;
  }
  else if (location == "point") {
    num_pts = 1;
  }

  View_Sc2 outdata("data from function evaluation", num_elem, num_pts);

  parallel_for("assembly eval func",
                 RangePolicy<LA_exec>(0,num_elem),
                 MRHYDE_LAMBDA (const int elem ) {
    for (size_type pt=0; pt<num_pts; ++pt) {
      outdata(elem,pt) = data(elem,pt);
    }
  });

  return outdata;
}
    

/////////////////////////////////////////////////////////////////////////////////////////////
// After the setup phase, we finalize the function managers
/////////////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::finalizeFunctions() {
  
  debugger->print("**** Starting AssemblyManager::finalizeFunctions()");
  
  for (size_t block=0; block<wkset.size(); ++block) {
    this->finalizeFunctions(function_managers[block], wkset[block]);
  }
  
#ifndef MrHyDE_NO_AD
  if (type_AD == -1) {
    for (size_t block=0; block<wkset_AD.size(); ++block) {
      this->finalizeFunctions(function_managers_AD[block], wkset_AD[block]);
    }
  }
  else if (type_AD == 2) {
    for (size_t block=0; block<wkset_AD2.size(); ++block) {
      this->finalizeFunctions(function_managers_AD2[block], wkset_AD2[block]);
    }
  }
  else if (type_AD == 4) {
    for (size_t block=0; block<wkset_AD4.size(); ++block) {
      this->finalizeFunctions(function_managers_AD4[block], wkset_AD4[block]);
    }
  }
  else if (type_AD == 8) {
    for (size_t block=0; block<wkset_AD8.size(); ++block) {
      this->finalizeFunctions(function_managers_AD8[block], wkset_AD8[block]);
    }
  }
  else if (type_AD == 16) {
    for (size_t block=0; block<wkset_AD16.size(); ++block) {
      this->finalizeFunctions(function_managers_AD16[block], wkset_AD16[block]);
    }
  }
  else if (type_AD == 18) {
    for (size_t block=0; block<wkset_AD18.size(); ++block) {
      this->finalizeFunctions(function_managers_AD18[block], wkset_AD18[block]);
    }
  }
  else if (type_AD == 24) {
    for (size_t block=0; block<wkset_AD24.size(); ++block) {
      this->finalizeFunctions(function_managers_AD24[block], wkset_AD24[block]);
    }
  }
  else if (type_AD == 32) {
    for (size_t block=0; block<wkset_AD32.size(); ++block) {
      this->finalizeFunctions(function_managers_AD32[block], wkset_AD32[block]);
    }
  }
#endif
  
  debugger->print("**** Finished AssemblyManager::finalizeFunctions()");
  
}

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::finalizeFunctions(Teuchos::RCP<FunctionManager<EvalT> > & fman,
                                              Teuchos::RCP<Workset<EvalT> > & wset) {
  fman->setupLists(params->paramnames);
  fman->wkset = wset;
  if (wset->isInitialized) {
    fman->decomposeFunctions();
    if (verbosity >= 20) {
      fman->printFunctions();
      wset->printSolutionFields();
      wset->printScalarFields();
    }
  }
}



////////////////////////////////////////////////////////////////////////////////
// Configure lumped-port source and conductivity functions
////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::configureLumpedPorts() {

  if (!settings->isSublist("Physics") ||
      !settings->sublist("Physics").isSublist("Lumped ports")) {
    return;
  }

  TEUCHOS_TEST_FOR_EXCEPTION(!store_nodes, std::runtime_error,
                             "Lumped ports require Solver: store nodes: true.");

  Teuchos::ParameterList & port_list =
    settings->sublist("Physics").sublist("Lumped ports");
  Teuchos::ParameterList & function_list = settings->sublist("Functions");

  auto scalarString = [](const ScalarT & value) {
    std::ostringstream stream;
    stream << std::setprecision(17) << value;
    return stream.str();
  };

  auto getExpression = [](const Teuchos::ParameterList & list,
                          const string & name,
                          const string & fallback) {
    if (list.isType<string>(name)) {
      return list.get<string>(name);
    }
    if (list.isType<double>(name)) {
      std::ostringstream stream;
      stream << std::setprecision(17) << list.get<double>(name);
      return stream.str();
    }
    if (list.isType<int>(name)) {
      return std::to_string(list.get<int>(name));
    }
    return fallback;
  };

  auto combineExpression = [&](Teuchos::ParameterList & functions,
                               const string & name,
                               const string & alias,
                               const string & addition) {
    const string existing = getExpression(functions, name,
      getExpression(functions, alias, "0.0"));
    functions.set(name, "(" + existing + ")+(" + addition + ")");
  };

  Teuchos::ParameterList::ConstIterator port_itr = port_list.begin();
  while (port_itr != port_list.end()) {
    const string port_name = port_itr->first;
    TEUCHOS_TEST_FOR_EXCEPTION(!port_list.isSublist(port_name), std::runtime_error,
                               "Each Physics: Lumped ports entry must be a sublist.");

    Teuchos::ParameterList & port_settings = port_list.sublist(port_name);
    const string block_name = port_settings.get<string>("block", "");

    TEUCHOS_TEST_FOR_EXCEPTION(block_name.empty(), std::runtime_error,
                               "Lumped port '" << port_name << "' requires a block name.");

    size_t block = blocknames.size();
    for (size_t b = 0; b < blocknames.size(); ++b) {
      if (blocknames[b] == block_name) {
        block = b;
        break;
      }
    }
    TEUCHOS_TEST_FOR_EXCEPTION(block == blocknames.size(), std::runtime_error,
                               "Lumped port '" << port_name << "' references block '"
                               << block_name << "', which is not present in the mesh.");
    TEUCHOS_TEST_FOR_EXCEPTION(!function_list.isSublist(block_name), std::runtime_error,
                               "Lumped port '" << port_name << "' requires a Functions sublist "
                               << "for block '" << block_name << "'.");

    ScalarT px = 0.0;
    ScalarT py = 0.0;
    ScalarT pz = 0.0;
    if (port_settings.isSublist("polarization")) {
      Teuchos::ParameterList & polarization =
        port_settings.sublist("polarization");
      px = polarization.get<ScalarT>("x", 0.0);
      py = polarization.get<ScalarT>("y", 0.0);
      pz = polarization.get<ScalarT>("z", 0.0);
    }
    else {
      px = port_settings.get<ScalarT>("polarization x", 0.0);
      py = port_settings.get<ScalarT>("polarization y", 0.0);
      pz = port_settings.get<ScalarT>("polarization z", 0.0);
    }

    const ScalarT polarization_norm = std::sqrt(px*px + py*py + pz*pz);
    TEUCHOS_TEST_FOR_EXCEPTION(polarization_norm <= 1.0e-30, std::runtime_error,
                               "Lumped port '" << port_name
                               << "' requires a nonzero polarization vector.");
    px /= polarization_norm;
    py /= polarization_norm;
    pz /= polarization_norm;

    const ScalarT impedance = port_settings.get<ScalarT>("impedance", 50.0);
    const ScalarT amplitude = port_settings.get<ScalarT>("amplitude", 1.0);
    const ScalarT min_frequency =
      port_settings.get<ScalarT>("min_frequency", 0.0);
    const ScalarT max_frequency =
      port_settings.get<ScalarT>("max_frequency", 0.0);
    const ScalarT frequency =
      port_settings.get<ScalarT>("frequency", 0.0);
    const ScalarT offset_multiplier =
      port_settings.get<ScalarT>("offset", 6.0);
    string source_type =
      port_settings.get<string>("type", "gaussian_derivative");
    for (size_t i = 0; i < source_type.size(); ++i) {
      source_type[i] = static_cast<char>(
        std::tolower(static_cast<unsigned char>(source_type[i])));
      if (source_type[i] == '-' || source_type[i] == ' ') {
        source_type[i] = '_';
      }
    }
    if (source_type == "gaussian_derivative" ||
        source_type == "gaussian_deriv") {
      source_type = "gaussian_derivative";
    }
    else if (source_type == "gaussian_sinusoid" ||
             source_type == "gaussian_sinusoidal") {
      source_type = "gaussian_sinusoidal";
    }

    TEUCHOS_TEST_FOR_EXCEPTION(impedance <= 0.0, std::runtime_error,
                               "Lumped port '" << port_name
                               << "' requires a positive impedance.");
    TEUCHOS_TEST_FOR_EXCEPTION(std::abs(amplitude) <= 1.0e-30,
                               std::runtime_error,
                               "Lumped port '" << port_name
                               << "' requires a nonzero amplitude.");

    ScalarT tau = 0.0;
    if (source_type == "gaussian") {
      TEUCHOS_TEST_FOR_EXCEPTION(max_frequency <= 0.0, std::runtime_error,
                                 "Lumped Gaussian source '" << port_name
                                 << "' requires max_frequency > 0.");
      tau = std::sqrt(2.3)/(PI*max_frequency);
    }
    else if (source_type == "gaussian_derivative") {
      TEUCHOS_TEST_FOR_EXCEPTION(max_frequency <= 0.0, std::runtime_error,
                                 "Lumped Gaussian-derivative source '" << port_name
                                 << "' requires max_frequency > 0.");
      tau = std::sqrt(3.815)/(PI*max_frequency);
    }
    else if (source_type == "gaussian_sinusoidal") {
      TEUCHOS_TEST_FOR_EXCEPTION(min_frequency < 0.0 ||
                                 max_frequency <= min_frequency ||
                                 frequency <= 0.0,
                                 std::runtime_error,
                                 "Lumped Gaussian-sinusoidal source '" << port_name
                                 << "' requires 0 <= min_frequency < max_frequency "
                                 << "and frequency > 0.");
      tau = 2.0*std::sqrt(2.3)/(PI*(max_frequency - min_frequency));
    }
    else if (source_type == "sinusoidal") {
      TEUCHOS_TEST_FOR_EXCEPTION(frequency <= 0.0, std::runtime_error,
                                 "Lumped sinusoidal source '" << port_name
                                 << "' requires frequency > 0.");
      tau = 1.5/frequency;
    }
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                                 "Lumped port '" << port_name
                                 << "' type must be gaussian, gaussian_derivative, "
                                 << "gaussian_sinusoidal, or sinusoidal.");
    }

    ScalarT local_volume = 0.0;
    ScalarT local_min = std::numeric_limits<ScalarT>::infinity();
    ScalarT local_max = -std::numeric_limits<ScalarT>::infinity();

    for (size_t grp = 0; grp < groups[block].size(); ++grp) {
      auto group = groups[block][grp];
      TEUCHOS_TEST_FOR_EXCEPTION(!group->have_nodes, std::runtime_error,
                                 "Lumped ports require stored element nodes.");

      auto nodes = group->nodes;
      auto nodes_host = Kokkos::create_mirror_view(nodes);
      Kokkos::deep_copy(nodes_host, nodes);

      const size_type num_nodes = nodes_host.extent(1);
      TEUCHOS_TEST_FOR_EXCEPTION(num_nodes != 4 && num_nodes != 5 &&
                                 num_nodes != 6 && num_nodes != 8,
                                 std::runtime_error,
                                 "Lumped port '" << port_name
                                 << "' requires tet4, pyramid5, wedge6, or hex8 elements.");

      auto tetraVolume = [&](const size_type elem, const size_type n0,
                             const size_type n1, const size_type n2,
                             const size_type n3) {
        const ScalarT ax = nodes_host(elem, n1, 0) - nodes_host(elem, n0, 0);
        const ScalarT ay = nodes_host(elem, n1, 1) - nodes_host(elem, n0, 1);
        const ScalarT az = nodes_host(elem, n1, 2) - nodes_host(elem, n0, 2);
        const ScalarT bx = nodes_host(elem, n2, 0) - nodes_host(elem, n0, 0);
        const ScalarT by = nodes_host(elem, n2, 1) - nodes_host(elem, n0, 1);
        const ScalarT bz = nodes_host(elem, n2, 2) - nodes_host(elem, n0, 2);
        const ScalarT cx = nodes_host(elem, n3, 0) - nodes_host(elem, n0, 0);
        const ScalarT cy = nodes_host(elem, n3, 1) - nodes_host(elem, n0, 1);
        const ScalarT cz = nodes_host(elem, n3, 2) - nodes_host(elem, n0, 2);

        const ScalarT determinant =
          ax*(by*cz - bz*cy) -
          ay*(bx*cz - bz*cx) +
          az*(bx*cy - by*cx);
        return std::abs(determinant)/6.0;
      };

      for (size_type elem = 0; elem < nodes_host.extent(0); ++elem) {
        if (num_nodes == 4) {
          local_volume += tetraVolume(elem, 0, 1, 2, 3);
        }
        else if (num_nodes == 5) {
          local_volume += tetraVolume(elem, 0, 1, 2, 4);
          local_volume += tetraVolume(elem, 0, 2, 3, 4);
        }
        else if (num_nodes == 6) {
          local_volume += tetraVolume(elem, 0, 1, 2, 3);
          local_volume += tetraVolume(elem, 1, 2, 3, 4);
          local_volume += tetraVolume(elem, 2, 3, 4, 5);
        }
        else {
          local_volume += tetraVolume(elem, 0, 1, 3, 4);
          local_volume += tetraVolume(elem, 1, 2, 3, 6);
          local_volume += tetraVolume(elem, 1, 3, 4, 6);
          local_volume += tetraVolume(elem, 1, 4, 5, 6);
          local_volume += tetraVolume(elem, 3, 4, 6, 7);
        }

        for (size_type node = 0; node < num_nodes; ++node) {
          const ScalarT coordinate = px*nodes_host(elem, node, 0) +
                                     py*nodes_host(elem, node, 1) +
                                     pz*nodes_host(elem, node, 2);
          local_min = std::min(local_min, coordinate);
          local_max = std::max(local_max, coordinate);
        }
      }
    }

    ScalarT volume = 0.0;
    ScalarT minimum = 0.0;
    ScalarT maximum = 0.0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1,
                       &local_volume, &volume);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1,
                       &local_min, &minimum);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1,
                       &local_max, &maximum);

    const ScalarT height = maximum - minimum;
    TEUCHOS_TEST_FOR_EXCEPTION(volume <= 0.0 || !std::isfinite(height) ||
                               height <= 0.0, std::runtime_error,
                               "Lumped port '" << port_name
                               << "' has zero volume or zero extent along its polarization.");

    const ScalarT area = volume/height;
    const ScalarT conductivity = height/(area*impedance);
    const ScalarT offset = offset_multiplier*tau;

    port_settings.set("height", height);
    port_settings.set("area", area);
    port_settings.set("volume", volume);
    port_settings.set("conductivity", conductivity);
    port_settings.set("tau", tau);
    port_settings.set("time offset", offset);
    port_settings.set("polarization x", px);
    port_settings.set("polarization y", py);
    port_settings.set("polarization z", pz);
    port_settings.set("type", source_type);

    string waveform;
    const string u = "(t-" + scalarString(offset) + ")";
    const string a = "(" + u + "/" + scalarString(tau) + ")";
    const string envelope = "exp(-" + a + "*" + a + ")";

    if (source_type == "gaussian") {
      waveform = envelope;
    }
    else if (source_type == "gaussian_derivative") {
      waveform = "-2.0*" + a + "*" + envelope;
    }
    else if (source_type == "gaussian_sinusoidal") {
      waveform = envelope + "*cos(2.0*pi*" + scalarString(frequency) +
                 "*" + u + ")";
    }
    else {
      const string ramp_argument = "(min(" + u + ",0.0)/" +
                                   scalarString(tau) + ")";
      waveform = "exp(-" + ramp_argument + "*" + ramp_argument +
                 ")*cos(2.0*pi*" + scalarString(frequency) + "*" + u + ")";
    }

    const string scale = "(-2.0*" + scalarString(amplitude) +
      "/(" + scalarString(area) + "*sqrt(" + scalarString(impedance) + ")))";

    Teuchos::ParameterList & functions = function_list.sublist(block_name);
    combineExpression(functions, "current x", "source J x",
                      scale + "*(" + waveform + ")*" + scalarString(px));
    combineExpression(functions, "current y", "source J y",
                      scale + "*(" + waveform + ")*" + scalarString(py));
    combineExpression(functions, "current z", "source J z",
                      scale + "*(" + waveform + ")*" + scalarString(pz));

    const ScalarT polarization[3] = {px, py, pz};
    const string sigma_names[3][3] = {
      {"sigma_xx", "sigma_xy", "sigma_xz"},
      {"sigma_yx", "sigma_yy", "sigma_yz"},
      {"sigma_zx", "sigma_zy", "sigma_zz"}
    };
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        combineExpression(functions, sigma_names[i][j], sigma_names[i][j],
          scalarString(conductivity*polarization[i]*polarization[j]));
      }
    }

    ++port_itr;
  }
}


////////////////////////////////////////////////////////////////////////////////
// Create the function managers
////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::createFunctions() {
    
  
  for (size_t block=0; block<blocknames.size(); ++block) {
    function_managers.push_back(Teuchos::rcp(new FunctionManager<ScalarT>(blocknames[block],
                                                                     groupData[block]->num_elem,
                                                                     disc->numip[block],
                                                                     disc->numip_side[block])));
  }
  physics->defineFunctions(function_managers);

#ifndef MrHyDE_NO_AD
  if (type_AD == -1) {
    for (size_t block=0; block<blocknames.size(); ++block) {
      function_managers_AD.push_back(Teuchos::rcp(new FunctionManager<AD>(blocknames[block],
                                                                          groupData[block]->num_elem,
                                                                          disc->numip[block],
                                                                          disc->numip_side[block])));
    }
    physics->defineFunctions(function_managers_AD);
  }
  else if (type_AD == 2) {
    for (size_t block=0; block<blocknames.size(); ++block) {
      function_managers_AD2.push_back(Teuchos::rcp(new FunctionManager<AD2>(blocknames[block],
                                                                            groupData[block]->num_elem,
                                                                            disc->numip[block],
                                                                            disc->numip_side[block])));
    }
    physics->defineFunctions(function_managers_AD2);
  }
  else if (type_AD == 4) {
    for (size_t block=0; block<blocknames.size(); ++block) {
      function_managers_AD4.push_back(Teuchos::rcp(new FunctionManager<AD4>(blocknames[block],
                                                                            groupData[block]->num_elem,
                                                                            disc->numip[block],
                                                                            disc->numip_side[block])));
    }
    physics->defineFunctions(function_managers_AD4);
  }
  else if (type_AD == 8) {
    for (size_t block=0; block<blocknames.size(); ++block) {
      function_managers_AD8.push_back(Teuchos::rcp(new FunctionManager<AD8>(blocknames[block],
                                                                            groupData[block]->num_elem,
                                                                            disc->numip[block],
                                                                            disc->numip_side[block])));
    }
    physics->defineFunctions(function_managers_AD8);
  }
  else if (type_AD == 16) {
    for (size_t block=0; block<blocknames.size(); ++block) {
      function_managers_AD16.push_back(Teuchos::rcp(new FunctionManager<AD16>(blocknames[block],
                                                                              groupData[block]->num_elem,
                                                                              disc->numip[block],
                                                                              disc->numip_side[block])));
    }
    physics->defineFunctions(function_managers_AD16);
  }
  else if (type_AD == 18) {
    for (size_t block=0; block<blocknames.size(); ++block) {
      function_managers_AD18.push_back(Teuchos::rcp(new FunctionManager<AD18>(blocknames[block],
                                                                              groupData[block]->num_elem,
                                                                              disc->numip[block],
                                                                              disc->numip_side[block])));
    }
    physics->defineFunctions(function_managers_AD18);
  }
  else if (type_AD == 24) {
    for (size_t block=0; block<blocknames.size(); ++block) {
      function_managers_AD24.push_back(Teuchos::rcp(new FunctionManager<AD24>(blocknames[block],
                                                                              groupData[block]->num_elem,
                                                                              disc->numip[block],
                                                                              disc->numip_side[block])));
    }
    physics->defineFunctions(function_managers_AD24);
  }
  else if (type_AD == 32) {
    for (size_t block=0; block<blocknames.size(); ++block) {
      function_managers_AD32.push_back(Teuchos::rcp(new FunctionManager<AD32>(blocknames[block],
                                                                              groupData[block]->num_elem,
                                                                              disc->numip[block],
                                                                              disc->numip_side[block])));
    }
    physics->defineFunctions(function_managers_AD32);
  }
#endif
}

