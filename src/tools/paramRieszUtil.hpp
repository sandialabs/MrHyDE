/***********************************************************************
 MrHyDE - Shared helpers for the parameter Riesz metric: the control
 basis's seminorm operator and the element-block name matcher.

 Questions? Contact Alexey Voronin
 ************************************************************************/

#ifndef MRHYDE_PARAM_RIESZ_UTIL_HPP
#define MRHYDE_PARAM_RIESZ_UTIL_HPP

#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>
#include "Teuchos_TestForException.hpp"

namespace MrHyDE {

  // The control basis's natural Sobolev seminorm operator, or "" if it has none.
  inline std::string seminormOpForBasis(const std::string & btype) {
    if (btype.substr(0,5) == "HCURL") { return "curl"; }
    if (btype.substr(0,5) == "HGRAD") { return "grad"; }
    if (btype.substr(0,4) == "HDIV")  { return "div"; }
    return "";
  }

  // Mask of the blocks named in a whitespace-separated spec; throws on a name matching none.
  inline std::vector<bool> matchNamedBlocks(const std::string & spec,
                                            const std::vector<std::string> & block_names,
                                            const std::string & err_prefix) {
    std::vector<bool> mask(block_names.size(), false);
    std::stringstream ss(spec);
    std::string name;
    while (ss >> name) {
      bool found = false;
      for (size_t b=0; b<block_names.size(); ++b) {
        if (block_names[b] == name) { mask[b] = true; found = true; }
      }
      TEUCHOS_TEST_FOR_EXCEPTION(!found, std::runtime_error,
                                 err_prefix + ": '" + name + "' matches no element block.");
    }
    return mask;
  }

} // namespace MrHyDE

#endif
