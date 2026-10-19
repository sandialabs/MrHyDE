/***********************************************************************
 MrHyDE - Parsed configuration of the parameter Riesz metric, the ROL
 control inner product.

 Questions? Contact Alexey Voronin
 ************************************************************************/

#ifndef MRHYDE_PARAM_RIESZ_CONFIG_HPP
#define MRHYDE_PARAM_RIESZ_CONFIG_HPP

#include "trilinos.hpp"
#include "preferences.hpp"
#include "block_prec/ParamUtils.hpp"

namespace MrHyDE {

  // Parsed configuration of the parameter Riesz metric (the ROL control inner product).
  struct ParamRieszConfig {
    std::string type = "mass";        // mass, sobolev, or regularizer
    std::string apply = "diagonal";   // diagonal, exact or iterative
    std::string solver = "KLU2";      // Amesos2 solver for apply exact
    bool drop_zeros = false;          // factor only the stored nonzeros
    bool strict = false;              // iterative apply: abort on non-convergence instead of warning
    std::string krylov = "CG";        // CG or GMRES for apply iterative
    std::string prec = "RefMaxwell";  // RefMaxwell or Jacobi for apply iterative
    std::string xml;                  // MueLu list for RefMaxwell
    Teuchos::RCP<Teuchos::ParameterList> belos;  // tolerance, iterations, Belos Settings
    std::string blocks;               // blank means every block
    std::string file;                 // blank means the operator is not written
    ScalarT l2_weight = 1.0, semi_weight = 0.0;

    // present: a "Parameter Riesz map" sublist exists, which flips the type/apply defaults.
    void parse(Teuchos::ParameterList & rset, const bool present) {
      type = rset.get<std::string>("type", present ? "regularizer" : "mass");
      apply = rset.get<std::string>("apply", present ? "exact" : "diagonal");
      solver = rset.get<std::string>("direct solver", "KLU2");
      drop_zeros = rset.get<bool>("drop explicit zeros", false);
      strict = false;
      if (apply == "iterative") {
        Teuchos::ParameterList & iset = rset.sublist("Iterative Settings");
        krylov = iset.get<std::string>("solver", "CG");
        prec = iset.get<std::string>("preconditioner", "RefMaxwell");
        strict = iset.get<bool>("strict", false);
        xml = iset.get<std::string>("xml param file", "");
        belos = Teuchos::rcp(new Teuchos::ParameterList("Belos"));
        belos->set("Convergence Tolerance",
                   static_cast<double>(block_prec::scalarParam(iset, "tolerance", 1.0e-14)));
        belos->set("Maximum Iterations", iset.get<int>("max iterations", 500));
        if (krylov == "GMRES") { belos->set("Num Blocks", 50); }
        belos->setParameters(iset.sublist("Belos Settings"));
      }
      l2_weight = block_prec::scalarParam(rset, "l2 weight", 1.0);
      semi_weight = block_prec::scalarParam(rset, "seminorm weight", 0.0);
      blocks = rset.get<std::string>("blocks", "");
      file = rset.get<std::string>("output file", "");
    }
  };

} // namespace MrHyDE

#endif
