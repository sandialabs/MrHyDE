/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.

 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
 ************************************************************************/

// ========================================================================================
// ========================================================================================

template <class Node>
View_Sc2 PostprocessManager<Node>::getDerivedQuantities(const int &block, CompressedView<View_Sc2> &wts)
{

  int numElem = wts.extent(0);
  View_Sc2 fields("grp field data", numElem, derivedquantities_list[block].size());

  int prog = 0;

  for (size_t set = 0; set < physics->modules.size(); ++set) {
    for (size_t m = 0; m < physics->modules[set][block].size(); ++m) {

      // vector<View_AD2> dqvals = physics->modules[set][block][m]->getDerivedValues();
      auto dqvals = physics->modules[set][block][m]->getDerivedValues();
      for (size_t k = 0; k < dqvals.size(); k++) {
        auto cfield = subview(fields, ALL(), prog);
        auto cdq = dqvals[k];

        if (cellfield_reduction == "mean") { // default
          parallel_for("physics get extra grp fields", RangePolicy<AssemblyExec>(0, wts.extent(0)), MRHYDE_LAMBDA(const int e) {
            ScalarT grpmeas = 0.0;
            for (size_t pt=0; pt<wts.extent(1); pt++) {
              grpmeas += wts(e,pt);
            }
            for (size_t j=0; j<wts.extent(1); j++) {
              ScalarT val = cdq(e,j);
              cfield(e) += val*wts(e,j)/grpmeas;
            }
          });
        }
        else if (cellfield_reduction == "max") {
          parallel_for("physics get extra grp fields", RangePolicy<AssemblyExec>(0, wts.extent(0)), MRHYDE_LAMBDA(const int e) {
            for (size_t j=0; j<wts.extent(1); j++) {
              ScalarT val = cdq(e,j);
              if (val>cfield(e)) {
                cfield(e) = val;
              }
            }
          });
        }
        else if (cellfield_reduction == "min") {
          parallel_for("physics get extra grp fields", RangePolicy<AssemblyExec>(0, wts.extent(0)), MRHYDE_LAMBDA(const int e) {
            for (size_t j=0; j<wts.extent(1); j++) {
              ScalarT val = cdq(e,j);
              if (val<cfield(e)) {
                cfield(e) = val;
              }
            }
          });
        }

        prog++;
      }
    }
  }
  return fields;
}
