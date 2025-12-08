/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.

 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
 ************************************************************************/

// ========================================================================================
// Create a vector of flux response objects on each block
// ========================================================================================

template <class Node>
void PostprocessManager<Node>::addFluxResponses(Teuchos::ParameterList &flux_resp,
                                                const size_t &block)
{
  Teuchos::ParameterList::ConstIterator fluxr_itr = flux_resp.begin();
  while (fluxr_itr != flux_resp.end())
  {
    Teuchos::ParameterList frsettings = flux_resp.sublist(fluxr_itr->first);
    fluxResponse newflux(frsettings, fluxr_itr->first, block);
    fluxes.push_back(newflux);
    assembler->addFunction(block, "flux weight " + newflux.name, newflux.weight, "side ip");
    fluxr_itr++;
  }
}

// ========================================================================================
// ========================================================================================

template <class Node>
void PostprocessManager<Node>::computeFluxResponse(vector<vector_RCP> &current_soln, const ScalarT &currenttime)
{

  for (size_t block = 0; block < assembler->groupData.size(); ++block)
  {
    for (size_t grp = 0; grp < assembler->boundary_groups[block].size(); ++grp)
    {
      // setup workset for this bgrp

      assembler->updateWorksetBoundary(block, grp, 0, 0, true);

      // compute the flux
      assembler->wkset[block]->flux = View_Sc3("flux", assembler->wkset[block]->maxElem,
                                               assembler->wkset[block]->numVars[0], // hard coded
                                               assembler->wkset[block]->numsideip);

      physics->computeFlux<ScalarT>(0, block);    // hard coded
      auto cflux = assembler->wkset[block]->flux; // View_AD3

      for (size_t f = 0; f < fluxes.size(); ++f)
      {

        if (fluxes[f].block == block)
        {
          string sidename = assembler->boundary_groups[block][grp]->sidename;
          size_t found = fluxes[f].sidesets.find(sidename);

          if (found != std::string::npos)
          {

            View_Sc2 wts = assembler->evaluateFunction(block, "flux weight " + fluxes[f].name, "side ip");
            auto iwts = assembler->wkset[block]->wts_side;

            for (size_type v = 0; v < fluxes[f].vals.extent(0); ++v)
            {
              ScalarT value = 0.0;
              auto vflux = subview(cflux, ALL(), v, ALL());
              parallel_reduce(RangePolicy<AssemblyExec>(0, iwts.extent(0)), KOKKOS_LAMBDA(const int elem, ScalarT &update) {
                for( size_t pt=0; pt<iwts.extent(1); pt++ ) {
                  ScalarT up = vflux(elem,pt)*wts(elem,pt)*iwts(elem,pt);
                  update += up;
                } }, value);
              fluxes[f].vals(v) += value;
            }
          }
        }
      }
    }
  }
}
