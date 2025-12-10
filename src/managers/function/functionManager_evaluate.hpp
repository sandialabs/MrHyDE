/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

//////////////////////////////////////////////////////////////////////////////////////
// Evaluate a function
//////////////////////////////////////////////////////////////////////////////////////

template<class EvalT>
Vista<EvalT> FunctionManager<EvalT>::evaluate(const string & fname, const string & location) {
  
  bool ffound = false, tfound = false;
  size_t fiter=0, titer=0;
  while(!ffound && fiter<forests_.size()) {
    if (forests_[fiter].location_ == location) {
      ffound = true;
      tfound = false;
      while (!tfound && titer<forests_[fiter].trees_.size()) {
        if (fname == forests_[fiter].trees_[titer].name_) {
          tfound = true;
          if (!forests_[fiter].trees_[titer].branches_[0].is_decomposed_) {
            this->decomposeFunctions();
          }
          if (!forests_[fiter].trees_[titer].branches_[0].is_constant_) {
            this->evaluate(fiter,titer,0);
          }
        }
        else {
          titer++;
        }
      }
    }
    else {
      fiter++;
    }
  }
  
  if (!ffound || !tfound) { // meaning that the requested function was not registered at this location
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: function manager could not evaluate: " + fname + " at " + location);
  }
  
  if (!forests_[fiter].trees_[titer].branches_[0].is_constant_) {
    forests_[fiter].trees_[titer].updateVista();
  }
  
  return forests_[fiter].trees_[titer].vista_;
  
}

//////////////////////////////////////////////////////////////////////////////////////
// Evaluate a function
//////////////////////////////////////////////////////////////////////////////////////

template<class EvalT>
void FunctionManager<EvalT>::evaluate( const size_t & findex, const size_t & tindex, const size_t & bindex) {
  
  //if (!forests_[findex].trees_[tindex].branches_[bindex].isConstant) {
    if (forests_[findex].trees_[tindex].branches_[bindex].is_leaf_) {
      if (forests_[findex].trees_[tindex].branches_[bindex].is_workset_data_) {
        int wdindex = forests_[findex].trees_[tindex].branches_[bindex].workset_data_index_;
        if (wkset->isOnSide) {
          if (forests_[findex].trees_[tindex].branches_[bindex].is_AD_) {
            if (!wkset->side_soln_fields[wdindex].is_updated_) {
              wkset->evaluateSideSolutionField(wdindex);
            }
            forests_[findex].trees_[tindex].branches_[bindex].viewdata_ = wkset->side_soln_fields[wdindex].data_;
          }
          else {
            forests_[findex].trees_[tindex].branches_[bindex].viewdata_Sc_ = wkset->side_scalar_fields[wdindex].data_;
          } 
        }
        else if (wkset->isOnPoint) {
          if (forests_[findex].trees_[tindex].branches_[bindex].is_AD_) {
            if (!wkset->point_soln_fields[wdindex].is_updated_) {
              wkset->evaluateSolutionField(wdindex);
            }
            forests_[findex].trees_[tindex].branches_[bindex].viewdata_ = wkset->point_soln_fields[wdindex].data_;
          }
          else {
            forests_[findex].trees_[tindex].branches_[bindex].viewdata_Sc_ = wkset->point_scalar_fields[wdindex].data_;
          }
        }
        else {
          if (forests_[findex].trees_[tindex].branches_[bindex].is_AD_) {
            if (!wkset->soln_fields[wdindex].is_updated_) {
              wkset->evaluateSolutionField(wdindex);
            }
            forests_[findex].trees_[tindex].branches_[bindex].viewdata_ = wkset->soln_fields[wdindex].data_;
          }
          else {
            forests_[findex].trees_[tindex].branches_[bindex].viewdata_Sc_ = wkset->scalar_fields[wdindex].data_;
          }
        }
      }
      else if (forests_[findex].trees_[tindex].branches_[bindex].is_parameter_) {
        // Should be set correctly already
      }
      else if (forests_[findex].trees_[tindex].branches_[bindex].is_time_) {
        forests_[findex].trees_[tindex].branches_[bindex].data_Sc_ = wkset->time;
      }
    }
    else if (forests_[findex].trees_[tindex].branches_[bindex].is_func_) {
      int funcIndex = forests_[findex].trees_[tindex].branches_[bindex].func_index_;
      this->evaluate(findex,funcIndex, 0);
      if (forests_[findex].trees_[tindex].branches_[bindex].is_AD_) {
        if (forests_[findex].trees_[tindex].branches_[bindex].is_view_) { // use viewdata
          forests_[findex].trees_[tindex].branches_[bindex].viewdata_ = forests_[findex].trees_[funcIndex].branches_[0].viewdata_;
        }
        else { // use data
          forests_[findex].trees_[tindex].branches_[bindex].data_ = forests_[findex].trees_[funcIndex].branches_[0].data_;
        }
      }
      else {
        if (forests_[findex].trees_[tindex].branches_[bindex].is_view_) { // use viewdata_Sc
          forests_[findex].trees_[tindex].branches_[bindex].viewdata_Sc_ = forests_[findex].trees_[funcIndex].branches_[0].viewdata_Sc_;
        }
        else { // use data_Sc
          forests_[findex].trees_[tindex].branches_[bindex].data_Sc_ = forests_[findex].trees_[funcIndex].branches_[0].data_Sc_;
        }
      }
    }
    else {
      bool isAD = forests_[findex].trees_[tindex].branches_[bindex].is_AD_;
      bool isView = forests_[findex].trees_[tindex].branches_[bindex].is_view_;
      for (size_t k=0; k<forests_[findex].trees_[tindex].branches_[bindex].dep_list_.size(); k++) {
        
        int dep = forests_[findex].trees_[tindex].branches_[bindex].dep_list_[k];
        this->evaluate(findex, tindex, dep);
        
        bool termisAD = forests_[findex].trees_[tindex].branches_[dep].is_AD_;
        bool termisView = forests_[findex].trees_[tindex].branches_[dep].is_view_;
        bool termisParameter = forests_[findex].trees_[tindex].branches_[dep].is_parameter_;
        if (isView) {
          if (termisView) {
            if (isAD) {
              if (termisAD) {
                if (termisParameter) {
                  this->evaluateOpParamToV(forests_[findex].trees_[tindex].branches_[bindex].viewdata_,
                                           forests_[findex].trees_[tindex].branches_[dep].param_data_,
                                           forests_[findex].trees_[tindex].branches_[dep].param_index_,
                                           forests_[findex].trees_[tindex].branches_[bindex].dep_ops_[k]);
                }
                else {
                  this->evaluateOpVToV(forests_[findex].trees_[tindex].branches_[bindex].viewdata_,
                                       forests_[findex].trees_[tindex].branches_[dep].viewdata_,
                                       forests_[findex].trees_[tindex].branches_[bindex].dep_ops_[k]);
                }
                
              }
              else {
                this->evaluateOpVToV(forests_[findex].trees_[tindex].branches_[bindex].viewdata_,
                                     forests_[findex].trees_[tindex].branches_[dep].viewdata_Sc_,
                                     forests_[findex].trees_[tindex].branches_[bindex].dep_ops_[k]);
              }
            }
            else {
              if (termisAD) {
                // output error
              }
              else {
                this->evaluateOpVToV(forests_[findex].trees_[tindex].branches_[bindex].viewdata_Sc_,
                                     forests_[findex].trees_[tindex].branches_[dep].viewdata_Sc_,
                                     forests_[findex].trees_[tindex].branches_[bindex].dep_ops_[k]);
              }
            }
          }
          else { // Scalar data
            if (isAD) {
              if (termisAD) {
                this->evaluateOpSToV(forests_[findex].trees_[tindex].branches_[bindex].viewdata_,
                                     forests_[findex].trees_[tindex].branches_[dep].data_,
                                     forests_[findex].trees_[tindex].branches_[bindex].dep_ops_[k]);
              }
              else {
                this->evaluateOpSToV(forests_[findex].trees_[tindex].branches_[bindex].viewdata_,
                                     forests_[findex].trees_[tindex].branches_[dep].data_Sc_,
                                     forests_[findex].trees_[tindex].branches_[bindex].dep_ops_[k]);
              }
            }
            else {
              if (termisAD) {
                //error
              }
              else {
                this->evaluateOpSToV(forests_[findex].trees_[tindex].branches_[bindex].viewdata_Sc_,
                                     forests_[findex].trees_[tindex].branches_[dep].data_Sc_,
                                     forests_[findex].trees_[tindex].branches_[bindex].dep_ops_[k]);
              }
            }
            
          }
        }
        else {
          if (termisView) {
            //error
          }
          else {
            if (isAD) {
              if (termisAD) {
                this->evaluateOpSToS(forests_[findex].trees_[tindex].branches_[bindex].data_,
                                     forests_[findex].trees_[tindex].branches_[dep].data_,
                                     forests_[findex].trees_[tindex].branches_[bindex].dep_ops_[k]);
              }
              else {
                this->evaluateOpSToS(forests_[findex].trees_[tindex].branches_[bindex].data_,
                                     forests_[findex].trees_[tindex].branches_[dep].data_Sc_,
                                     forests_[findex].trees_[tindex].branches_[bindex].dep_ops_[k]);
              }
            }
            else {
              if (termisAD) {
                //error
              }
              else {
                this->evaluateOpSToS(forests_[findex].trees_[tindex].branches_[bindex].data_Sc_,
                                     forests_[findex].trees_[tindex].branches_[dep].data_Sc_,
                                     forests_[findex].trees_[tindex].branches_[bindex].dep_ops_[k]);
              }
            }
          }
        }
      }
    }
 // }
}

//////////////////////////////////////////////////////////////////////////////////////
// Evaluate an operator
//////////////////////////////////////////////////////////////////////////////////////

template<class EvalT> 
template<class T1, class T2>
void FunctionManager<EvalT>::evaluateOpVToV(T1 data, T2 tdata, const string & op) {
  
  size_t dim0 = std::min(data.extent(0),tdata.extent(0));
  using namespace std;
  
  if (op == "") {
    parallel_for("funcman evaluate equals",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = tdata(elem,pt);
      }
    });
  }
  else if (op == "plus") {
    parallel_for("funcman evaluate plus",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) += tdata(elem,pt);
      }
    });
  }
  else if (op == "minus") {
    parallel_for("funcman evaluate minus",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) += -tdata(elem,pt);
      }
    });
  }
  else if (op == "times") {
    parallel_for("funcman evaluate times",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) *= tdata(elem,pt);
      }
    });
  }
  else if (op == "divide") {
    parallel_for("funcman evaluate divide",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) /= tdata(elem,pt);
      }
    });
  }
  else if (op == "power") {
    parallel_for("funcman evaluate power",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = pow(data(elem,pt),tdata(elem,pt));
      }
    });
  }
  else if (op == "sin") {
    parallel_for("funcman evaluate sin",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = sin(tdata(elem,pt));
      }
    });
  }
  else if (op == "cos") {
    parallel_for("funcman evaluate cos",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = cos(tdata(elem,pt));
      }
    });
  }
  else if (op == "tan") {
    parallel_for("funcman evaluate tan",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = tan(tdata(elem,pt));
      }
    });
  }
  else if (op == "exp") {
    parallel_for("funcman evaluate exp",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = exp(tdata(elem,pt));
      }
    });
  }
  else if (op == "log") {
    parallel_for("funcman evaluate log",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = log(tdata(elem,pt));
      }
    });
  }
  else if (op == "abs") {
    parallel_for("funcman evaluate abs",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        if (tdata(elem,pt) < 0.0) {
          data(elem,pt) = -tdata(elem,pt);
        }
        else {
          data(elem,pt) = tdata(elem,pt);
        }
      }
    });
  }
  else if (op == "max") {
    parallel_for("funcman evaluate max",
                 RangePolicy<AssemblyExec>(0,dim0),
                 KOKKOS_CLASS_LAMBDA (const int e ) {
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (unsigned int n=0; n<dim1; n++) {
        if (tdata(e,n) > data(e,n)) {
          data(e,n) = tdata(e,n);
        }
      }
    });
  }
  else if (op == "min") {
    parallel_for("funcman evaluate min",
                 RangePolicy<AssemblyExec>(0,dim0),
                 KOKKOS_CLASS_LAMBDA (const int e ) {
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (unsigned int n=0; n<dim1; n++) {
        if (tdata(e,n) < data(e,n)) {
          data(e,n) = tdata(e,n);
        }
      }
    });
  }
  else if (op == "mean") {
    parallel_for("funcman evaluate mean",
                 RangePolicy<AssemblyExec>(0,dim0),
                 KOKKOS_CLASS_LAMBDA (const int e ) {
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (unsigned int n=0; n<dim1; n++) {
        data(e,n) = 0.5*data(e,n) + 0.5*tdata(e,n);
      }
    });
  }
  else if (op == "emax") { // maximum over rows ... usually corr. to max over element/face at ip
    parallel_for("funcman evaluate emax",
                 RangePolicy<AssemblyExec>(0,dim0),
                 KOKKOS_CLASS_LAMBDA (const int e ) {
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      data(e,0) = tdata(e,0);
      for (unsigned int n=0; n<dim1; n++) {
        if (tdata(e,n) > tdata(e,0)) {
          data(e,0) = tdata(e,n);
        }
      }
      for (unsigned int n=0; n<dim1; n++) { // copy max value at all ip
        data(e,n) = data(e,0);
      }
    });
  }
  else if (op == "emin") { // minimum over rows ... usually corr. to min over element/face at ip
    parallel_for("funcman evaluate emin",
                 RangePolicy<AssemblyExec>(0,dim0),
                 KOKKOS_CLASS_LAMBDA (const int e ) {
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      data(e,0) = tdata(e,0);
      for (unsigned int n=0; n<dim1; n++) {
        if (tdata(e,n) < tdata(e,0)) {
          data(e,0) = tdata(e,n);
        }
      }
      for (unsigned int n=0; n<dim1; n++) { // copy min value at all ip
        data(e,n) = data(e,0);
      }
    });
  }
  else if (op == "emean") { // mean over rows ... usually corr. to mean over element/face
    parallel_for("funcman evaluate emean",
                 RangePolicy<AssemblyExec>(0,dim0),
                 KOKKOS_CLASS_LAMBDA (const int e ) {
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      ScalarT scale = (ScalarT)dim1;
      data(e,0) = tdata(e,0)/scale;
      for (unsigned int n=0; n<dim1; n++) {
        data(e,0) += tdata(e,n)/scale;
      }
      for (unsigned int n=0; n<dim1; n++) { // copy max value at all ip
        data(e,n) = data(e,0);
      }
    });
  }
  else if (op == "lt") {
    parallel_for("funcman evaluate lt",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        if (data(elem,pt) < tdata(elem,pt)) {
          data(elem,pt) = 1.0;
        }
        else {
          data(elem,pt) = 0.0;
        }
      }
    });
  }
  else if (op == "lte") {
    parallel_for("funcman evaluate lte",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        if (data(elem,pt) <= tdata(elem,pt)) {
          data(elem,pt) = 1.0;
        }
        else {
          data(elem,pt) = 0.0;
        }
      }
    });
  }
  else if (op == "gt") {
    parallel_for("funcman evaluate gt",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        if (data(elem,pt) > tdata(elem,pt)) {
          data(elem,pt) = 1.0;
        }
        else {
          data(elem,pt) = 0.0;
        }
      }
    });
  }
  else if (op == "gte") {
    parallel_for("funcman evaluate gte",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        if (data(elem,pt) >= tdata(elem,pt)) {
          data(elem,pt) = 1.0;
        }
        else {
          data(elem,pt) = 0.0;
        }
      }
    });
  }
  else if (op == "sqrt") {
    parallel_for("funcman evaluate sqrt",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        if (tdata(elem,pt) <= 0.0) {
          data(elem,pt) = 0.0;
        }
        else {
          data(elem,pt) = sqrt(tdata(elem,pt));
        }
      }
    });
  }
  else if (op == "sinh") {
    parallel_for("funcman evaluate sinh",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = sinh(tdata(elem,pt));
      }
    });
  }
  else if (op == "cosh") {
    parallel_for("funcman evaluate cosh",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = cosh(tdata(elem,pt));
      }
    });
  }
}

//////////////////////////////////////////////////////////////////////////////////////
// Evaluate an operator
//////////////////////////////////////////////////////////////////////////////////////

template<class EvalT>
template<class T1, class T2>
void FunctionManager<EvalT>::evaluateOpParamToV(T1 data, T2 tdata, const int & pIndex_, const string & op) {
  
  size_t dim0 = data.extent(0);
  using namespace std;
  
  int pIndex = pIndex_;
  
  if (op == "") {
    parallel_for("funcman evaluate equals",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = tdata(pIndex);
      }
    });
  }
  else if (op == "plus") {
    parallel_for("funcman evaluate plus",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) += tdata(pIndex);
      }
    });
  }
  else if (op == "minus") {
    parallel_for("funcman evaluate minus",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) += -tdata(pIndex);
      }
    });
  }
  else if (op == "times") {
    parallel_for("funcman evaluate times",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) *= tdata(pIndex);
      }
    });
  }
  else if (op == "divide") {
    parallel_for("funcman evaluate divide",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) /= tdata(pIndex);
      }
    });
  }
  else if (op == "power") {
    parallel_for("funcman evaluate power",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = pow(data(elem,pt),tdata(pIndex));
      }
    });
  }
  else if (op == "sin") {
    parallel_for("funcman evaluate sin",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = sin(tdata(pIndex));
      }
    });
  }
  else if (op == "cos") {
    parallel_for("funcman evaluate cos",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = cos(tdata(pIndex));
      }
    });
  }
  else if (op == "sinh") {
    parallel_for("funcman evaluate sinh",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = sinh(tdata(pIndex));
      }
    });
  }
  else if (op == "cosh") {
    parallel_for("funcman evaluate cosh",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = cosh(tdata(pIndex));
      }
    });
  }
  else if (op == "tan") {
    parallel_for("funcman evaluate tan",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = tan(tdata(pIndex));
      }
    });
  }
  else if (op == "exp") {
    parallel_for("funcman evaluate exp",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = exp(tdata(pIndex));
      }
    });
  }
  else if (op == "log") {
    parallel_for("funcman evaluate log",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = log(tdata(pIndex));
      }
    });
  }
  else if (op == "sqrt") {
    parallel_for("funcman evaluate sqrt",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        if(tdata(pIndex) <= 0.) {
          data(elem,pt) = 0.;
        } else {
          data(elem,pt) = sqrt(tdata(pIndex));
        }
      }
    });
  }
  else if (op == "abs") {
    parallel_for("funcman evaluate abs",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        if (tdata(pIndex) < 0.0) {
          data(elem,pt) = -tdata(pIndex);
        }
        else {
          data(elem,pt) = tdata(pIndex);
        }
      }
    });
  }
  else if (op == "max") {
    parallel_for("funcman evaluate max",
                 RangePolicy<AssemblyExec>(0,dim0),
                 KOKKOS_CLASS_LAMBDA (const int e ) {
      size_t dim1 = data.extent(1);
      for (unsigned int n=0; n<dim1; n++) {
        if (tdata(pIndex) > data(e,n)) {
          data(e,n) = tdata(pIndex);
        }
      }
    });
  }
  else if (op == "min") {
    parallel_for("funcman evaluate min",
                 RangePolicy<AssemblyExec>(0,dim0),
                 KOKKOS_CLASS_LAMBDA (const int e ) {
      size_t dim1 = data.extent(1);
      for (unsigned int n=0; n<dim1; n++) { // copy min value at all ip
        if (tdata(pIndex) < data(e,n)) {
          data(e,n) = tdata(pIndex);
        }
      }
    });
  }
  else if (op == "mean") {
    parallel_for("funcman evaluate mean",
                 RangePolicy<AssemblyExec>(0,dim0),
                 KOKKOS_CLASS_LAMBDA (const int e ) {
      size_t dim1 = data.extent(1);
      for (unsigned int n=0; n<dim1; n++) { // copy max value at all ip
        data(e,n) = 0.5*data(e,n) + 0.5*tdata(pIndex);
      }
    });
  }
  else if (op == "emax") { // maximum over rows ... usually corr. to max over element/face at ip
    parallel_for("funcman evaluate emax",
                 RangePolicy<AssemblyExec>(0,dim0),
                 KOKKOS_CLASS_LAMBDA (const int e ) {
      size_t dim1 = data.extent(1);
      data(e,0) = tdata(pIndex);
      for (unsigned int n=0; n<dim1; n++) { // copy max value at all ip
        data(e,n) = data(e,0);
      }
    });
  }
  else if (op == "emin") { // minimum over rows ... usually corr. to min over element/face at ip
    parallel_for("funcman evaluate emin",
                 RangePolicy<AssemblyExec>(0,dim0),
                 KOKKOS_CLASS_LAMBDA (const int e ) {
      size_t dim1 = data.extent(1);
      data(e,0) = tdata(pIndex);
      for (unsigned int n=0; n<dim1; n++) { // copy min value at all ip
        data(e,n) = data(e,0);
      }
    });
  }
  else if (op == "emean") { // mean over rows ... usually corr. to mean over element/face
    parallel_for("funcman evaluate emean",
                 RangePolicy<AssemblyExec>(0,dim0),
                 KOKKOS_CLASS_LAMBDA (const int e ) {
      size_t dim1 = data.extent(1);
      data(e,0) = tdata(pIndex);
      for (unsigned int n=0; n<dim1; n++) { // copy max value at all ip
        data(e,n) = data(e,0);
      }
    });
  }
  else if (op == "lt") {
    parallel_for("funcman evaluate lt",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        if (data(elem,pt) < tdata(pIndex)) {
          data(elem,pt) = 1.0;
        }
        else {
          data(elem,pt) = 0.0;
        }
      }
    });
  }
  else if (op == "lte") {
    parallel_for("funcman evaluate lte",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        if (data(elem,pt) <= tdata(pIndex)) {
          data(elem,pt) = 1.0;
        }
        else {
          data(elem,pt) = 0.0;
        }
      }
    });
  }
  else if (op == "gt") {
    parallel_for("funcman evaluate gt",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        if (data(elem,pt) > tdata(pIndex)) {
          data(elem,pt) = 1.0;
        }
        else {
          data(elem,pt) = 0.0;
        }
      }
    });
  }
  else if (op == "gte") {
    parallel_for("funcman evaluate gte",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        if (data(elem,pt) >= tdata(pIndex)) {
          data(elem,pt) = 1.0;
        }
        else {
          data(elem,pt) = 0.0;
        }
      }
    });
  }
  
}

//////////////////////////////////////////////////////////////////////////////////////
// Evaluate an operator
//////////////////////////////////////////////////////////////////////////////////////

template<class EvalT>
template<class T1, class T2>
void FunctionManager<EvalT>::evaluateOpSToV(T1 data, T2 & tdata_, const string & op) {
  
  T2 tdata = tdata_; // Probably don't need to do this if pass by value
  size_t dim0 = data.extent(0);
  using namespace std;
  
  if (op == "") {
    parallel_for("funcman evaluate equals",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = tdata;
      }
    });
  }
  else if (op == "plus") {
    parallel_for("funcman evaluate plus",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) += tdata;
      }
    });
  }
  else if (op == "minus") {
    parallel_for("funcman evaluate minus",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) += -tdata;
      }
    });
  }
  else if (op == "times") {
    parallel_for("funcman evaluate times",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) *= tdata;
      }
    });
  }
  else if (op == "divide") {
    parallel_for("funcman evaluate divide",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) /= tdata;
      }
    });
  }
  else if (op == "power") {
    parallel_for("funcman evaluate power",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = pow(data(elem,pt),tdata);
      }
    });
  }
  else if (op == "sin") {
    parallel_for("funcman evaluate sin",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = sin(tdata);
      }
    });
  }
  else if (op == "cos") {
    parallel_for("funcman evaluate cos",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = cos(tdata);
      }
    });
  }
  else if (op == "tan") {
    parallel_for("funcman evaluate tan",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = tan(tdata);
      }
    });
  }
  else if (op == "sinh") {
    parallel_for("funcman evaluate sinh",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = sinh(tdata);
      }
    });
  }
  else if (op == "cosh") {
    parallel_for("funcman evaluate cosh",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = cosh(tdata);
      }
    });
  }
  else if (op == "exp") {
    parallel_for("funcman evaluate exp",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = exp(tdata);
      }
    });
  }
  else if (op == "log") {
    parallel_for("funcman evaluate log",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = log(tdata);
      }
    });
  }
  else if (op == "sqrt") {
    parallel_for("funcman evaluate sqrt",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        if(tdata < 0) {
          data(elem,pt) = 0.0;
        } else {
          data(elem,pt) = sqrt(tdata);
        }
      }
    });
  }
  else if (op == "abs") {
    parallel_for("funcman evaluate abs",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        if (tdata < 0.0) {
          data(elem,pt) = -tdata;
        }
        else {
          data(elem,pt) = tdata;
        }
      }
    });
  }
  else if (op == "max") {
    parallel_for("funcman evaluate max",
                 RangePolicy<AssemblyExec>(0,dim0),
                 KOKKOS_CLASS_LAMBDA (const int e ) {
      size_t dim1 = data.extent(1);
      for (unsigned int n=0; n<dim1; n++) {
        if (tdata > data(e,n)) {
          data(e,n) = tdata;
        }
      }
    });
  }
  else if (op == "min") {
    parallel_for("funcman evaluate min",
                 RangePolicy<AssemblyExec>(0,dim0),
                 KOKKOS_CLASS_LAMBDA (const int e ) {
      size_t dim1 = data.extent(1);
      for (unsigned int n=0; n<dim1; n++) { // copy min value at all ip
        if (tdata < data(e,n)) {
          data(e,n) = tdata;
        }}
    });
  }
  else if (op == "mean") {
    parallel_for("funcman evaluate mean",
                 RangePolicy<AssemblyExec>(0,dim0),
                 KOKKOS_CLASS_LAMBDA (const int e ) {
      size_t dim1 = data.extent(1);
      for (unsigned int n=0; n<dim1; n++) { // copy max value at all ip
        data(e,n) = 0.5*data(e,n) + 0.5*tdata;
      }
    });
  }
  else if (op == "emax") { // maximum over rows ... usually corr. to max over element/face at ip
    parallel_for("funcman evaluate emax",
                 RangePolicy<AssemblyExec>(0,dim0),
                 KOKKOS_CLASS_LAMBDA (const int e ) {
      size_t dim1 = data.extent(1);
      data(e,0) = tdata;
      for (unsigned int n=0; n<dim1; n++) { // copy max value at all ip
        data(e,n) = data(e,0);
      }
    });
  }
  else if (op == "emin") { // minimum over rows ... usually corr. to min over element/face at ip
    parallel_for("funcman evaluate emin",
                 RangePolicy<AssemblyExec>(0,dim0),
                 KOKKOS_CLASS_LAMBDA (const int e ) {
      size_t dim1 = data.extent(1);
      data(e,0) = tdata;
      for (unsigned int n=0; n<dim1; n++) { // copy min value at all ip
        data(e,n) = data(e,0);
      }
    });
  }
  else if (op == "emean") { // mean over rows ... usually corr. to mean over element/face
    parallel_for("funcman evaluate emean",
                 RangePolicy<AssemblyExec>(0,dim0),
                 KOKKOS_CLASS_LAMBDA (const int e ) {
      size_t dim1 = data.extent(1);
      data(e,0) = tdata;
      for (unsigned int n=0; n<dim1; n++) { // copy max value at all ip
        data(e,n) = data(e,0);
      }
    });
  }
  else if (op == "lt") {
    parallel_for("funcman evaluate lt",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        if (data(elem,pt) < tdata) {
          data(elem,pt) = 1.0;
        }
        else {
          data(elem,pt) = 0.0;
        }
      }
    });
  }
  else if (op == "lte") {
    parallel_for("funcman evaluate lte",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        if (data(elem,pt) <= tdata) {
          data(elem,pt) = 1.0;
        }
        else {
          data(elem,pt) = 0.0;
        }
      }
    });
  }
  else if (op == "gt") {
    parallel_for("funcman evaluate gt",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        if (data(elem,pt) > tdata) {
          data(elem,pt) = 1.0;
        }
        else {
          data(elem,pt) = 0.0;
        }
      }
    });
  }
  else if (op == "gte") {
    parallel_for("funcman evaluate gte",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        if (data(elem,pt) >= tdata) {
          data(elem,pt) = 1.0;
        }
        else {
          data(elem,pt) = 0.0;
        }
      }
    });
  }
  
}

//////////////////////////////////////////////////////////////////////////////////////
// Evaluate an operator
//////////////////////////////////////////////////////////////////////////////////////

template<class EvalT>
template<class T1, class T2>
void FunctionManager<EvalT>::evaluateOpSToS(T1 & data, T2 & tdata, const string & op) {
  
  using namespace std;
  
  if (op == "") {
    data = tdata;
  }
  else if (op == "plus") {
    data += tdata;
  }
  else if (op == "minus") {
    data += -tdata;
  }
  else if (op == "times") {
    data *= tdata;
  }
  else if (op == "divide") {
    data /= tdata;
  }
  else if (op == "power") {
    data = pow(data,tdata);
  }
  else if (op == "sin") {
    data = sin(tdata);
  }
  else if (op == "cos") {
    data = cos(tdata);
  }
  else if (op == "tan") {
    data = tan(tdata);
  }
  else if (op == "sinh") {
    data = sinh(tdata);
  }
  else if (op == "cosh") {
    data = cosh(tdata);
  }
  else if (op == "exp") {
    data = exp(tdata);
  }
  else if (op == "log") {
    data = log(tdata);
  }
  else if (op == "sqrt") {
    if (tdata <= 0.0) {
      data = 0.0;
    }
    else {
      data = sqrt(tdata);
    }
  }
  else if (op == "abs") {
    if (tdata < 0.0) {
      data = -tdata;
    }
    else {
      data = tdata;
    }
  }
  else if (op == "max") {
    data = max(data,tdata);
  }
  else if (op == "min") {
    data = min(data,tdata);
  }
  else if (op == "mean") {
    data = 0.5*data+0.5*tdata;
  }
  else if (op == "emax") { // maximum over rows ... usually corr. to max over element/face at ip
    data = tdata;
  }
  else if (op == "emin") { // minimum over rows ... usually corr. to min over element/face at ip
    data = tdata;
  }
  else if (op == "emean") { // mean over rows ... usually corr. to mean over element/face
    data = tdata;
  }
  else if (op == "lt") {
    if (data < tdata) {
      data = 1.0;
    }
    else {
      data = 0.0;
    }
  }
  else if (op == "lte") {
    if (data <= tdata) {
      data = 1.0;
    }
    else {
      data = 0.0;
    }
  }
  else if (op == "gt") {
    if (data > tdata) {
      data = 1.0;
    }
    else {
      data = 0.0;
    }
  }
  else if (op == "gte") { 
    if (data >= tdata) {
      data = 1.0;
    }
    else {
      data = 0.0;
    }
  }
  
}
