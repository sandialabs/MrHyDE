// @HEADER
// ************************************************************************
//
//                           Intrepid2 Package
//                 Copyright (2007) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Kyungjoo Kim  (kyukim@sandia.gov), or
//                    Mauro Perego  (mperego@sandia.gov)
//
// ************************************************************************
// @HEADER

/** \file   Intrepid2_HDIV_HEX_In_FEMDef.hpp
 \brief  Definition file for FEM basis functions of degree n for HFACE functions on TET cells.
 \author Created by T. Wildey based on implementation by R. Kirby, P. Bochev, D. Ridzal and K. Peterson.
 Kokkorized by Kyungjoo Kim
 */

#ifndef __INTREPID2_HFACE_TET_IN_FEM_DEF_HPP__
#define __INTREPID2_HFACE_TET_IN_FEM_DEF_HPP__

#include "Intrepid2_HGRAD_TET_Cn_FEM_ORTH.hpp"
//#include "Intrepid2_HGRAD_TET_Cn_FEM.hpp"

namespace Intrepid2 {
  
  // -------------------------------------------------------------------------------------
  namespace Impl {
    
    template<EOperator opType>
    template<typename OutputViewType,
    typename inputViewType,
    typename workViewType,
    typename vinvViewType>
    KOKKOS_INLINE_FUNCTION
    void Basis_HFACE_TET_In_FEM::Serial<opType>::getValues(OutputViewType output,
                                                           const inputViewType  input,
                                                           workViewType   work,
                                                           const vinvViewType   vinvTri) {
      const ordinal_type cardTri = vinvTri.extent(0);
      const ordinal_type npts = input.extent(0);
      
      typedef typename Kokkos::DynRankView<typename workViewType::value_type, typename workViewType::memory_space> viewType;
      auto vcprop = Kokkos::common_view_alloc_prop(work);
      auto ptr = work.data();
      
      // compute order
      ordinal_type order = 0;
      for (ordinal_type p=0;p<=Parameters::MaxOrder;++p) {
        if (cardTri == Intrepid2::getPnCardinality<2>(p)) {
          order = p;
          break;
        }
      }
      
      switch (opType) {
        case OPERATOR_VALUE: {
          const viewType phis(Kokkos::view_wrap(ptr, vcprop), cardTri, npts);
          viewType dummyView;
          
          if (order == 0) {
            for (ordinal_type i=0;i<cardTri;++i) {
              for (ordinal_type j=0;j<npts;++j) {
                phis(i,j) = 1.0;
              }
            }
          }
          else {
            Impl::Basis_HGRAD_TRI_Cn_FEM_ORTH::
            Serial<opType>::getValues(phis, input, dummyView, order);
          }
          
          ordinal_type cprog = 0;
          for (ordinal_type i=0;i<4*cardTri;++i) {
            for (ordinal_type j=0;j<npts;++j) {
              output.access(i,j) = 0.0;
            }
          }
          
          for (ordinal_type i=0;i<cardTri;++i) {
            for (ordinal_type j=0;j<npts;++j) {
              //output.access(i,j) = 0.0;
              bool on_edge = false;
              if (std::abs(input(j,1)-0.0)<1.0e-12) {
                on_edge = true;
              }
              if (on_edge) {
                for (ordinal_type k=0;k<cardTri;++k) {
                  output.access(cprog,j) += vinvTri(k,i)*phis.access(k,j);
                }
              }
            }
            cprog++;
          }
          
          for (ordinal_type i=0;i<cardTri;++i) {
            for (ordinal_type j=0;j<npts;++j) {
              //output.access(i,j) = 0.0;
              bool on_edge = false;
              if (std::abs(input(j,0)+input(j,1)+input(j,2) - 1.0)<1.0e-12) {
                on_edge = true;
              }
              if (on_edge) {
                for (ordinal_type k=0;k<cardTri;++k) {
                  output.access(cprog,j) += vinvTri(k,i)*phis.access(k,j);
                }
              }
            }
            cprog++;
          }
          
          for (ordinal_type i=0;i<cardTri;++i) {
            for (ordinal_type j=0;j<npts;++j) {
              //output.access(i,j) = 0.0;
              bool on_edge = false;
              if (std::abs(input(j,0)-0.0)<1.0e-12) {
                on_edge = true;
              }
              if (on_edge) {
                for (ordinal_type k=0;k<cardTri;++k) {
                  output.access(cprog,j) += vinvTri(k,i)*phis.access(k,j);
                }
              }
            }
            cprog++;
          }
          
          for (ordinal_type i=0;i<cardTri;++i) {
            for (ordinal_type j=0;j<npts;++j) {
              //output.access(i,j) = 0.0;
              bool on_edge = false;
              if (std::abs(input(j,2)-0.0)<1.0e-12) {
                on_edge = true;
              }
              if (on_edge) {
                for (ordinal_type k=0;k<cardTri;++k) {
                  output.access(cprog,j) += vinvTri(k,i)*phis.access(k,j);
                }
              }
            }
            cprog++;
          }
          break;
        }
        case OPERATOR_GRAD: {
          // not needed
          break;
        }
        default: {
          INTREPID2_TEST_FOR_ABORT( true,
                                   ">>> ERROR: (Intrepid2::Basis_HFACE_HEX_In_FEM::Serial::getValues) operator is not supported" );
        }
      }
    }
    
    template<typename DT, ordinal_type numPtsPerEval,
    typename outputValueValueType, class ...outputValueProperties,
    typename inputPointValueType,  class ...inputPointProperties,
    typename vinvValueType,        class ...vinvProperties>
    void Basis_HFACE_TET_In_FEM::getValues(Kokkos::DynRankView<outputValueValueType,outputValueProperties...> outputValues,
                                           const Kokkos::DynRankView<inputPointValueType, inputPointProperties...>  inputPoints,
                                           const Kokkos::DynRankView<vinvValueType,       vinvProperties...>        vinvTri,
                                           const EOperator operatorType ) {
      typedef          Kokkos::DynRankView<outputValueValueType,outputValueProperties...>         outputValueViewType;
      typedef          Kokkos::DynRankView<inputPointValueType, inputPointProperties...>          inputPointViewType;
      typedef          Kokkos::DynRankView<vinvValueType,       vinvProperties...>                vinvViewType;
      typedef typename ExecSpace<typename inputPointViewType::execution_space,typename DT::execution_space>::ExecSpaceType ExecSpaceType;
      
      // loopSize corresponds to cardinality
      const auto loopSizeTmp1 = (inputPoints.extent(0)/numPtsPerEval);
      const auto loopSizeTmp2 = (inputPoints.extent(0)%numPtsPerEval != 0);
      const auto loopSize = loopSizeTmp1 + loopSizeTmp2;
      Kokkos::RangePolicy<ExecSpaceType,Kokkos::Schedule<Kokkos::Static> > policy(0, loopSize);
      
      typedef typename inputPointViewType::value_type inputPointType;
      
      const ordinal_type cardinality = outputValues.extent(0);
      //get basis order based on basis cardinality.
      ordinal_type order = 0;   // = std::sqrt(cardinality/2);
      ordinal_type cardTri;  // = cardBubble+1;
      if (cardinality == 4) {
        cardTri = 1;
      }
      else {
        do {
          cardTri = Intrepid2::getPnCardinality<2>(++order);
        } while((4*cardTri !=  cardinality) && (order != Parameters::MaxOrder));
      }
      auto vcprop = Kokkos::common_view_alloc_prop(inputPoints);
      typedef typename Kokkos::DynRankView< inputPointType, typename inputPointViewType::memory_space> workViewType;
      
      switch (operatorType) {
        case OPERATOR_VALUE: {
          auto workSize = 4*Serial<OPERATOR_VALUE>::getWorkSizePerPoint(order);
          workViewType  work(Kokkos::view_alloc("Basis_HFACE_TET_In_FEM::getValues::work", vcprop), workSize, inputPoints.extent(0));
          typedef Functor<outputValueViewType,inputPointViewType,vinvViewType, workViewType,
          OPERATOR_VALUE,numPtsPerEval> FunctorType;
          Kokkos::parallel_for( policy, FunctorType(outputValues, inputPoints, vinvTri, work) );
          break;
        }
        case OPERATOR_GRAD: {
          break;
        }
        default: {
          INTREPID2_TEST_FOR_EXCEPTION( true , std::invalid_argument,
                                       ">>> ERROR (Basis_HFACE_TET_In_FEM): Operator type not implemented" );
        }
      }
    }
  }
  
  // -------------------------------------------------------------------------------------
  template<typename DT, typename OT, typename PT>
  Basis_HFACE_TET_In_FEM<DT,OT,PT>::
  Basis_HFACE_TET_In_FEM( const ordinal_type order,
                         const EPointType   pointType ) {
    
    INTREPID2_TEST_FOR_EXCEPTION( !(pointType == POINTTYPE_EQUISPACED ||
                                    pointType == POINTTYPE_WARPBLEND), std::invalid_argument,
                                 ">>> ERROR (Basis_HFACE_TET_In_FEM): pointType must be either equispaced or warpblend.");
    
    // this should be in host
    ordinal_type cardTri;
    Kokkos::DynRankView<typename ScalarViewType::value_type,DT> dofCoordsTri;
    
    if (order == 0) {
      cardTri = 1;
      this->vinvTri_   = Kokkos::DynRankView<typename ScalarViewType::value_type,DT>("HFACE::TET::In::vinvTri", cardTri, cardTri);
      this->vinvTri_(0,0) = 1.0;
      dofCoordsTri = Kokkos::DynRankView<typename ScalarViewType::value_type,DT>("dofCoordsLine", cardTri, 2);
      dofCoordsTri(0,0) = 1.0/3.0;
      dofCoordsTri(0,1) = 1.0/3.0;
    }
    else {
      Basis_HGRAD_TRI_Cn_FEM<DT,OT,PT> triBasis( order, pointType );
      cardTri = triBasis.getCardinality();
      this->vinvTri_   = Kokkos::DynRankView<typename ScalarViewType::value_type,DT>("HFACE::TET::In::vinvTri", cardTri, cardTri);
      triBasis.getVandermondeInverse(this->vinvTri_);
      triBasis.getDofCoords(dofCoordsTri);
    }
    
    this->basisCardinality_  = 4*cardTri;
    this->basisDegree_       = order;
    this->basisCellTopologyKey_ = shards::Tetrahedron<4>::key; //shards::CellTopology(shards::getCellTopologyData<shards::Tetrahedron<4> >() );
    this->basisType_         = BASIS_FEM_LAGRANGIAN;
    this->basisCoordinates_  = COORDINATES_CARTESIAN;
    this->functionSpace_     = FUNCTION_SPACE_HGRAD;
    
    constexpr ordinal_type spaceDim = 3;
    
    // initialize tags
    {
      // Basis-dependent initializations
      const ordinal_type tagSize  = 4;        // size of DoF tag, i.e., number of fields in the tag
      const ordinal_type posScDim = 0;        // position in the tag, counting from 0, of the subcell dim
      const ordinal_type posScOrd = 1;        // position in the tag, counting from 0, of the subcell ordinal
      const ordinal_type posDfOrd = 2;        // position in the tag, counting from 0, of DoF ordinal relative to the subcell
      
      // An array with local DoF tags assigned to the basis functions, in the order of their local enumeration
      constexpr ordinal_type maxCardTri = Parameters::MaxOrder + 1;
      ordinal_type tags[4*maxCardTri][4];
      
      //const ordinal_type face_yz[2] = {3, 1};
      //const ordinal_type face_xz[2] = {0, 2};
      //const ordinal_type face_xy[2] = {4, 5};
      
      {
        ordinal_type idx = 0;
        
        // y=0 face
        for (ordinal_type j=0;j<cardTri;++j) { // y
          tags[idx][0] = 2; // face dof
          tags[idx][1] = 0;
          tags[idx][2] = j; // local dof id
          tags[idx][3] = cardTri; // total number of dofs in this vertex
          idx++;
        }
        
        // z = 1-x-y
        for (ordinal_type j=0;j<cardTri;++j) { // y
          tags[idx][0] = 2; // edge dof
          tags[idx][1] = 1;
          tags[idx][2] = j; // local dof id
          tags[idx][3] = cardTri; // total number of dofs in this vertex
          idx++;
        }
        
        // x=0 face
        for (ordinal_type j=0;j<cardTri;++j) { // y
          tags[idx][0] = 2; // face dof
          tags[idx][1] = 2;
          tags[idx][2] = j; // local dof id
          tags[idx][3] = cardTri; // total number of dofs in this vertex
          idx++;
        }
        
        // z=0 face
        for (ordinal_type j=0;j<cardTri;++j) { // y
          tags[idx][0] = 2; // edge dof
          tags[idx][1] = 3;
          tags[idx][2] = j; // local dof id
          tags[idx][3] = cardTri; // total number of dofs in this vertex
          idx++;
        }
        
        INTREPID2_TEST_FOR_EXCEPTION( idx != this->basisCardinality_ , std::runtime_error,
                                     ">>> ERROR (Basis_HFACE_TET_In_FEM): " \
                                     "counted tag index is not same as cardinality." );
      }
      
      OrdinalTypeArray1DHost tagView(&tags[0][0], this->basisCardinality_*4);
      
      // Basis-independent function sets tag and enum data in tagToOrdinal_ and ordinalToTag_ arrays:
      // tags are constructed on host
      this->setOrdinalTagData(this->tagToOrdinal_,
                              this->ordinalToTag_,
                              tagView,
                              this->basisCardinality_,
                              tagSize,
                              posScDim,
                              posScOrd,
                              posDfOrd);
    }
    
    // dofCoords on host and create its mirror view to device
    Kokkos::DynRankView<typename ScalarViewType::value_type,typename DT::execution_space::array_layout,Kokkos::HostSpace>
    dofCoordsHost("dofCoordsHost", this->basisCardinality_, spaceDim);
    
    // dofCoeffs on host and create its mirror view to device
    //Kokkos::DynRankView<typename ScalarViewType::value_type,typename SpT::array_layout,Kokkos::HostSpace>
    //dofCoeffsHost("dofCoeffsHost", this->basisCardinality_, this->basisCellTopology_.getDimension());
    
    //Kokkos::DynRankView<typename ScalarViewType::value_type,SpT> dofCoordsTri("dofCoordsLine", cardTri, 2);
    
    //triBasis.getDofCoords(dofCoordsTri);
    auto dofCoordsTriHost = Kokkos::create_mirror_view(Kokkos::HostSpace(), dofCoordsTri);
    Kokkos::deep_copy(dofCoordsTriHost, dofCoordsTri);
    
    {
      ordinal_type idx = 0;
      
      // y=0 face
      for (ordinal_type j=0;j<cardTri;++j) { // x
        dofCoordsHost(idx,0) = dofCoordsTriHost(j,0);
        dofCoordsHost(idx,1) = 0.0;
        dofCoordsHost(idx,2) = dofCoordsTriHost(j,1);
        idx++;
      }
      
      // z=1-x-y face
      for (ordinal_type j=0;j<cardTri;++j) { // x
        dofCoordsHost(idx,0) = dofCoordsTriHost(j,0);
        dofCoordsHost(idx,1) = dofCoordsTriHost(j,1);
        dofCoordsHost(idx,2) = 1.0 - dofCoordsTriHost(j,0) - dofCoordsTriHost(j,1);
        idx++;
      }
      
      // x=0 face
      for (ordinal_type j=0;j<cardTri;++j) { // y
        dofCoordsHost(idx,0) = 0.0;
        dofCoordsHost(idx,1) = dofCoordsTriHost(j,0);
        dofCoordsHost(idx,2) = dofCoordsTriHost(j,1);
        idx++;
      }
      
      
      
      // z=0 face
      for (ordinal_type j=0;j<cardTri;++j) { // y
        dofCoordsHost(idx,0) = dofCoordsTriHost(j,0);
        dofCoordsHost(idx,1) = dofCoordsTriHost(j,1);
        dofCoordsHost(idx,2) = 0.0;
        idx++;
      }
      
      
      
    }
    
    this->dofCoords_ = Kokkos::create_mirror_view(typename DT::memory_space(), dofCoordsHost);
    Kokkos::deep_copy(this->dofCoords_, dofCoordsHost);
    
    //this->dofCoeffs_ = Kokkos::create_mirror_view(typename SpT::memory_space(), dofCoeffsHost);
    //Kokkos::deep_copy(this->dofCoeffs_, dofCoeffsHost);
  }
  
}

#endif
