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
 \brief  Definition file for FEM basis functions of degree n for HFACE functions on HEX cells.
 \author Created by T. Wildey based on implementation by R. Kirby, P. Bochev, D. Ridzal and K. Peterson.
 Kokkorized by Kyungjoo Kim
 */

#ifndef __INTREPID2_HFACE_HEX_IN_FEM_DEF_HPP__
#define __INTREPID2_HFACE_HEX_IN_FEM_DEF_HPP__

namespace Intrepid2 {
  
  // -------------------------------------------------------------------------------------
  namespace Impl {
    
    template<EOperator opType>
    template<typename OutputViewType,
    typename inputViewType,
    typename workViewType,
    typename vinvViewType>
    KOKKOS_INLINE_FUNCTION
    void Basis_HFACE_HEX_In_FEM::Serial<opType>::getValues(OutputViewType output,
                                                           const inputViewType  input,
                                                           workViewType   work,
                                                           const vinvViewType   vinvLine) {
      const ordinal_type cardLine = vinvLine.extent(0);
      
      const ordinal_type npts = input.extent(0);
      
      typedef Kokkos::pair<ordinal_type,ordinal_type> range_type;
      const auto input_x = Kokkos::subview(input, Kokkos::ALL(), range_type(0,1));
      const auto input_y = Kokkos::subview(input, Kokkos::ALL(), range_type(1,2));
      const auto input_z = Kokkos::subview(input, Kokkos::ALL(), range_type(2,3));
      
      const int dim_s = get_dimension_scalar(work);
      auto ptr0 = work.data();
      auto ptr1 = work.data()+cardLine*npts*dim_s;
      auto ptr2 = work.data()+2*cardLine*npts*dim_s;
      auto ptr3 = work.data()+3*cardLine*npts*dim_s;
      
      
      typedef typename Kokkos::DynRankView<typename workViewType::value_type, typename workViewType::memory_space> viewType;
      auto vcprop = Kokkos::common_view_alloc_prop(work);
      
      switch (opType) {
        case OPERATOR_VALUE: {
          viewType workLine(Kokkos::view_wrap(ptr0, vcprop), cardLine, npts);
          viewType output_x(Kokkos::view_wrap(ptr1, vcprop), cardLine, npts);
          viewType output_y(Kokkos::view_wrap(ptr2, vcprop), cardLine, npts);
          viewType output_z(Kokkos::view_wrap(ptr3, vcprop), cardLine, npts);
          
          Impl::Basis_HGRAD_LINE_Cn_FEM::Serial<OPERATOR_VALUE>::getValues(output_x, input_x, workLine, vinvLine);
          Impl::Basis_HGRAD_LINE_Cn_FEM::Serial<OPERATOR_VALUE>::getValues(output_y, input_y, workLine, vinvLine);
          Impl::Basis_HGRAD_LINE_Cn_FEM::Serial<OPERATOR_VALUE>::getValues(output_z, input_z, workLine, vinvLine);
          
          ordinal_type idx = 0;
          // left side (x=-1) first
          {
            bool on_edge = false;
            if (std::abs(input_x(0)+1.0)<1.0e-12) {
              on_edge = true;
            }
            if (on_edge) {
              
              for (ordinal_type j=0;j<cardLine;++j) {
                for (ordinal_type i=0;i<cardLine;++i) {
                  for (ordinal_type k=0;k<npts;++k) {
                    output.access(idx,k) = output_y.access(i,k)*output_z.access(j,k);
                  }
                  idx++;
                }
              }
            }
            else {
              for (ordinal_type j=0;j<cardLine;++j) {
                for (ordinal_type i=0;i<cardLine;++i) {
                  for (ordinal_type k=0;k<npts;++k) {
                    output.access(idx,k) = 0.0;
                  }
                  idx++;
                }
              }
            }
          }
          
          // bottom side (y=-1)
          {
            bool on_edge = false;
            if (std::abs(input_y(0)+1.0)<1.0e-12) {
              on_edge = true;
            }
            
            if (on_edge) {
              for (ordinal_type j=0;j<cardLine;++j) {
                for (ordinal_type i=0;i<cardLine;++i) {
                  for (ordinal_type k=0;k<npts;++k) {
                    output.access(idx,k) = output_x.access(i,k)*output_z.access(j,k);
                  }
                  idx++;
                }
              }
            }
            else {
              for (ordinal_type j=0;j<cardLine;++j) { // y
                for (ordinal_type i=0;i<cardLine;++i) { // y
                  for (ordinal_type k=0;k<npts;++k) {
                    output.access(idx,k) = 0.0;
                  }
                  idx++;
                }
              }
            }
          }
          
          // right side (x=1)
          {
            bool on_edge = false;
            if (std::abs(input_x(0)-1.0)<1.0e-12) {
              on_edge = true;
            }
            if (on_edge) {
              for (ordinal_type j=0;j<cardLine;++j) {
                for (ordinal_type i=0;i<cardLine;++i) { // y
                  for (ordinal_type k=0;k<npts;++k) {
                    output.access(idx,k) = output_y.access(j,k)*output_z.access(i,k);
                  }
                  idx++;
                }
              }
            }
            else {
              for (ordinal_type j=0;j<cardLine;++j) {
                for (ordinal_type i=0;i<cardLine;++i) { // y
                  for (ordinal_type k=0;k<npts;++k) {
                    output.access(idx,k) = 0.0;
                  }
                  idx++;
                }
              }
            }
          }
          
          // top side (y=1)
          {
            bool on_edge = false;
            if (std::abs(input_y(0)-1.0)<1.0e-12) {
              on_edge = true;
            }
            
            if (on_edge) {
              for (ordinal_type j=0;j<cardLine;++j) {
                for (ordinal_type i=0;i<cardLine;++i) {// x
                  for (ordinal_type k=0;k<npts;++k) {
                    output.access(idx,k) = output_x.access(i,k)*output_z.access(j,k);
                  }
                  idx++;
                }
              }
            }
            else {
              for (ordinal_type j=0;j<cardLine;++j) { // y
                for (ordinal_type i=0;i<cardLine;++i) {// x
                  for (ordinal_type k=0;k<npts;++k) {
                    output.access(idx,k) = 0.0;
                  }
                  idx++;
                }
              }
            }
          }
          
          // front side (z=-1)
          {
            bool on_edge = false;
            if (std::abs(input_z(0)+1.0)<1.0e-12) {
              on_edge = true;
            }
            
            if (on_edge) {
              for (ordinal_type j=0;j<cardLine;++j) {
                for (ordinal_type i=0;i<cardLine;++i) {// x
                  for (ordinal_type k=0;k<npts;++k) {
                    output.access(idx,k) = output_x.access(i,k)*output_y.access(j,k);
                  }
                  idx++;
                }
              }
            }
            else {
              for (ordinal_type j=0;j<cardLine;++j) { // y
                for (ordinal_type i=0;i<cardLine;++i) {// x
                  for (ordinal_type k=0;k<npts;++k) {
                    output.access(idx,k) = 0.0;
                  }
                  idx++;
                }
              }
            }
          }
          
          // back side (z=1)
          {
            bool on_edge = false;
            if (std::abs(input_z(0)-1.0)<1.0e-12) {
              on_edge = true;
            }
            
            if (on_edge) {
              for (ordinal_type j=0;j<cardLine;++j) {
                for (ordinal_type i=0;i<cardLine;++i) {// x
                  for (ordinal_type k=0;k<npts;++k) {
                    output.access(idx,k) = output_x.access(i,k)*output_y.access(j,k);
                  }
                  idx++;
                }
              }
            }
            else {
              for (ordinal_type j=0;j<cardLine;++j) { // y
                for (ordinal_type i=0;i<cardLine;++i) {// x
                  for (ordinal_type k=0;k<npts;++k) {
                    output.access(idx,k) = 0.0;
                  }
                  idx++;
                }
              }
            }
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
    void Basis_HFACE_HEX_In_FEM::getValues(Kokkos::DynRankView<outputValueValueType,outputValueProperties...> outputValues,
                                           const Kokkos::DynRankView<inputPointValueType, inputPointProperties...>  inputPoints,
                                           const Kokkos::DynRankView<vinvValueType,       vinvProperties...>        vinvLine,
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
      ordinal_type cardLine;  // = cardBubble+1;
      do {
        cardLine = Intrepid2::getPnCardinality<1>(++order);
      } while((6*cardLine !=  cardinality) && (order != Parameters::MaxOrder));
      
      auto vcprop = Kokkos::common_view_alloc_prop(inputPoints);
      typedef typename Kokkos::DynRankView< inputPointType, typename inputPointViewType::memory_space> workViewType;
      
      switch (operatorType) {
        case OPERATOR_VALUE: {
          auto workSize = 4*Serial<OPERATOR_VALUE>::getWorkSizePerPoint(order);
          workViewType  work(Kokkos::view_alloc("Basis_HFACE_HEX_In_FEM::getValues::work", vcprop), workSize, inputPoints.extent(0));
          typedef Functor<outputValueViewType,inputPointViewType,vinvViewType, workViewType,
          OPERATOR_VALUE,numPtsPerEval> FunctorType;
          Kokkos::parallel_for( policy, FunctorType(outputValues, inputPoints, vinvLine, work) );
          break;
        }
        case OPERATOR_GRAD: {
          break;
        }
        default: {
          INTREPID2_TEST_FOR_EXCEPTION( true , std::invalid_argument,
                                       ">>> ERROR (Basis_HFACE_HEX_In_FEM): Operator type not implemented" );
        }
      }
    }
  }
  
  // -------------------------------------------------------------------------------------
  template<typename DT, typename OT, typename PT>
  Basis_HFACE_HEX_In_FEM<DT,OT,PT>::
  Basis_HFACE_HEX_In_FEM( const ordinal_type order,
                         const EPointType   pointType ) {
    
    INTREPID2_TEST_FOR_EXCEPTION( !(pointType == POINTTYPE_EQUISPACED ||
                                    pointType == POINTTYPE_WARPBLEND), std::invalid_argument,
                                 ">>> ERROR (Basis_HFACE_HEX_In_FEM): pointType must be either equispaced or warpblend.");
    
    // this should be in host
    Basis_HGRAD_LINE_Cn_FEM<DT,OT,PT> lineBasis( order, pointType );
    
    const ordinal_type
    cardLine = lineBasis.getCardinality();
    
    this->vinvLine_   = Kokkos::DynRankView<typename ScalarViewType::value_type,DT>("HFACE::Hex::In::vinvLine", cardLine, cardLine);
    
    lineBasis.getVandermondeInverse(this->vinvLine_);
    
    this->basisCardinality_  = 6*cardLine*cardLine;
    this->basisDegree_       = order;
    this->basisCellTopology_ = shards::CellTopology(shards::getCellTopologyData<shards::Hexahedron<8> >() );
    this->basisType_         = BASIS_FEM_LAGRANGIAN;
    this->basisCoordinates_  = COORDINATES_CARTESIAN;
    this->functionSpace_     = FUNCTION_SPACE_HGRAD;
    
    // initialize tags
    {
      // Basis-dependent initializations
      const ordinal_type tagSize  = 4;        // size of DoF tag, i.e., number of fields in the tag
      const ordinal_type posScDim = 0;        // position in the tag, counting from 0, of the subcell dim
      const ordinal_type posScOrd = 1;        // position in the tag, counting from 0, of the subcell ordinal
      const ordinal_type posDfOrd = 2;        // position in the tag, counting from 0, of DoF ordinal relative to the subcell
      
      // An array with local DoF tags assigned to the basis functions, in the order of their local enumeration
      constexpr ordinal_type maxCardLine = Parameters::MaxOrder + 1;
      ordinal_type tags[6*maxCardLine*maxCardLine][4];
      
      //const ordinal_type face_yz[2] = {3, 1};
      //const ordinal_type face_xz[2] = {0, 2};
      //const ordinal_type face_xy[2] = {4, 5};
      
      {
        ordinal_type idx = 0;
        
        // left edge
        for (ordinal_type j=0;j<cardLine;++j) { // y
          for (ordinal_type i=0;i<cardLine;++i) { // y
            //const auto tag_y = lineBasis.getDofTag(j);
            //const auto tag_z = lineBasis.getDofTag(i);
            tags[idx][0] = 2; // face dof
            tags[idx][1] = 3;
            tags[idx][2] = j*cardLine+i; // local dof id
            tags[idx][3] = cardLine*cardLine; // total number of dofs in this vertex
            idx++;
          }
        }
        
        // bottom edge
        for (ordinal_type j=0;j<cardLine;++j) { // y
          for (ordinal_type i=0;i<cardLine;++i) { // y
            //const auto tag_x = lineBasis.getDofTag(j);
            //const auto tag_z = lineBasis.getDofTag(i);
            tags[idx][0] = 2; // face dof
            tags[idx][1] = 0;
            tags[idx][2] = j*cardLine+i; // local dof id
            tags[idx][3] = cardLine*cardLine; // total number of dofs in this vertex
            idx++;
          }
        }
        
        // right edge
        for (ordinal_type j=0;j<cardLine;++j) { // y
          for (ordinal_type i=0;i<cardLine;++i) { // y
            //const auto tag_y = lineBasis.getDofTag(j);
            tags[idx][0] = 2; // edge dof
            tags[idx][1] = 1;
            tags[idx][2] = j*cardLine+i; // local dof id
            tags[idx][3] = cardLine*cardLine; // total number of dofs in this vertex
            idx++;
          }
        }
        
        // top edge
        for (ordinal_type j=0;j<cardLine;++j) { // y
          for (ordinal_type i=0;i<cardLine;++i) { // y
            //const auto tag_x = lineBasis.getDofTag(j);
            tags[idx][0] = 2; // edge dof
            tags[idx][1] = 2;
            tags[idx][2] = j*cardLine+i; // local dof id
            tags[idx][3] = cardLine*cardLine; // total number of dofs in this vertex
            idx++;
          }
        }
        
        // front edge
        for (ordinal_type j=0;j<cardLine;++j) { // y
          for (ordinal_type i=0;i<cardLine;++i) { // y
            //const auto tag_x = lineBasis.getDofTag(j);
            tags[idx][0] = 2; // edge dof
            tags[idx][1] = 4;
            tags[idx][2] = j*cardLine+i; // local dof id
            tags[idx][3] = cardLine*cardLine; // total number of dofs in this vertex
            idx++;
          }
        }
        
        // top edge
        for (ordinal_type j=0;j<cardLine;++j) { // y
          for (ordinal_type i=0;i<cardLine;++i) { // y
            //const auto tag_x = lineBasis.getDofTag(j);
            tags[idx][0] = 2; // edge dof
            tags[idx][1] = 5;
            tags[idx][2] = j*cardLine+i; // local dof id
            tags[idx][3] = cardLine*cardLine; // total number of dofs in this vertex
            idx++;
          }
        }
        INTREPID2_TEST_FOR_EXCEPTION( idx != this->basisCardinality_ , std::runtime_error,
                                     ">>> ERROR (Basis_HFACE_HEX_In_FEM): " \
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
    dofCoordsHost("dofCoordsHost", this->basisCardinality_, this->basisCellTopology_.getDimension());
    
    // dofCoeffs on host and create its mirror view to device
    //Kokkos::DynRankView<typename ScalarViewType::value_type,typename SpT::array_layout,Kokkos::HostSpace>
    //dofCoeffsHost("dofCoeffsHost", this->basisCardinality_, this->basisCellTopology_.getDimension());
    
    Kokkos::DynRankView<typename ScalarViewType::value_type,DT>
    dofCoordsLine("dofCoordsLine", cardLine, 1);
    
    lineBasis.getDofCoords(dofCoordsLine);
    auto dofCoordsLineHost = Kokkos::create_mirror_view(Kokkos::HostSpace(), dofCoordsLine);
    Kokkos::deep_copy(dofCoordsLineHost, dofCoordsLine);
    
    {
      ordinal_type idx = 0;
      
      // x component (lineBasis(y) bubbleBasis(x))
      
      // left side
      for (ordinal_type j=0;j<cardLine;++j) { // y
        for (ordinal_type i=0;i<cardLine;++i) { // z
          dofCoordsHost(idx,0) = -1.0;
          dofCoordsHost(idx,1) = dofCoordsLineHost(j,0);
          dofCoordsHost(idx,2) = dofCoordsLineHost(i,0);
          idx++;
        }
      }
      
      // bottom side
      for (ordinal_type j=0;j<cardLine;++j) { // x
        for (ordinal_type i=0;i<cardLine;++i) { // z
          dofCoordsHost(idx,0) = dofCoordsLineHost(j,0);
          dofCoordsHost(idx,1) = -1.0;
          dofCoordsHost(idx,2) = dofCoordsLineHost(i,0);
          idx++;
        }
      }
      
      // right side
      for (ordinal_type j=0;j<cardLine;++j) { // y
        for (ordinal_type i=0;i<cardLine;++i) { // z
          dofCoordsHost(idx,0) = 1.0;
          dofCoordsHost(idx,1) = dofCoordsLineHost(j,0);
          dofCoordsHost(idx,2) = dofCoordsLineHost(i,0);
          idx++;
        }
      }
      
      // top side
      for (ordinal_type j=0;j<cardLine;++j) { // x
        for (ordinal_type i=0;i<cardLine;++i) { // z
          dofCoordsHost(idx,0) = dofCoordsLineHost(j,0);
          dofCoordsHost(idx,1) = 1.0;
          dofCoordsHost(idx,2) = dofCoordsLineHost(i,0);
          idx++;
        }
      }
      
      // front side
      for (ordinal_type j=0;j<cardLine;++j) { // x
        for (ordinal_type i=0;i<cardLine;++i) { // y
          dofCoordsHost(idx,0) = dofCoordsLineHost(j,0);
          dofCoordsHost(idx,1) = dofCoordsLineHost(i,0);
          dofCoordsHost(idx,2) = -1.0;
          idx++;
        }
      }
      
      // back side
      for (ordinal_type j=0;j<cardLine;++j) { // x
        for (ordinal_type i=0;i<cardLine;++i) { // y
          dofCoordsHost(idx,0) = dofCoordsLineHost(j,0);
          dofCoordsHost(idx,1) = dofCoordsLineHost(i,0);
          dofCoordsHost(idx,2) = 1.0;
          idx++;
        }
      }
    }
    
    this->dofCoords_ = Kokkos::create_mirror_view(typename DT::memory_space(), dofCoordsHost);
    Kokkos::deep_copy(this->dofCoords_, dofCoordsHost);
    
    //this->dofCoeffs_ = Kokkos::create_mirror_view(typename SpT::memory_space(), dofCoeffsHost);
    //Kokkos::deep_copy(this->dofCoeffs_, dofCoeffsHost);
  }
  
}

#endif
