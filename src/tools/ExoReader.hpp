/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#ifndef MRHYDE_EXOREADER_H
#define MRHYDE_EXOREADER_H

#include "preferences.hpp"
#include "trilinos.hpp"
#include <exodusII.h>

#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <limits>

namespace MrHyDE {

// ========================================================================================
// ExoReader class: reads and processes Exodus II mesh files
// ========================================================================================
class ExoReader {

  typedef Kokkos::View<ScalarT*, AssemblyDevice> View_Scalar1;

  public:

    // ========================================================================================
    // public domain geometry data
    // ========================================================================================
    
    // dimensions of the domain
    double domainX    = 0.0;  // domain width in x-direction
    double domainY    = 0.0;  // domain width in y-direction
    double domainDiag = 0.0;  // domain diagonal length

    // origin, spacing, and number of cells in the domain
    double x0 = 0.0;  // domain origin in x-direction
    double y0 = 0.0;  // domain origin in y-direction
    double dx = 1.0;  // element width in x-direction
    double dy = 1.0;  // element width in y-direction
    int    nx = 1;    // number of elements in x-direction
    int    ny = 1;    // number of elements in y-direction

    // ========================================================================================
    // constructor and destructor
    // ========================================================================================
    explicit ExoReader(const std::string& path);  // constructor: takes path to exodus file
    ~ExoReader();  // destructor: automatically closes file if open

    // ========================================================================================
    // file operations
    // ========================================================================================
    void open();  // open the exodus file for reading
    void close();  // close the exodus file
    bool isOpen() const { return exoid_ >= 0; }  // check if file is currently open

    // ========================================================================================
    // metadata access
    // ========================================================================================
    ex_entity_id blockId();  // get the element block ID
    int64_t      numElems();  // get the number of elements in the block

    // ========================================================================================
    // geometry computation
    // ========================================================================================
    void computeGridAndDomain();  // compute domain geometry from mesh coordinates

    // ========================================================================================
    // variable access
    // ========================================================================================
    const std::unordered_map<std::string,int>& elemVarIndex();  // get variable name to index mapping

    // read a variable by index
    std::vector<double> readElemVarByIndex(int var_idx, int time_step = 1);

    // read a variable by name
    std::vector<double> readElemVarByName (const std::string& name, int time_step = 1);

    // templated API: caller controls the memory space by ViewType
    template <class ViewType>
    ViewType readCellToViewT(const std::string& name, int time_step = 1);


    // START NEW

    // Reads a nodal field from the Exodus file into a host vector.
    // The field must exist as a nodal variable in the file.
    void readNodalField(const std::string& fieldName,
      std::vector<double>& fieldData,
      int timeStep = 1) const;

    // Scatters a host nodal vector into the discretized Rfield (Tpetra Vector).
    void scatterToRField(const std::vector<double>& nodalValues,
      const Teuchos::RCP<Tpetra::Vector<double, LO, GO>>& Rfield,
      const Teuchos::RCP<const Tpetra::Map<LO, GO>>& dofMap) const;

    // END NEW

  private:
    // ========================================================================================
    // private member variables
    // ========================================================================================

    // path and exodus handle
    std::string file_;  // path to the exodus file
    int  exoid_ = -1;   // exodus file handle (negative means closed)

    // cached metadata
    bool init_cached_     = false;  // flag for cached exodus init data
    int cached_num_dims_       = 0;
    int cached_num_nodes_      = 0;
    int cached_num_elems_      = 0;
    int cached_num_blocks_     = 0;
    int cached_num_node_sets_  = 0;
    int cached_num_side_sets_  = 0;
    bool grid_ok_          = false;  // flag indicating if grid geometry has been computed
    bool var_map_ok_       = false;  // flag indicating if variable name map has been built
    ex_entity_id block_id_ = 0;      // cached element block ID
    int64_t num_elems_     = -1;     // cached number of elements
    int num_elem_vars_     = 0;      // cached number of element variables
    std::unordered_map<std::string,int> name_to_varidx_;  // variable name to index mapping

    // derived helpers
    double inv_dx_ = 1.0;  // inverse of x-direction spacing (1/dx)
    double inv_dy_ = 1.0;  // inverse of y-direction spacing (1/dy)

    // ========================================================================================
    // internal helper methods
    // ========================================================================================
    void ensureOpen_() const;  // ensure file is open, throw if not
    void queryInit_(int& numDims, int& numNodes, int& numElems, int& numBlocks, int& numNodeSets, int& numSideSets);  // query exodus initialization data
    void requireSingleBlock_();  // ensure exactly one element block exists
    void buildElemVarNameIndex_();  // build variable name to index mapping
    static std::string trimPad_(const char* s);  // trim padding from exodus string
    static std::vector<double> unique_sorted_(const std::vector<double>& a, double eps);  // get unique sorted values with tolerance
};


// ========================================================================================
// templated method implementation: read element variable to Kokkos view
// ========================================================================================
template <class ViewType>
ViewType ExoReader::readCellToViewT(const std::string& name, int time_step) {
  ensureOpen_();
  requireSingleBlock_();

  if (!var_map_ok_) buildElemVarNameIndex_();
  auto it = name_to_varidx_.find(name);
  if (it == name_to_varidx_.end())
    throw std::runtime_error("ExoReader::readCellToView: element variable '" + name + "' not found");

  const int ex_var_idx = it->second;           // Exodus is 1-based
  const int64_t ne = num_elems_;

  static_assert(ViewType::rank == 1, "readCellToViewT expects a 1D View");
  using value_type = typename ViewType::non_const_value_type;

  // allocate return view in the caller-selected memory space
  ViewType v(Kokkos::view_alloc(Kokkos::WithoutInitializing, "exo_" + name), ne);

  // read Exodus into a host buffer (double), then copy/convert to v
  std::vector<double> buf(ne);
  if (ex_get_var(exoid_, time_step, EX_ELEM_BLOCK, ex_var_idx, block_id_, ne, buf.data()) != 0)
    throw std::runtime_error("ExoReader::readCellToView: ex_get_var failed for '" + name + "'");

  auto h = Kokkos::create_mirror_view(v);
  for (int64_t i = 0; i < ne; ++i) h(i) = static_cast<value_type>(buf[i]);
  Kokkos::deep_copy(v, h);

  return v;
}

}

#endif
