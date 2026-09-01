/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "ExoReader.hpp"

#include <cstring>

using namespace MrHyDE;

// ========================================================================================
// constructor: initialize with file path
// ========================================================================================
ExoReader::ExoReader(const std::string& path) : file_(path) {}

// ========================================================================================
// destructor: automatically close file if open
// ========================================================================================
ExoReader::~ExoReader() {
  try { close(); } catch (...) {}
}

// ========================================================================================
// ensure file is open: throw exception if not
// ========================================================================================
void ExoReader::ensureOpen_() const {
  if (exoid_ < 0) throw std::runtime_error("ExoReader: file is not open");
}

// ========================================================================================
// open exodus file for reading
// ========================================================================================
void ExoReader::open() {
  if (isOpen()) return;
  int cpu_ws    = 8;   // use doubles
  int io_ws     = 0;   // read as stored
  float version = 0.0f;
  exoid_        = ex_open(file_.c_str(), EX_READ, &cpu_ws, &io_ws, &version);


  if (exoid_ < 0) throw std::runtime_error("ExoReader: ex_open failed for '" + file_ + "'");

  // reset caches
  grid_ok_       = false;
  var_map_ok_    = false;
  block_id_      = 0;
  num_elems_     = -1;
  num_elem_vars_ = 0;
  init_cached_   = false;
  cached_num_dims_ = cached_num_nodes_ = cached_num_elems_ = 0;
  cached_num_blocks_ = cached_num_node_sets_ = cached_num_side_sets_ = 0;

  name_to_varidx_.clear();
}

// ========================================================================================
// close exodus file
// ========================================================================================
void ExoReader::close() {
  if (!isOpen()) return;
  (void)ex_close(exoid_); // keep state consistent even if close errors
  exoid_ = -1;
}

// ========================================================================================
// query exodus initialization data: get basic file dimensions
// ========================================================================================
void ExoReader::queryInit_(int& numDims, int& numNodes, int& numElems, int& numBlocks, int& numNodeSets, int& numSideSets) {
  ensureOpen_();
  if (!init_cached_) {
    char title[MAX_LINE_LENGTH+1];
    std::memset(title, 0, sizeof(title));
    numDims=numNodes=numElems=numBlocks=numNodeSets=numSideSets=0;
    if (ex_get_init(exoid_, title, &numDims, &numNodes, &numElems, &numBlocks, &numNodeSets, &numSideSets) != 0) {
      throw std::runtime_error("ExoReader: ex_get_init failed");
    }
    cached_num_dims_      = numDims;
    cached_num_nodes_     = numNodes;
    cached_num_elems_     = numElems;
    cached_num_blocks_    = numBlocks;
    cached_num_node_sets_ = numNodeSets;
    cached_num_side_sets_ = numSideSets;
    init_cached_ = true;
  } else {
    numDims     = cached_num_dims_;
    numNodes    = cached_num_nodes_;
    numElems    = cached_num_elems_;
    numBlocks   = cached_num_blocks_;
    numNodeSets = cached_num_node_sets_;
    numSideSets = cached_num_side_sets_;
  }
}

// ========================================================================================
// require single element block: ensure exactly one block exists and cache its info
// ========================================================================================
void ExoReader::requireSingleBlock_() {
  ensureOpen_();
  if (block_id_ != 0 && num_elems_ >= 0) return; // already cached

  int numDims=0, numNodes=0, numElems=0, numBlocks=0, numNodeSets=0, numSideSets=0;
  queryInit_(numDims, numNodes, numElems, numBlocks, numNodeSets, numSideSets);
  if (numBlocks != 1) {
    throw std::runtime_error("ExoReader: expected exactly 1 element block, found " + std::to_string(numBlocks));
  }

  std::vector<ex_entity_id> ids(numBlocks, 0);
  if (ex_get_ids(exoid_, EX_ELEM_BLOCK, ids.data()) != 0) {
    throw std::runtime_error("ExoReader: ex_get_ids(EX_ELEM_BLOCK) failed");
  }
  block_id_ = ids[0];

  // query element count of this block
  char elem_type[MAX_STR_LENGTH+1] = {0};
  int64_t numElemsInBlock=0, nodesPerElem=0, numEdges=0, numFaces=0, numAttr=0;
  if (ex_get_block(exoid_, EX_ELEM_BLOCK, block_id_, elem_type, &numElemsInBlock, &nodesPerElem, &numEdges, &numFaces, &numAttr) != 0) {
    throw std::runtime_error("ExoReader: ex_get_block failed");
  }
  num_elems_ = numElemsInBlock;
}

// ========================================================================================
// get element block ID
// ========================================================================================
ex_entity_id ExoReader::blockId() {
  requireSingleBlock_();
  return block_id_;
}

// ========================================================================================
// get number of elements in the block
// ========================================================================================
int64_t ExoReader::numElems() {
  requireSingleBlock_();
  return num_elems_;
}

// ========================================================================================
// compute grid and domain geometry from mesh coordinates
// ========================================================================================
void ExoReader::computeGridAndDomain() {
  ensureOpen_();

  int numDims=0, numNodes=0, numElems=0, numBlocks=0, numNodeSets=0, numSideSets=0;
  queryInit_(numDims, numNodes, numElems, numBlocks, numNodeSets, numSideSets);

  // coordinates
  std::vector<double> xCoords(numNodes), yCoords(numNodes);
  if (ex_get_coord(exoid_, xCoords.data(), yCoords.data(), nullptr) != 0) {
    throw std::runtime_error("ExoReader: ex_get_coord failed");
  }
  if (numNodes <= 0) throw std::runtime_error("ExoReader: no nodes in file");

  // unique sorted X/Y with tolerance
  const double eps = 1e-10;
  auto Xu = unique_sorted_(xCoords, eps);
  auto Yu = unique_sorted_(yCoords, eps);
  if (Xu.size() < 2 || Yu.size() < 2) {
    throw std::runtime_error("ExoReader: degenerate grid (need >= 2 unique coords per axis)");
  }

  // compute spacings and sanity-check uniform rectilinear grid
  std::vector<double> xSteps, ySteps;
  for (size_t i = 1; i < Xu.size(); ++i) xSteps.push_back(Xu[i] - Xu[i-1]);
  for (size_t j = 1; j < Yu.size(); ++j) ySteps.push_back(Yu[j] - Yu[j-1]);

  // enforce near-uniform spacing (tolerance is generous to allow minor floating differences)
  auto checkUniform = [](const std::vector<double>& deltas, const char axis)->double {
    if (deltas.empty()) return 1.0;
    const double base = deltas.front();
    const double relTol = 1e-4;
    const double absTol = 1e-8;
    const double tol    = std::max(absTol, relTol * std::max(1.0, std::abs(base)));
    for (size_t i = 1; i < deltas.size(); ++i) {
      if (std::abs(deltas[i] - base) > tol) {
        throw std::runtime_error(std::string("ExoReader: non-uniform spacing in ") + axis + "-direction");
      }
    }
    return base;
  };

  double dx_uniform = checkUniform(xSteps, 'x');
  double dy_uniform = checkUniform(ySteps, 'y');

  // fill public geometry
  x0         = Xu.front();
  y0         = Yu.front();
  dx         = dx_uniform;
  dy         = dy_uniform;
  nx         = static_cast<int>(Xu.size()) - 1;
  ny         = static_cast<int>(Yu.size()) - 1;
  domainX    = Xu.back() - Xu.front();
  domainY    = Yu.back() - Yu.front();
  domainDiag = std::sqrt(domainX*domainX + domainY*domainY);
  inv_dx_    = 1.0 / dx;
  inv_dy_    = 1.0 / dy;

  // consistency with element count
  requireSingleBlock_();
  if (static_cast<int64_t>(nx) * static_cast<int64_t>(ny) != num_elems_) {
    throw std::runtime_error("ExoReader: nx*ny != number of elements in block");
  }

  grid_ok_ = true;
}

// ========================================================================================
// get element variable name to index mapping
// ========================================================================================
const std::unordered_map<std::string,int>& ExoReader::elemVarIndex() {
  if (!var_map_ok_) buildElemVarNameIndex_();
  return name_to_varidx_;
}

// ========================================================================================
// build element variable name to index mapping
// ========================================================================================
void ExoReader::buildElemVarNameIndex_() {
  ensureOpen_();
  // number of element variables
  int numElemVars = 0;
  if (ex_get_variable_param(exoid_, EX_ELEM_BLOCK, &numElemVars) != 0) {
    throw std::runtime_error("ExoReader: ex_get_variable_param(EX_ELEM_BLOCK) failed");
  }
  num_elem_vars_ = numElemVars;

  name_to_varidx_.clear();
  if (numElemVars <= 0) { var_map_ok_ = true; return; }

  // exodus names are fixed-length char arrays
  std::vector<char*> names(numElemVars, nullptr);
  std::vector<std::vector<char>> storage(numElemVars, std::vector<char>(MAX_STR_LENGTH+1, 0));
  for (int i = 0; i < numElemVars; ++i) names[i] = storage[i].data();

  if (ex_get_variable_names(exoid_, EX_ELEM_BLOCK, numElemVars, names.data()) != 0) {
    throw std::runtime_error("ExoReader: ex_get_variable_names(EX_ELEM_BLOCK) failed");
  }

  for (int i = 0; i < numElemVars; ++i) {
    std::string varName = trimPad_(names[i]);
    if (!varName.empty()) name_to_varidx_[varName] = i+1; // 1-based index
  }
  var_map_ok_ = true;
}

// ========================================================================================
// read element variable by index
// ========================================================================================
std::vector<double> ExoReader::readElemVarByIndex(int var_idx, int time_step) {
  ensureOpen_();
  requireSingleBlock_();

  if (var_idx <= 0)
    throw std::out_of_range("ExoReader: element variable index must be >= 1");

  if (num_elems_ <= 0) return {};
  if (num_elem_vars_ == 0) buildElemVarNameIndex_(); // ensure we know count

  const int ex_var_idx = var_idx;
  if (ex_var_idx > num_elem_vars_) throw std::out_of_range("ExoReader: element variable index out of range");

  std::vector<double> vals(static_cast<size_t>(num_elems_), 0.0);
  if (ex_get_var(exoid_, time_step, EX_ELEM_BLOCK, ex_var_idx, block_id_, num_elems_, vals.data()) != 0) {
    throw std::runtime_error("ExoReader: ex_get_var failed for element variable index");
  }
  return vals;
}

// ========================================================================================
// read element variable by name
// ========================================================================================
std::vector<double> ExoReader::readElemVarByName(const std::string& name, int time_step) {
  if (!var_map_ok_) buildElemVarNameIndex_();
  auto it = name_to_varidx_.find(name);
  if (it == name_to_varidx_.end())
    throw std::runtime_error("ExoReader: element variable '" + name + "' not found");
  return readElemVarByIndex(it->second, time_step);
}


// ========================================================================================
// trim padding from exodus string: remove trailing spaces and nulls
// ========================================================================================
std::string ExoReader::trimPad_(const char* s) {
  if (!s) return std::string();
  std::string t(s);
  while (!t.empty() && (t.back() == ' ' || t.back() == '\0'))
    t.pop_back();
  return t;
}

// ========================================================================================
// get unique sorted values with tolerance: remove duplicates within epsilon
// ========================================================================================
std::vector<double> ExoReader::unique_sorted_(const std::vector<double>& a, double eps) {
  std::vector<double> b = a;
  std::sort(b.begin(), b.end());
  auto eq = [eps](double A, double B){ return std::abs(A-B) < eps; };
  b.erase(std::unique(b.begin(), b.end(), eq), b.end());
  return b;
}


// ============================================================================
// Read a nodal field from the Exodus file
// ============================================================================
void ExoReader::readNodalField(const std::string& fieldName,
  std::vector<double>& fieldData,
  int timeStep) const
{
  if (exoid_ < 0) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
      "ExoReader::readNodalField called before Exodus file open.");
  }

  // Query node count
  int numDims = 0, numNodes = 0, numElems = 0, numBlocks = 0, numNodeSets = 0, numSideSets = 0;
  const_cast<ExoReader*>(this)->queryInit_(numDims, numNodes, numElems, numBlocks, numNodeSets, numSideSets);

  fieldData.assign(numNodes, 0.0);

  // Default to last step if timeStep < 0
  int numSteps = ex_inquire_int(exoid_, EX_INQ_TIME);
  int step = (timeStep < 0) ? numSteps : timeStep;
  if (step <= 0) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
      "ExoReader::readNodalField: invalid timestep.");
  }

  // Get list of nodal variable names
  int numNodeVars = 0;
  if (ex_get_variable_param(exoid_, EX_NODAL, &numNodeVars) != 0) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
      "ExoReader::readNodalField: ex_get_variable_param(EX_NODAL) failed.");
  }
  if (numNodeVars <= 0) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
      "ExoReader::readNodalField: no nodal variables in file.");
  }

  std::vector<std::string> varNames(numNodeVars);
  std::vector<char*> namePtrs(numNodeVars);
  std::vector<std::vector<char>> storage(numNodeVars, std::vector<char>(MAX_STR_LENGTH + 1, 0));
  for (int i = 0; i < numNodeVars; ++i) namePtrs[i] = storage[i].data();
  if (ex_get_variable_names(exoid_, EX_NODAL, numNodeVars, namePtrs.data()) != 0) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
      "ExoReader::readNodalField: ex_get_variable_names(EX_NODAL) failed.");
  }

  for (int i = 0; i < numNodeVars; ++i)
    varNames[i] = trimPad_(namePtrs[i]);

  // Find variable index (1-based)
  int varIndex = -1;
  for (int i = 0; i < numNodeVars; ++i) {
    if (varNames[i] == fieldName) { varIndex = i + 1; break; }
  }
  if (varIndex < 0) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
      "ExoReader::readNodalField: variable '" + fieldName + "' not found.");
  }

  // Read the data
  std::vector<double> buf(numNodes);
  int err = ex_get_var(exoid_, step, EX_NODAL, varIndex, 1, numNodes, buf.data());
  TEUCHOS_TEST_FOR_EXCEPTION(err != 0, std::runtime_error,
    "ExoReader::readNodalField: ex_get_var failed for nodal variable '" + fieldName + "'.");

  fieldData.swap(buf);
}

// ============================================================================
// Scatter host vector -> Tpetra Rfield vector
// ============================================================================
void ExoReader::scatterToRField(const std::vector<double>& nodalValues,
   const Teuchos::RCP<Tpetra::Vector<double, LO, GO>>& Rfield,
   const Teuchos::RCP<const Tpetra::Map<LO, GO>>& dofMap) const
{
TEUCHOS_TEST_FOR_EXCEPTION(Rfield.is_null(), std::runtime_error,
"scatterToRField: Rfield vector is null.");
TEUCHOS_TEST_FOR_EXCEPTION(dofMap.is_null(), std::runtime_error,
"scatterToRField: DOF map is null.");

auto& vec = *Rfield;
const auto& map = *dofMap;

for (LO lid = 0; lid < map.getLocalNumElements(); ++lid) {
GO gid = map.getGlobalElement(lid);
if (static_cast<size_t>(gid) < nodalValues.size()) {
vec.replaceLocalValue(lid, nodalValues[gid]);
}
}
}
