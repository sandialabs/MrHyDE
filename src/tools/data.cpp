/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "data.hpp"

using namespace MrHyDE;

/////////////////////////////////////////////////////////////////////////////
//  Various constructors depending on the characteristics of the data (spatial,
//  transient, stochastic, etc.)
/////////////////////////////////////////////////////////////////////////////

Data::Data(const std::string & name_, const ScalarT & val) {
  name = name_;
  is_spatialdep = false;
  is_timedep = false;
  is_stochastic = false;
  spaceDim = 0;
}

/////////////////////////////////////////////////////////////////////////////


Data::Data(const std::string & name_, const std::string & datafile) {
  
  name = name_;
  spaceDim = 0;
  
  is_spatialdep = false;
  is_timedep = false;
  is_stochastic = false;
  
}

/////////////////////////////////////////////////////////////////////////////

Data::Data(const std::string & name_, const int & spaceDim_, const std::string & ptsfile) {
  
  name = name_;
  spaceDim = spaceDim_;
  
  is_spatialdep = true;
  is_timedep = false;
  is_stochastic = false;
  
  this->importPoints(ptsfile, spaceDim);
}

/////////////////////////////////////////////////////////////////////////////

Data::Data(const std::string & name_, const int & spaceDim_, const std::string & ptsfile,
           const std::string & dataprefix) {
  
  name = name_;
  spaceDim = spaceDim_;
  
  is_spatialdep = true;
  is_timedep = false;
  is_stochastic = false;
  
  this->importPoints(ptsfile, spaceDim);
  
  for (size_type i=0; i<points.extent(0); i++) {
    std::stringstream ss;
    ss << i;
    std::string str = ss.str();
    std::string filename = dataprefix + "." + str + ".dat";
    this->importData(filename);
  }
  
}

/////////////////////////////////////////////////////////////////////////////

Data::Data(const std::string & name_, const int & spaceDim_, const std::string & ptsfile,
           const std::string & dataprefix, const bool & separate_files) {
  
  name = name_;
  spaceDim = spaceDim_;
  
  is_spatialdep = true;
  is_timedep = false;
  is_stochastic = false;
  
  this->importPoints(ptsfile, spaceDim);
  if (separate_files) {
    for (size_type i=0; i<points.extent(0); i++) {
      std::stringstream ss;
      ss << i;
      std::string str = ss.str();
      std::string filename = dataprefix + "." + str + ".dat";
      this->importData(filename);
    }
  }
  else {
    this->importDataOneFile(dataprefix);
  }
  
}



Data::Data(const std::string & name_, const int & spaceDim_, const std::string & ptsfile,
           const std::string & dataprefix, const bool & separate_files,
           const int & Nx, const int & Ny, const int & Nz) {
  
  name = name_;
  spaceDim = spaceDim_;
  
  is_spatialdep = true;
  is_timedep = false;
  is_stochastic = false;
  
  this->importGridPoints(ptsfile, spaceDim, Nx, Ny, Nz);
  if (separate_files) {
    for (size_type i=0; i<points.extent(0); i++) {
      std::stringstream ss;
      ss << i;
      std::string str = ss.str();
      std::string dataname = dataprefix + "." + str + ".dat";
      this->importData(dataname);
    }
  }
  else {
    this->importDataOneFile(dataprefix);
  }
  
  
}

/////////////////////////////////////////////////////////////////////////////

Data::Data(const std::string & name_, const int & spaceDim_, const std::string & ptsfile,
           const int & Nsens, const std::string & dataprefix) {
  
  name = name_;
  spaceDim = spaceDim_;
  
  is_spatialdep = true;
  is_timedep = false;
  is_stochastic = false;
  
  this->importPoints(ptsfile, spaceDim);
  
  for (size_type i=0; i<points.extent(0); i++) {
    std::stringstream ss;
    ss << i;
    std::string str = ss.str();
    std::string dataname = dataprefix + "." + str + ".dat";
    this->importData(dataname);
  }
  
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

void Data::importPoints(const std::string & ptsfile, const int & spaceDim) {
  
  Teuchos::TimeMonitor timer(*pointImportTimer);
  
  std::ifstream fnmast(ptsfile.c_str());
  if (!fnmast.good()) {
    TEUCHOS_TEST_FOR_EXCEPTION(!fnmast.good(),std::runtime_error,"Error: could not find the data point file: " + ptsfile);
  }
  
  FILE* PointsFile = fopen(ptsfile.c_str(),"r");
  float x,y,z;
  
  std::vector<ScalarT> xvec,yvec,zvec;
  while( !feof(PointsFile) ) {
    char line[100] = "";
    fgets(line,100,PointsFile);
    if( strcmp(line,"") ) {
      if (spaceDim == 1) {
        sscanf(line, "%f", &x);
        xvec.push_back(x);
      }
      if (spaceDim == 2) {
        sscanf(line, "%f %f", &x, &y);
        xvec.push_back(x);
        yvec.push_back(y);
      }
      if (spaceDim == 3) {
        sscanf(line, "%f %f %f", &x, &y, &z);
        xvec.push_back(x);
        yvec.push_back(y);
        zvec.push_back(z);
      }
    }
  }
  size_t numSensors = xvec.size();
  
  points = Kokkos::View<ScalarT**,HostDevice>("data points",numSensors,spaceDim);
  for (size_t i=0; i<numSensors; i++) {
    if (spaceDim >0)
      points(i,0) = xvec[i];
    if (spaceDim >1)
      points(i,1) = yvec[i];
    if (spaceDim >2)
      points(i,2) = zvec[i];
  }
  
  
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

void Data::importGridPoints(const std::string & ptsfile, const int & spaceDim,
                            const int & Nx, const int & Ny, const int & Nz) {
  
  Teuchos::TimeMonitor timer(*pointImportTimer);
  
  std::ifstream fnmast(ptsfile.c_str());
  if (!fnmast.good()) {
    TEUCHOS_TEST_FOR_EXCEPTION(!fnmast.good(),std::runtime_error,"Error: could not find the data point file: " + ptsfile);
  }
  
  FILE* PointsFile = fopen(ptsfile.c_str(),"r");
  float x,y,z;
  
  std::vector<ScalarT> xvec,yvec,zvec;
  while( !feof(PointsFile) ) {
    char line[100] = "";
    fgets(line,100,PointsFile);
    if( strcmp(line,"") ) {
      if (spaceDim == 1) {
        sscanf(line, "%f", &x);
        xvec.push_back(x);
      }
      if (spaceDim == 2) {
        sscanf(line, "%f %f", &x, &y);
        xvec.push_back(x);
        yvec.push_back(y);
      }
      if (spaceDim == 3) {
        sscanf(line, "%f %f %f", &x, &y, &z);
        xvec.push_back(x);
        yvec.push_back(y);
        zvec.push_back(z);
      }
    }
  }
  
  if (spaceDim == 1) {
    grid_x = Kokkos::View<ScalarT*,HostDevice>("sensor grid x pts",Nx);
    for (int k=0; k<Nx; k++) {
      grid_x(k) = xvec[k];
    }
  }
  else if (spaceDim == 2) {
    grid_x = Kokkos::View<ScalarT*,HostDevice>("sensor grid x pts",Nx);
    for (int k=0; k<Nx; k++) {
      grid_x(k) = xvec[k];
    }
    
    grid_y = Kokkos::View<ScalarT*,HostDevice>("sensor grid y pts",Ny);
    for (int k=0; k<Ny; k++) {
      grid_y(k) = yvec[(k)*Nx];
    }
  }
  else if (spaceDim == 3) {
    grid_x = Kokkos::View<ScalarT*,HostDevice>("sensor grid x pts",Nx);
    for (int k=0; k<Nx; k++) {
      grid_x(k) = xvec[k];
    }
    
    grid_y = Kokkos::View<ScalarT*,HostDevice>("sensor grid y pts",Ny);
    for (int k=0; k<Ny; k++) {
      grid_y(k) = yvec[(k)*Nx];
    }
    
    grid_z = Kokkos::View<ScalarT*,HostDevice>("sensor grid z pts",Nz);
    for (int k=0; k<Nz; k++) {
      grid_z(k) = zvec[(k)*Nx*Ny];
    }
  }
  
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

void Data::importDataOneFile(const std::string & datafile) {
  
  Teuchos::TimeMonitor timer(*dataImportTimer);
  
  std::ifstream fnmast(datafile.c_str());
  if (!fnmast.good()) {
    TEUCHOS_TEST_FOR_EXCEPTION(!fnmast.good(),std::runtime_error,"Error: could not find the data file: " + datafile);
  }
  
  
  std::vector<std::vector<ScalarT> > values;
  std::ifstream fin(datafile.c_str());
  
  for (std::string line; std::getline(fin, line); )
  {
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream in(line);
    values.push_back(
                     std::vector<ScalarT>(std::istream_iterator<ScalarT>(in),
                                          std::istream_iterator<ScalarT>()));
  }
  
  int maxstates = 0;
  for (size_t i=0; i<values.size(); i++) {
    maxstates = std::max(maxstates,(int)values[i].size());
  }
  
  for (size_t i=0; i<values.size(); i++) {
    Kokkos::View<ScalarT**,HostDevice> newdata("sensor data",1,maxstates);
    for (size_t j=0; j<values[i].size(); j++) {
      newdata(0,j) = values[i][j];
    }
    data.push_back(newdata);
  }
  
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

void Data::importData(const std::string & datafile) {
  
  Teuchos::TimeMonitor timer(*dataImportTimer);
  
  std::ifstream fnmast(datafile.c_str());
  if (!fnmast.good()) {
    TEUCHOS_TEST_FOR_EXCEPTION(!fnmast.good(),std::runtime_error,"Error: could not find the data file: " + datafile);
  }
  
  std::vector<std::vector<ScalarT> > values;
  std::ifstream fin(datafile.c_str());
  
  for (std::string line; std::getline(fin, line); )
  {
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream in(line);
    values.push_back(
                     std::vector<ScalarT>(std::istream_iterator<ScalarT>(in),
                                          std::istream_iterator<ScalarT>()));
  }
  
  int maxstates = 0;
  for (size_t i=0; i<values.size(); i++) {
    maxstates = std::max(maxstates,(int)values[i].size());
  }
  
  Kokkos::View<ScalarT**,HostDevice> newdata("sensor data",values.size(),maxstates);
  for (size_t i=0; i<values.size(); i++) {
    for (size_t j=0; j<values[i].size(); j++) {
      newdata(i,j) = values[i][j];
    }
  }
  
  data.push_back(newdata);
}


/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

void Data::findClosestPoint(const Kokkos::View<ScalarT**, AssemblyDevice> &tstpts,
                            Kokkos::View<int*, CompadreDevice> &closestpts) const {
  
  Teuchos::TimeMonitor timer(*dataClosestTimer);
  Kokkos::View<ScalarT*, AssemblyDevice> distance("distance",points.extent(0));
 
  Compadre::NeighborLists<Kokkos::View<int*, CompadreDevice> > neighborlists = CompadreInterface_constructNeighborLists(points, tstpts, distance);
  auto closestpts_tmp = neighborlists.getNeighborLists();
  
  // Safeguard against multiple neighbors ... just take the first (closest)
  int prog = 0;
  for (size_type pt=0; pt<closestpts.extent(0); ++pt) {
    int np = neighborlists.getNumberOfNeighborsHost((int)pt);
    closestpts(pt) = closestpts_tmp(prog);
    prog += np;
  }
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

void Data::findClosestPoint(const Kokkos::View<ScalarT**, AssemblyDevice> &tstpts,
                            Kokkos::View<int*, CompadreDevice> &closestpts,
                            Kokkos::View<ScalarT*, AssemblyDevice> &distance) const {

  Teuchos::TimeMonitor timer(*dataClosestTimer);
  
  Compadre::NeighborLists<Kokkos::View<int*, CompadreDevice> > neighborlists = CompadreInterface_constructNeighborLists(points, tstpts, distance);
  auto closestpts_tmp = neighborlists.getNeighborLists();
  
  // Safeguard against multiple neighbors ... just take the first (closest)
  int prog = 0;
  for (size_type pt=0; pt<closestpts.extent(0); ++pt) {
    int np = neighborlists.getNumberOfNeighborsHost((int)pt);
    closestpts(pt) = closestpts_tmp(prog);
    prog += np;
  }
  
  // Turning off the brute force methos, but leaving in the code for testing
  bool bruteforce = false;
  
  if (bruteforce) {
    for (size_type pt=0; pt<tstpts.extent(0); ++pt) {
      ScalarT currdist = 1.0e20;
      int cpt = -1;
      for (size_type s=0; s<points.extent(0); ++s) {
        ScalarT tdist = 0.0;
        for (size_type dim=0; dim<points.extent(1); ++dim) {
          tdist += std::pow((points(s,dim) - tstpts(pt,dim)),2.0);
        }
        if (tdist < currdist) {
          cpt = (int)s;
          currdist = tdist;
          //std::cout << "Error: found a closer point " << currdist << "  " << tdist << std::endl;
        }
      }
      closestpts(pt) = cpt;
      distance(pt) = std::sqrt(currdist);
    }
  }
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

int Data::findClosestGridPoint(const ScalarT & x, const ScalarT & y,
                               const ScalarT & z, ScalarT & distance) const {
  
  Teuchos::TimeMonitor timer(*dataClosestTimer);
  
  int node = 0;
  
  if (spaceDim == 1) {
    ScalarT dist = (ScalarT)RAND_MAX;
    int node_x = 0;
    for(size_type i=0; i<grid_x.extent(0); i++ ) {
      ScalarT xhat = grid_x(i);
      ScalarT d = (x-xhat)*(x-xhat);
      if( d<dist ) {
        node_x = i;
        dist = d;
      }
    }
    node = node_x;
    distance = sqrt(dist);
  }
  if (spaceDim == 2) {
    ScalarT dist_x = (ScalarT)RAND_MAX;
    ScalarT dist_y = (ScalarT)RAND_MAX;
    int node_x=0, node_y=0;
    for(size_type i=0; i<grid_x.extent(0); i++ ) {
      ScalarT xhat = grid_x(i);
      ScalarT d = (x-xhat)*(x-xhat);
      if( d<dist_x ) {
        node_x = i;
        dist_x = d;
      }
    }
    for(size_type i=0; i<grid_y.extent(0); i++ ) {
      ScalarT yhat = grid_y(i);
      ScalarT d = (y-yhat)*(y-yhat);
      if( d<dist_y ) {
        node_y = i;
        dist_y = d;
      }
    }
    node = node_y*grid_x.extent(0) + node_x;
    distance = sqrt(dist_x + dist_y);
  }
  if (spaceDim == 3) {
    ScalarT dist_x = (ScalarT)RAND_MAX;
    ScalarT dist_y = (ScalarT)RAND_MAX;
    ScalarT dist_z = (ScalarT)RAND_MAX;
    int node_x=0, node_y=0, node_z=0;
    for(size_type i=0; i<grid_x.extent(0); i++ ) {
      ScalarT xhat = grid_x(i);
      ScalarT d = (x-xhat)*(x-xhat);
      if( d<dist_x ) {
        node_x = i;
        dist_x = d;
      }
    }
    for(size_type i=0; i<grid_y.extent(0); i++ ) {
      ScalarT yhat = grid_y(i);
      ScalarT d = (y-yhat)*(y-yhat);
      if( d<dist_y ) {
        node_y = i;
        dist_y = d;
      }
    }
    for(size_type i=0; i<grid_z.extent(0); i++ ) {
      ScalarT zhat = grid_z(i);
      ScalarT d = (z-zhat)*(z-zhat);
      if( d<dist_z ) {
        node_z = i;
        dist_z = d;
      }
    }
    node = node_z*grid_x.extent(0)*grid_y.extent(0) + node_y*grid_x.extent(0) + node_x;
    distance = sqrt(dist_x + dist_y + dist_z);
    
  }
  
  return node;
  
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

std::vector<Kokkos::View<ScalarT**,HostDevice> > Data::getData() {
  return data;
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

Kokkos::View<ScalarT**,HostDevice> Data::getData(const int & sensnum) {
  return data[sensnum];
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

Kokkos::View<ScalarT**,HostDevice> Data::getPoints() {
  return points;
}
