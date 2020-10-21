/***********************************************************************
 This is a framework for solving Multi-resolution Hybridized
 Differential Equations (MrHyDE), an optimized version of
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "data.hpp"

using namespace MrHyDE;

/////////////////////////////////////////////////////////////////////////////
//  Various constructors depending on the characteristics of the data (spatial,
//  transient, stochastic, etc.)
/////////////////////////////////////////////////////////////////////////////

data::data(const std::string & name_, const ScalarT & val) {
  name = name_;
  is_spatialdep = false;
  is_timedep = false;
  is_stochastic = false;
  //mydata = FC(1);
  //mydata(0) = val;
  spaceDim = 0;
  numSensors = 0;
}

/////////////////////////////////////////////////////////////////////////////


data::data(const std::string & name_, const std::string & datafile) {
  
  name = name_;
  spaceDim = 0;
  
  is_spatialdep = false;
  is_timedep = false;
  is_stochastic = false;
  
  /*
   vector<ScalarT> datavec;
   
   FILE* DataFile = fopen(datafile.c_str(),"r");
   float d;
   while( !feof(DataFile) ) {
   char line[100] = "";
   fgets(line,100,DataFile);
   if( strcmp(line,"") ) {
   sscanf(line, "%f", &d);
   datavec.push_back(d);
   }
   }
   mydata = FC(datavec.size(),1);
   for (size_t k=0; k<datavec.size(); k++) {
   mydata(k) = datavec[k];
   }
   */
}

/////////////////////////////////////////////////////////////////////////////

data::data(const std::string & name_, const int & spaceDim_, const std::string & ptsfile) {
  
  name = name_;
  spaceDim = spaceDim_;
  
  is_spatialdep = true;
  is_timedep = false;
  is_stochastic = false;
  
  this->importPoints(ptsfile, spaceDim);
}

/////////////////////////////////////////////////////////////////////////////

data::data(const std::string & name_, const int & spaceDim_, const std::string & ptsfile,
           const std::string & sensorprefix) {
  
  name = name_;
  spaceDim = spaceDim_;
  
  is_spatialdep = true;
  is_timedep = false;
  is_stochastic = false;
  
  this->importPoints(ptsfile, spaceDim);
  
  for (int i=0; i<numSensors; i++) {
    std::stringstream ss;
    ss << i;
    std::string str = ss.str();
    std::string sensorname = sensorprefix + "." + str + ".dat";
    this->importSensor(sensorname);
  }
  
}

/////////////////////////////////////////////////////////////////////////////

data::data(const std::string & name_, const int & spaceDim_, const std::string & ptsfile,
           const std::string & sensorprefix, const bool & separate_files) {
  
  name = name_;
  spaceDim = spaceDim_;
  
  is_spatialdep = true;
  is_timedep = false;
  is_stochastic = false;
  
  this->importPoints(ptsfile, spaceDim);
  if (separate_files) {
    for (int i=0; i<numSensors; i++) {
      std::stringstream ss;
      ss << i;
      std::string str = ss.str();
      std::string sensorname = sensorprefix + "." + str + ".dat";
      this->importSensor(sensorname);
    }
  }
  else {
    this->importSensorOneFile(sensorprefix);
  }
  
}



data::data(const std::string & name_, const int & spaceDim_, const std::string & ptsfile,
           const std::string & sensorprefix, const bool & separate_files,
           const int & Nx, const int & Ny, const int & Nz) {
  
  name = name_;
  spaceDim = spaceDim_;
  
  is_spatialdep = true;
  is_timedep = false;
  is_stochastic = false;
  
  this->importGridPoints(ptsfile, spaceDim, Nx, Ny, Nz);
  if (separate_files) {
    for (int i=0; i<numSensors; i++) {
      std::stringstream ss;
      ss << i;
      std::string str = ss.str();
      std::string sensorname = sensorprefix + "." + str + ".dat";
      this->importSensor(sensorname);
    }
  }
  else {
    this->importSensorOneFile(sensorprefix);
  }
  
  
}

/////////////////////////////////////////////////////////////////////////////

data::data(const std::string & name_, const int & spaceDim_, const std::string & ptsfile,
           const int & Nsens, const std::string & sensorprefix) {
  
  name = name_;
  spaceDim = spaceDim_;
  
  is_spatialdep = true;
  is_timedep = false;
  is_stochastic = false;
  
  this->importPoints(ptsfile, spaceDim);
  numSensors = Nsens;
  
  for (int i=0; i<numSensors; i++) {
    std::stringstream ss;
    ss << i;
    std::string str = ss.str();
    std::string sensorname = sensorprefix + "." + str + ".dat";
    this->importSensor(sensorname);
  }
  
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

void data::importPoints(const std::string & ptsfile, const int & spaceDim) {
  
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
  numSensors = xvec.size();
  
  sensorlocations = Kokkos::View<ScalarT**,HostDevice>("sensor locartions",numSensors,spaceDim);
  for (int i=0; i<numSensors; i++) {
    if (spaceDim >0)
      sensorlocations(i,0) = xvec[i];
    if (spaceDim >1)
      sensorlocations(i,1) = yvec[i];
    if (spaceDim >2)
      sensorlocations(i,2) = zvec[i];
  }
  
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

void data::importGridPoints(const std::string & ptsfile, const int & spaceDim,
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
  numSensors = xvec.size();
  
  if (spaceDim == 1) {
    sensorGrid_x = Kokkos::View<ScalarT*,HostDevice>("sensor grid x pts",Nx);
    for (int k=0; k<Nx; k++) {
      sensorGrid_x(k) = xvec[k];
    }
  }
  else if (spaceDim == 2) {
    sensorGrid_x = Kokkos::View<ScalarT*,HostDevice>("sensor grid x pts",Nx);
    for (int k=0; k<Nx; k++) {
      sensorGrid_x(k) = xvec[k];
    }
    
    sensorGrid_y = Kokkos::View<ScalarT*,HostDevice>("sensor grid y pts",Ny);
    for (int k=0; k<Ny; k++) {
      sensorGrid_y(k) = yvec[(k)*Nx];
    }
  }
  else if (spaceDim == 3) {
    sensorGrid_x = Kokkos::View<ScalarT*,HostDevice>("sensor grid x pts",Nx);
    for (int k=0; k<Nx; k++) {
      sensorGrid_x(k) = xvec[k];
    }
    
    sensorGrid_y = Kokkos::View<ScalarT*,HostDevice>("sensor grid y pts",Ny);
    for (int k=0; k<Ny; k++) {
      sensorGrid_y(k) = yvec[(k)*Nx];
    }
    
    sensorGrid_z = Kokkos::View<ScalarT*,HostDevice>("sensor grid z pts",Nz);
    for (int k=0; k<Nz; k++) {
      sensorGrid_z(k) = zvec[(k)*Nx*Ny];
    }
  }
  
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

void data::importSensorOneFile(const std::string & sensorfile) {
  
  Teuchos::TimeMonitor timer(*dataImportTimer);
  
  std::ifstream fnmast(sensorfile.c_str());
  if (!fnmast.good()) {
    TEUCHOS_TEST_FOR_EXCEPTION(!fnmast.good(),std::runtime_error,"Error: could not find the sensor data file: " + sensorfile);
  }
  
  
  std::vector<std::vector<ScalarT> > values;
  std::ifstream fin(sensorfile.c_str());
  
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
    sensordata.push_back(newdata);
  }
  
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

void data::importSensor(const std::string & sensorfile) {
  
  Teuchos::TimeMonitor timer(*dataImportTimer);
  
  std::ifstream fnmast(sensorfile.c_str());
  if (!fnmast.good()) {
    TEUCHOS_TEST_FOR_EXCEPTION(!fnmast.good(),std::runtime_error,"Error: could not find the sensor data file: " + sensorfile);
  }
  
  std::vector<std::vector<ScalarT> > values;
  std::ifstream fin(sensorfile.c_str());
  
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
  
  sensordata.push_back(newdata);
}


/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

ScalarT data::getvalue(const ScalarT & x, const ScalarT & y, const ScalarT & z,
                       const ScalarT & time, const std::string & label) const {
  
  Teuchos::TimeMonitor timer(*dataValueTimer);
  
  // find the closest point
  ScalarT val = 0.0;
  if (is_spatialdep) {
    int cnode = this->findClosestNode(x, y, z);
    //cout << "cnode = " << cnode << endl;
    
    Kokkos::View<ScalarT**,HostDevice> sdata = sensordata[cnode];
    
    //cout << "sdata = " << sdata << endl;
    //std::vector<string> slabels = sensorlabels[cnode];
    int index = 0;
    //for (size_t j=0; j<slabels.size(); j++) {
    //   if (slabels[j] == label)
    //      index = j;
    //}
    int timeindex = 0;
    //for (size_t j=0; j<slabels.size(); j++) {
    //   if (slabels[j] == "t")
    //      timeindex = j;
    //}
    if (is_timedep) {
      size_t tn = 0;
      bool found = false;
      while (!found && tn<sdata.extent(0)) {
        if (time>=sdata(tn,timeindex) && time<=sdata(tn+1,timeindex))
          found = true;
        else
          tn += 1;
      }
      if (!found)
        val = sdata(sdata.extent(0)-1,index);
      else {
        ScalarT alpha = (sdata(tn+1,timeindex)-time)/(sdata(tn+1,timeindex)-sdata(tn,timeindex));
        val = alpha*sdata(tn,index) + (1.0-alpha)*sdata(tn+1,index);
      }
      
    }
    else
      val = sdata(0,index);
  }
  
  return val;
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

int data::findClosestNode(const ScalarT & x, const ScalarT & y, const ScalarT & z) const {
  
  Teuchos::TimeMonitor timer(*dataClosestTimer);
  
  int node = 0;
  ScalarT dist = (ScalarT)RAND_MAX;
  
  if (spaceDim == 1) {
    for( int i=0; i<numSensors; i++ ) {
      ScalarT xhat = sensorlocations(i,0);
      ScalarT d = (x-xhat)*(x-xhat);
      if( d<dist ) {
        node = i;
        dist = d;
      }
    }
  }
  if (spaceDim == 2) {
    for( int i=0; i<numSensors; i++ ) {
      ScalarT xhat = sensorlocations(i,0);
      ScalarT yhat = sensorlocations(i,1);
      ScalarT d = (x-xhat)*(x-xhat) + (y-yhat)*(y-yhat);
      if( d<dist ) {
        node = i;
        dist = d;
      }
    }
  }
  if (spaceDim == 3) {
    for( int i=0; i<numSensors; i++ ) {
      ScalarT xhat = sensorlocations(i,0);
      ScalarT yhat = sensorlocations(i,1);
      ScalarT zhat = sensorlocations(i,2);
      ScalarT d = (x-xhat)*(x-xhat) + (y-yhat)*(y-yhat) + (z-zhat)*(z-zhat);
      if( d<dist ) {
        node = i;
        dist = d;
      }
    }
  }
  return node;
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

int data::findClosestNode(const ScalarT & x, const ScalarT & y, const ScalarT & z, ScalarT & distance) const {
  
  Teuchos::TimeMonitor timer(*dataClosestTimer);
  
  int node = 0;
  ScalarT dist = (ScalarT)RAND_MAX;
  
  if (spaceDim == 1) {
    for( int i=0; i<numSensors; i++ ) {
      ScalarT xhat = sensorlocations(i,0);
      ScalarT d = (x-xhat)*(x-xhat);
      if( d<dist ) {
        node = i;
        dist = d;
      }
    }
  }
  if (spaceDim == 2) {
    for( int i=0; i<numSensors; i++ ) {
      ScalarT xhat = sensorlocations(i,0);
      ScalarT yhat = sensorlocations(i,1);
      ScalarT d = (x-xhat)*(x-xhat) + (y-yhat)*(y-yhat);
      if( d<dist ) {
        node = i;
        dist = d;
      }
    }
  }
  if (spaceDim == 3) {
    for( int i=0; i<numSensors; i++ ) {
      ScalarT xhat = sensorlocations(i,0);
      ScalarT yhat = sensorlocations(i,1);
      ScalarT zhat = sensorlocations(i,2);
      ScalarT d = (x-xhat)*(x-xhat) + (y-yhat)*(y-yhat) + (z-zhat)*(z-zhat);
      if( d<dist ) {
        node = i;
        dist = d;
      }
    }
  }
  distance = sqrt(dist);
  return node;
}


/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

int data::findClosestGridNode(const ScalarT & x, const ScalarT & y, const ScalarT & z, ScalarT & distance) const {
  
  Teuchos::TimeMonitor timer(*dataClosestTimer);
  
  int node = 0;
  
  if (spaceDim == 1) {
    ScalarT dist = (ScalarT)RAND_MAX;
    int node_x = 0;
    for(size_type i=0; i<sensorGrid_x.extent(0); i++ ) {
      ScalarT xhat = sensorGrid_x(i);
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
    for(size_type i=0; i<sensorGrid_x.extent(0); i++ ) {
      ScalarT xhat = sensorGrid_x(i);
      ScalarT d = (x-xhat)*(x-xhat);
      if( d<dist_x ) {
        node_x = i;
        dist_x = d;
      }
    }
    for(size_type i=0; i<sensorGrid_y.extent(0); i++ ) {
      ScalarT yhat = sensorGrid_y(i);
      ScalarT d = (y-yhat)*(y-yhat);
      if( d<dist_y ) {
        node_y = i;
        dist_y = d;
      }
    }
    node = node_y*sensorGrid_x.extent(0) + node_x;
    distance = sqrt(dist_x + dist_y);
  }
  if (spaceDim == 3) {
    ScalarT dist_x = (ScalarT)RAND_MAX;
    ScalarT dist_y = (ScalarT)RAND_MAX;
    ScalarT dist_z = (ScalarT)RAND_MAX;
    int node_x=0, node_y=0, node_z=0;
    for(size_type i=0; i<sensorGrid_x.extent(0); i++ ) {
      ScalarT xhat = sensorGrid_x(i);
      ScalarT d = (x-xhat)*(x-xhat);
      if( d<dist_x ) {
        node_x = i;
        dist_x = d;
      }
    }
    for(size_type i=0; i<sensorGrid_y.extent(0); i++ ) {
      ScalarT yhat = sensorGrid_y(i);
      ScalarT d = (y-yhat)*(y-yhat);
      if( d<dist_y ) {
        node_y = i;
        dist_y = d;
      }
    }
    for(size_type i=0; i<sensorGrid_z.extent(0); i++ ) {
      ScalarT zhat = sensorGrid_z(i);
      ScalarT d = (z-zhat)*(z-zhat);
      if( d<dist_z ) {
        node_z = i;
        dist_z = d;
      }
    }
    node = node_z*sensorGrid_x.extent(0)*sensorGrid_y.extent(0) + node_y*sensorGrid_x.extent(0) + node_x;
    distance = sqrt(dist_x + dist_y + dist_z);
    
  }
  
  return node;
  
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

std::string data::getname() {
  return name;
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

std::vector<Kokkos::View<ScalarT**,HostDevice> > data::getdata() {
  return sensordata;
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

Kokkos::View<ScalarT**,HostDevice> data::getdata(const int & sensnum) {
  return sensordata[sensnum];
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

Kokkos::View<ScalarT**,HostDevice> data::getpoints() {
  return sensorlocations;
}
