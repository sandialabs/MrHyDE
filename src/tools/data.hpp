/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef DATA_H
#define DATA_H

#include "trilinos.hpp"
#include <iostream>     
#include <iterator>     

using namespace std;

class data {
public:
  
  data() {} ;
  
  /////////////////////////////////////////////////////////////////////////////
  //  Various constructors depending on the characteristics of the data (spatial, 
  //  transient, stochastic, etc.)
  /////////////////////////////////////////////////////////////////////////////
  
  data(const std::string & name_, const ScalarT & val) {
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
  
  
  data(const std::string & name_, const std::string & datafile) {
    
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
  
  data(const std::string & name_, const int & spaceDim_, const std::string & ptsfile) {
    
    name = name_;
    spaceDim = spaceDim_;
    
    is_spatialdep = true;
    is_timedep = false;
    is_stochastic = false;
    
    this->importPoints(ptsfile, spaceDim);
  }
  
  /////////////////////////////////////////////////////////////////////////////
  
  data(const std::string & name_, const int & spaceDim_, const std::string & ptsfile, 
       const std::string & sensorprefix) {
    
    name = name_;
    spaceDim = spaceDim_;
    
    is_spatialdep = true;
    is_timedep = false;
    is_stochastic = false;
    
    this->importPoints(ptsfile, spaceDim);
    
    for (size_t i=0; i<numSensors; i++) {
      stringstream ss;
      ss << i;
      std::string str = ss.str();
      std::string sensorname = sensorprefix + "." + str + ".dat";
      this->importSensor(sensorname);
    }
    
  }
  
  /////////////////////////////////////////////////////////////////////////////
  
  data(const std::string & name_, const int & spaceDim_, const std::string & ptsfile, 
       const std::string & sensorprefix, const bool & separate_files) {
    
    name = name_;
    spaceDim = spaceDim_;
    
    is_spatialdep = true;
    is_timedep = false;
    is_stochastic = false;
    
    this->importPoints(ptsfile, spaceDim);
    if (separate_files) {
      for (size_t i=0; i<numSensors; i++) {
        stringstream ss;
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
  
  data(const std::string & name_, const int & spaceDim_, const std::string & ptsfile, 
       const int & Nsens, const std::string & sensorprefix) {
    
    name = name_;
    spaceDim = spaceDim_;
    
    is_spatialdep = true;
    is_timedep = false;
    is_stochastic = false;
    
    this->importPoints(ptsfile, spaceDim);
    numSensors = Nsens;
    
    for (size_t i=0; i<numSensors; i++) {
      stringstream ss;
      ss << i;
      std::string str = ss.str();
      std::string sensorname = sensorprefix + "." + str + ".dat";
      this->importSensor(sensorname);
    }
    
  }
  
  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  
  void importPoints(const std::string & ptsfile, const int & spaceDim) {
    
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
    for (size_t i=0; i<numSensors; i++) {
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
  
  void importSensorOneFile(const std::string & sensorfile) {
    
    
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
  
  void importSensor(const std::string & sensorfile) {
    
    
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
  
  ScalarT getvalue(const ScalarT & x, const ScalarT & y, const ScalarT & z,
                  const ScalarT & time, const string & label) const {
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
  
  int findClosestNode(const ScalarT & x, const ScalarT & y, const ScalarT & z) const {
    int node = 0;
    ScalarT dist = (ScalarT)RAND_MAX;
    
    if (spaceDim == 1) {
      for( int i=0; i<numSensors; i++ ) {
        ScalarT xhat = sensorlocations(i,0);
        ScalarT d = sqrt((x-xhat)*(x-xhat));
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
        ScalarT d = sqrt((x-xhat)*(x-xhat) + (y-yhat)*(y-yhat));
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
        ScalarT d = sqrt((x-xhat)*(x-xhat) + (y-yhat)*(y-yhat) + (z-zhat)*(z-zhat));
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
  
  int findClosestNode(const ScalarT & x, const ScalarT & y, const ScalarT & z, ScalarT & distance) const {
    int node = 0;
    ScalarT dist = (ScalarT)RAND_MAX;
    
    if (spaceDim == 1) {
      for( int i=0; i<numSensors; i++ ) {
        ScalarT xhat = sensorlocations(i,0);
        ScalarT d = sqrt((x-xhat)*(x-xhat));
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
        ScalarT d = sqrt((x-xhat)*(x-xhat) + (y-yhat)*(y-yhat));
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
        ScalarT d = sqrt((x-xhat)*(x-xhat) + (y-yhat)*(y-yhat) + (z-zhat)*(z-zhat));
        if( d<dist ) {
          node = i;
          dist = d;
        }
      }
    }
    distance = dist;
    return node;
  }

  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  
  std::string getname() {
    return name;
  }
  
  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  
  std::vector<Kokkos::View<ScalarT**,HostDevice> > getdata() {
    return sensordata;
  }
  
  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  
  Kokkos::View<ScalarT**,HostDevice> getdata(const int & sensnum) {
    return sensordata[sensnum];
  }
  
  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  
  Kokkos::View<ScalarT**,HostDevice> getpoints() {
    return sensorlocations;
  }
  
protected:
  
  bool is_spatialdep;
  bool is_timedep;
  bool is_stochastic;
  
  int spaceDim, numSensors;
  std::string name;
  //vector<float> xvec, yvec, zvec, times;
  //FC mydata;
  
  Kokkos::View<ScalarT**,HostDevice> sensorlocations;
  //std::vector<FC > sensortimes;
  std::vector<Kokkos::View<ScalarT**,HostDevice> > sensordata;
  std::vector<std::vector<string> > sensorlabels;
  
};

#endif
