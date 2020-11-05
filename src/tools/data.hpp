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

#ifndef DATA_H
#define DATA_H

#include "trilinos.hpp"
#include <iostream>     
#include <iterator>     
#include "CompadreTools.hpp"

namespace MrHyDE {
  
  class data {
  public:
    
    data() {} ;
    
    /////////////////////////////////////////////////////////////////////////////
    //  Various constructors depending on the characteristics of the data (spatial,
    //  transient, stochastic, etc.)
    /////////////////////////////////////////////////////////////////////////////
    
    data(const std::string & name_, const ScalarT & val);
    
    /////////////////////////////////////////////////////////////////////////////
    
    data(const std::string & name_, const std::string & datafile);
    
    /////////////////////////////////////////////////////////////////////////////
    
    data(const std::string & name_, const int & spaceDim_, const std::string & ptsfile);
    
    /////////////////////////////////////////////////////////////////////////////
    
    data(const std::string & name_, const int & spaceDim_, const std::string & ptsfile,
         const std::string & sensorprefix);
    
    /////////////////////////////////////////////////////////////////////////////
    
    data(const std::string & name_, const int & spaceDim_, const std::string & ptsfile,
         const std::string & sensorprefix, const bool & separate_files);
    
    
    
    data(const std::string & name_, const int & spaceDim_, const std::string & ptsfile,
         const std::string & sensorprefix, const bool & separate_files,
         const int & Nx, const int & Ny, const int & Nz);
    
    /////////////////////////////////////////////////////////////////////////////
    
    data(const std::string & name_, const int & spaceDim_, const std::string & ptsfile,
         const int & Nsens, const std::string & sensorprefix);
    
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    
    void importPoints(const std::string & ptsfile, const int & spaceDim);
    
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    
    void importGridPoints(const std::string & ptsfile, const int & spaceDim,
                          const int & Nx, const int & Ny, const int & Nz);
    
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    
    void importSensorOneFile(const std::string & sensorfile);
    
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    
    void importSensor(const std::string & sensorfile);
    
    
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    
    ScalarT getvalue(const ScalarT & x, const ScalarT & y, const ScalarT & z,
                     const ScalarT & time, const string & label) const;
    
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////

    void findClosestNode(const Kokkos::View<ScalarT**, AssemblyDevice> &coords, 
                         Kokkos::View<int*, AssemblyDevice> &cnode) const;
    
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    
    void findClosestNode(const Kokkos::View<ScalarT**, AssemblyDevice> &coords, 
                         Kokkos::View<int*, AssemblyDevice> &cnode, 
                         Kokkos::View<ScalarT*, AssemblyDevice> &distance) const;
    
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    
    int findClosestGridNode(const ScalarT & x, const ScalarT & y, const ScalarT & z, ScalarT & distance) const;
    
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    
    std::string getname();
    
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    
    std::vector<Kokkos::View<ScalarT**,HostDevice> > getdata();
    
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    
    Kokkos::View<ScalarT**,HostDevice> getdata(const int & sensnum);
    
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    
    Kokkos::View<ScalarT**,HostDevice> getpoints();
    
  protected:
    
    bool is_spatialdep;
    bool is_timedep;
    bool is_stochastic;
    
    int spaceDim, numSensors;
    std::string name;
    
    Kokkos::View<ScalarT**,HostDevice> sensorlocations;
    std::vector<Kokkos::View<ScalarT**,HostDevice> > sensordata;
    std::vector<std::vector<string> > sensorlabels;
    Kokkos::View<ScalarT*,HostDevice> sensorGrid_x, sensorGrid_y, sensorGrid_z;
    
    // Profile timers
    Teuchos::RCP<Teuchos::Time> dataImportTimer = Teuchos::TimeMonitor::getNewCounter("MILO::data - import data");
    Teuchos::RCP<Teuchos::Time> pointImportTimer = Teuchos::TimeMonitor::getNewCounter("MILO::data - import points");
    Teuchos::RCP<Teuchos::Time> dataClosestTimer = Teuchos::TimeMonitor::getNewCounter("MILO::data::findClosestNode()");
    Teuchos::RCP<Teuchos::Time> dataValueTimer = Teuchos::TimeMonitor::getNewCounter("MILO::data::getValue()");
    
  };
}

#endif
