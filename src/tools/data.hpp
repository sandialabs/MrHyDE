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
  
  class Data {
  public:
    
    Data() {} ;
    
    ~Data() {} ;
    
    /////////////////////////////////////////////////////////////////////////////
    //  Various constructors depending on the characteristics of the data (spatial,
    //  transient, stochastic, etc.)
    /////////////////////////////////////////////////////////////////////////////
    
    Data(const std::string & name_, const ScalarT & val);
    
    /////////////////////////////////////////////////////////////////////////////
    
    Data(const std::string & name_, const std::string & datafile);
    
    /////////////////////////////////////////////////////////////////////////////
    
    Data(const std::string & name_, const int & spaceDim_, const std::string & ptsfile);
    
    /////////////////////////////////////////////////////////////////////////////
    
    Data(const std::string & name_, const int & spaceDim_, const std::string & ptsfile,
         const std::string & sensorprefix);
    
    /////////////////////////////////////////////////////////////////////////////
    
    Data(const std::string & name_, const int & spaceDim_, const std::string & ptsfile,
         const std::string & sensorprefix, const bool & separate_files);
    
    /////////////////////////////////////////////////////////////////////////////
    
    Data(const std::string & name_, const int & spaceDim_, const std::string & ptsfile,
         const std::string & sensorprefix, const bool & separate_files,
         const int & Nx, const int & Ny, const int & Nz);
    
    /////////////////////////////////////////////////////////////////////////////
    
    Data(const std::string & name_, const int & spaceDim_, const std::string & ptsfile,
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
    
    void importDataOneFile(const std::string & datafile);
    
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    
    void importData(const std::string & datafile);
        
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    
    ScalarT getValue(const ScalarT & x, const ScalarT & y, const ScalarT & z,
                     const ScalarT & time, const string & label) const;
    
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////

    void findClosestPoint(const Kokkos::View<ScalarT**, AssemblyDevice> &testpts,
                          Kokkos::View<int*, CompadreDevice> &closestpts) const;
    
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    
    void findClosestPoint(const Kokkos::View<ScalarT**, AssemblyDevice> &testpts,
                          Kokkos::View<int*, CompadreDevice> &closestpts,
                          Kokkos::View<ScalarT*, AssemblyDevice> &distance) const;
    
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    
    int findClosestGridPoint(const ScalarT & x, const ScalarT & y,
                             const ScalarT & z, ScalarT & distance) const;
    
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    
    std::string getName();
    
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    
    std::vector<Kokkos::View<ScalarT**,HostDevice> > getData();
    
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    
    Kokkos::View<ScalarT**,HostDevice> getData(const int & sensnum);
    
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    
    Kokkos::View<ScalarT**,HostDevice> getPoints();
    
  protected:
    
    bool is_spatialdep;
    bool is_timedep;
    bool is_stochastic;
    
    int spaceDim;
    std::string name;
    
    Kokkos::View<ScalarT**,HostDevice> points;
    std::vector<Kokkos::View<ScalarT**,HostDevice> > data;
    std::vector<std::vector<string> > labels;
    Kokkos::View<ScalarT*,HostDevice> grid_x, grid_y, grid_z;
    
    // Profile timers
    Teuchos::RCP<Teuchos::Time> dataImportTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::data - import data");
    Teuchos::RCP<Teuchos::Time> pointImportTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::data - import points");
    Teuchos::RCP<Teuchos::Time> dataClosestTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::data::findClosestNode()");
    Teuchos::RCP<Teuchos::Time> dataValueTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::data::getValue()");
    
  };
}

#endif
