/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "split_mpi_communicators.hpp"


SplitComm::SplitComm(Teuchos::RCP<Teuchos::ParameterList> & settings,
          LA_MpiComm & Comm,
          Teuchos::RCP<LA_MpiComm> & tcomm_LA,
          Teuchos::RCP<LA_MpiComm> & tcomm_S) {
  
  
  string analysis_type = settings->sublist("Analysis").get<string>("analysis type","forward");
  bool ms_split_comm = settings->sublist("Analysis").get<bool>("multiscale split comm",false);
  int numLA = Comm.getSize();
  int numGroups = 1;
  int procsPerGroup = numLA;
  if (analysis_type == "SOL"){
    numLA = settings->sublist("Analysis").get<int>("Number of LA processors",numLA);
    numGroups = Comm.getSize()/numLA;
    if (Comm.getSize()%numLA != 0){
      cout << "\n NUMBER OF LINEAR ALGEBRA PROCESSORS NEEDS TO BE A FACTOR OF TOTAL NUMBER OF PROCESSORS..." << endl;
      numLA = Comm.getSize();
    }
    split_mpi_communicators(tcomm_LA, tcomm_S, Comm.getRank(), numLA, numGroups);
  }
  else if (ms_split_comm) {
    numLA = settings->sublist("Analysis").get<int>("Number of macro processors",numLA);
    numGroups = numLA;
    procsPerGroup = Comm.getSize()/numLA;
    if (Comm.getSize()%numLA != 0){
      cout << "\n NUMBER OF MACRO PROCESSORS NEEDS TO BE A FACTOR OF TOTAL NUMBER OF PROCESSORS..." << endl;
      numLA = Comm.getSize();
      procsPerGroup = 1;
    }
    bool include_main = true;
    this->split_mpi_communicators(Comm, tcomm_LA, tcomm_S, Comm.getRank(), numLA, include_main);
  }
  else {
    this->split_mpi_communicators(tcomm_LA, tcomm_S, Comm.getRank(), numLA, numGroups);
  }
  Comm.barrier();
  if (analysis_type == "SOL"){ // TMW: Only printing this out if SOL is active
    cout << "MPI WORLD RANK: " << Comm.getRank()
    << "  LINEAR ALGEBRA RANK: " << tcomm_LA->getRank()
    << "  SAMPLING RANK: " << tcomm_S->getRank() << endl;
  }
  
}

void SplitComm::split_mpi_communicators( Teuchos::RCP<LA_MpiComm> & Comm_linalg,
                              Teuchos::RCP<LA_MpiComm> & Comm_collocation,
                              int rank, int M, int Ngroups ) {
  // Instantiate Linear Algebra Communicator
  MPI_Comm linalg_comm;
  MPI_Comm_split(MPI_COMM_WORLD,rank/M,rank,&linalg_comm);
  int comRank; // Process rank in linear algebra communicator
  int comSize; // Number of processes in linear algebra communicator
  MPI_Comm_rank(linalg_comm,&comRank);     // Get Process rank
  MPI_Comm_size(linalg_comm,&comSize);     // Get Communicator size
  Comm_linalg = Teuchos::rcp( new LA_MpiComm(linalg_comm) );

  // Determine group ranks for Collocation Distribution
  Teuchos::Array<int> granks(Ngroups);
  for (int i=0;i<Ngroups;i++) granks[i] = comRank + i*M;

  // Build MPI groups for collocation distribution
  MPI_Group world_comm; // Grab MPI_COMM_WORLD and place in world_comm
  MPI_Comm_group(MPI_COMM_WORLD,&world_comm);
  MPI_Group group;
  MPI_Group_incl(world_comm,Ngroups,&granks[0],&group);

  // Instantiate Collocation Communicator based on group 
  MPI_Comm collocation_comm;
  MPI_Comm_create(MPI_COMM_WORLD, group, &collocation_comm);
  int comRank1; // Process rank in collocation communicator
  int comSize1; // Number of processes in collocation communicator
  MPI_Comm_rank(collocation_comm,&comRank1);         // Get Process rank
  MPI_Comm_size(collocation_comm,&comSize1);         // Get Communicator size
  Comm_collocation = Teuchos::rcp( new LA_MpiComm(collocation_comm) );
}

void SplitComm::split_mpi_communicators(LA_MpiComm & Comm_orig,
                             Teuchos::RCP<LA_MpiComm> & Comm_main,
                             Teuchos::RCP<LA_MpiComm> & Comm_ms,
                             int myrank, int Nmain , bool & include_main) {
  
  int procsPerGroup = 1;
  int numGroups = Nmain;
  int numOrigProcs = Comm_orig.getSize();
  if (numOrigProcs%Nmain != 0){
    cout << "\n NUMBER OF MACRO PROCESSORS NEEDS TO BE A FACTOR OF TOTAL NUMBER OF PROCESSORS..." << endl;
    Nmain = numOrigProcs;
  }
  else {
    if (include_main) {
      procsPerGroup = numOrigProcs/Nmain;
    }
    else {
      procsPerGroup = numOrigProcs/Nmain - 1;
    }
  }
  
  // First, create main comm group
  MPI_Group world_comm;
  MPI_Comm_group(*(Comm_orig.getRawMpiComm()),&world_comm);
  MPI_Group main_group;
  
  Teuchos::Array<int> granks(Nmain);
  int prog = 0;
  for (int i=0;i<numOrigProcs;i++) {
    if (i%procsPerGroup == 0) {
      granks[prog] = i;
      prog++;
    }
  }
  MPI_Group_incl(world_comm,Nmain,&granks[0],&main_group);
  MPI_Comm main_comm;
  MPI_Comm_create(*(Comm_orig.getRawMpiComm()), main_group, &main_comm);
  Comm_main = Teuchos::rcp( new LA_MpiComm(main_comm) );
  
  cout << "got to here" << endl;
  
  // Next, create Nmain smaller groups
  MPI_Group ms_group;
  
  cout << myrank << endl;
  cout << myrank%procsPerGroup << endl;
  
  /*
   Teuchos::Array<int> msranks(procsPerGroup);
  if (myrank%procsPerGroup == 0) {
    for (int i=0;i<procsPerGroup;i++) {
      msranks[i] = i;
    }
  }
  MPI_Group_incl(world_comm,procsPerGroup,&msranks[0],&ms_group);
  MPI_Comm ms_comm;
  MPI_Comm_create(Comm_orig.Comm(), ms_group, &ms_comm);
  Comm_ms = Teuchos::rcp( new LA_MpiComm(ms_comm) );
  */
}
