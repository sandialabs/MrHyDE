# Teuchos::TimeMonitor report
---
Output mode: spacious
Number of processes: 4
Time unit: s
Statistics collected: 
  - MinOverProcs
  - MeanOverProcs
  - MaxOverProcs
  - MeanOverCallCounts
Timer names: 
  - "Belos: BlockGmresSolMgr total solve time"
  - "Belos: DGKS[2]: Ortho (Inner Product)"
  - "Belos: DGKS[2]: Ortho (Norm)"
  - "Belos: DGKS[2]: Ortho (Update)"
  - "Belos: DGKS[2]: Orthogonalization"
  - "Belos: Operation Op*x"
  - "Belos: Operation Prec*x"
  - "Ifpack2::Chebyshev::apply"
  - "Ifpack2::Chebyshev::compute"
  - "MILO::assembly::computeJacRes() - gather"
  - "MILO::assembly::computeJacRes() - insert"
  - "MILO::assembly::computeJacRes() - physics evaluation"
  - "MILO::assembly::computeJacRes() - total assembly"
  - "MILO::assembly::createCells()"
  - "MILO::assembly::createWorkset()"
  - "MILO::assembly::dofConstraints()"
  - "MILO::assembly::setDirichlet()"
  - "MILO::assembly::setInitial()"
  - "MILO::boundaryCell - build basis"
  - "MILO::cell::computeJacRes() - fill local Jacobian"
  - "MILO::cell::computeJacRes() - fill local residual"
  - "MILO::cell::computeJacRes() - volume residual"
  - "MILO::cell::computeSolAvg()"
  - "MILO::cell::computeSolnFaceIP()"
  - "MILO::cell::computeSolnVolIP()"
  - "MILO::cell::constructor - build basis"
  - "MILO::cell::constructor - build face basis"
  - "MILO::cellMetaData::constructor()"
  - "MILO::driver::total run time"
  - "MILO::driver::total setup and execution time"
  - "MILO::function::decompose"
  - "MILO::function::evaluate"
  - "MILO::physics::getSideInfo()"
  - "MILO::physics::setBCData()"
  - "MILO::physics::setDirichletData()"
  - "MILO::postprocess::computeError"
  - "MILO::postprocess::writeSolution"
  - "MILO::solver::linearSolver()"
  - "MILO::solver::projectDirichlet()"
  - "MILO::solver::setDirichlet()"
  - "MILO::solver::setInitial()"
  - "MILO::solver::setupFixedDOFs()"
  - "MILO::solver::setupLinearAlgebra()"
  - "MILO::thermal::volumeResidual() - evaluation of residual"
  - "MILO::thermal::volumeResidual() - function evaluation"
  - "MILO::workset::computeSolnVolIP - allocate/compute seeded"
  - "MILO::workset::computeSolnVolIP - compute seeded sol at ip"
  - "MILO::workset::reset*"
  - "MueLu: ParameterListInterpreter (ParameterList)"
  - "STK_Interface::setupExodusFile(filename)"
  - "STK_Interface::writeToExodus(timestep)"
  - "UtilitiesBase::GetMatrixDiagonalInverse"
  - "panzer::DOFManager::buildGlobalUnknowns"
  - "panzer::DOFManager::buildGlobalUnknowns::build_ghosted_array"
  - "panzer::DOFManager::buildGlobalUnknowns::build_local_ids"
  - "panzer::DOFManager::buildGlobalUnknowns::build_orientation"
  - "panzer::DOFManager::buildGlobalUnknowns::build_owned_vector"
  - "panzer::DOFManager::buildGlobalUnknowns_GUN"
  - "panzer::DOFManager::buildGlobalUnknowns_GUN::line_04 createOneToOne"
  - "panzer::DOFManager::buildGlobalUnknowns_GUN::line_05 alloc_unique_mv"
  - "panzer::DOFManager::buildGlobalUnknowns_GUN::line_06 export"
  - "panzer::DOFManager::buildGlobalUnknowns_GUN::line_07-09 local_count"
  - "panzer::DOFManager::buildGlobalUnknowns_GUN::line_10 prefix_sum"
  - "panzer::DOFManager::buildGlobalUnknowns_GUN::line_13-21 gid_assignment"
  - "panzer::DOFManager::buildGlobalUnknowns_GUN::line_23 final_import"
  - "panzer::DOFManager::buildTaggedMultiVector"
  - "panzer::DOFManager::buildTaggedMultiVector::allocate_tagged_multivector"
  - "panzer::DOFManager::buildTaggedMultiVector::fill_tagged_multivector"
  - "panzer::DOFManager::builderOverlapMapFromElements"
  - "panzer::SquareQuadMeshFactory::buildUncomittedMesh()"
  - "panzer::SquareQuadMeshFactory::completeMeshConstruction()"
Total times: 
  "Belos: BlockGmresSolMgr total solve time": 
    MinOverProcs: 0.00699496
    MeanOverProcs: 0.00705254
    MaxOverProcs: 0.00711203
    MeanOverCallCounts: 0.00235085
  "Belos: DGKS[2]: Ortho (Inner Product)": 
    MinOverProcs: 0.000282049
    MeanOverProcs: 0.000283182
    MaxOverProcs: 0.000283957
    MeanOverCallCounts: 1.28719e-05
  "Belos: DGKS[2]: Ortho (Norm)": 
    MinOverProcs: 0.000187874
    MeanOverProcs: 0.000301898
    MaxOverProcs: 0.000407219
    MeanOverCallCounts: 8.38604e-06
  "Belos: DGKS[2]: Ortho (Update)": 
    MinOverProcs: 0.000168324
    MeanOverProcs: 0.00018537
    MaxOverProcs: 0.000198841
    MeanOverCallCounts: 8.42593e-06
  "Belos: DGKS[2]: Orthogonalization": 
    MinOverProcs: 0.000804901
    MeanOverProcs: 0.000897348
    MaxOverProcs: 0.000971556
    MeanOverCallCounts: 6.40963e-05
  "Belos: Operation Op*x": 
    MinOverProcs: 0.000543833
    MeanOverProcs: 0.00122356
    MaxOverProcs: 0.00186849
    MeanOverCallCounts: 6.79758e-05
  "Belos: Operation Prec*x": 
    MinOverProcs: 0.0090723
    MeanOverProcs: 0.00921148
    MaxOverProcs: 0.00958085
    MeanOverCallCounts: 0.000657963
  "Ifpack2::Chebyshev::apply": 
    MinOverProcs: 0.0016706
    MeanOverProcs: 0.00172842
    MaxOverProcs: 0.00177693
    MeanOverCallCounts: 6.17291e-05
  "Ifpack2::Chebyshev::compute": 
    MinOverProcs: 0.00231409
    MeanOverProcs: 0.00356179
    MaxOverProcs: 0.00422525
    MeanOverCallCounts: 0.00118726
  "MILO::assembly::computeJacRes() - gather": 
    MinOverProcs: 3.98159e-05
    MeanOverProcs: 4.1008e-05
    MaxOverProcs: 4.22001e-05
    MeanOverCallCounts: 2.0504e-05
  "MILO::assembly::computeJacRes() - insert": 
    MinOverProcs: 0.000146389
    MeanOverProcs: 0.000152767
    MaxOverProcs: 0.000161648
    MeanOverCallCounts: 7.63834e-06
  "MILO::assembly::computeJacRes() - physics evaluation": 
    MinOverProcs: 0.00596023
    MeanOverProcs: 0.0059787
    MaxOverProcs: 0.00600433
    MeanOverCallCounts: 0.000298935
  "MILO::assembly::computeJacRes() - total assembly": 
    MinOverProcs: 0.00619507
    MeanOverProcs: 0.00621724
    MaxOverProcs: 0.00624299
    MeanOverCallCounts: 0.00310862
  "MILO::assembly::createCells()": 
    MinOverProcs: 0.00610995
    MeanOverProcs: 0.0062955
    MaxOverProcs: 0.00643706
    MeanOverCallCounts: 0.0062955
  "MILO::assembly::createWorkset()": 
    MinOverProcs: 0.000167131
    MeanOverProcs: 0.000173092
    MaxOverProcs: 0.000177145
    MeanOverCallCounts: 0.000173092
  "MILO::assembly::dofConstraints()": 
    MinOverProcs: 1.16825e-05
    MeanOverProcs: 1.24574e-05
    MaxOverProcs: 1.40667e-05
    MeanOverCallCounts: 6.22869e-06
  "MILO::assembly::setDirichlet()": 
    MinOverProcs: 0.000112057
    MeanOverProcs: 0.000165999
    MaxOverProcs: 0.000201941
    MeanOverCallCounts: 0.000165999
  "MILO::assembly::setInitial()": 
    MinOverProcs: 0.000700951
    MeanOverProcs: 0.000715017
    MaxOverProcs: 0.000725985
    MeanOverCallCounts: 0.000715017
  "MILO::boundaryCell - build basis": 
    MinOverProcs: 8.51154e-05
    MeanOverProcs: 0.000130773
    MaxOverProcs: 0.000179052
    MeanOverCallCounts: 5.2309e-05
  "MILO::cell::computeJacRes() - fill local Jacobian": 
    MinOverProcs: 6.48499e-05
    MeanOverProcs: 6.75321e-05
    MaxOverProcs: 7.00951e-05
    MeanOverCallCounts: 3.3766e-06
  "MILO::cell::computeJacRes() - fill local residual": 
    MinOverProcs: 3.62396e-05
    MeanOverProcs: 3.95179e-05
    MaxOverProcs: 4.26769e-05
    MeanOverCallCounts: 1.97589e-06
  "MILO::cell::computeJacRes() - volume residual": 
    MinOverProcs: 0.00265408
    MeanOverProcs: 0.00267082
    MaxOverProcs: 0.00269866
    MeanOverCallCounts: 0.000133541
  "MILO::cell::computeSolAvg()": 
    MinOverProcs: 9.13143e-05
    MeanOverProcs: 9.33409e-05
    MaxOverProcs: 9.60827e-05
    MeanOverCallCounts: 3.11136e-06
  "MILO::cell::computeSolnFaceIP()": 
    MinOverProcs: 0.00320029
    MeanOverProcs: 0.00325018
    MaxOverProcs: 0.00333762
    MeanOverCallCounts: 8.12545e-05
  "MILO::cell::computeSolnVolIP()": 
    MinOverProcs: 0.00452685
    MeanOverProcs: 0.00453591
    MaxOverProcs: 0.00454521
    MeanOverCallCounts: 0.000151197
  "MILO::cell::constructor - build basis": 
    MinOverProcs: 0.00117493
    MeanOverProcs: 0.00120974
    MaxOverProcs: 0.00125527
    MeanOverCallCounts: 0.000120974
  "MILO::cell::constructor - build face basis": 
    MinOverProcs: 0.00325394
    MeanOverProcs: 0.00329477
    MaxOverProcs: 0.00332689
    MeanOverCallCounts: 0.000329477
  "MILO::cellMetaData::constructor()": 
    MinOverProcs: 0.00068593
    MeanOverProcs: 0.000688016
    MaxOverProcs: 0.000690937
    MeanOverCallCounts: 0.000688016
  "MILO::driver::total run time": 
    MinOverProcs: 0.105886
    MeanOverProcs: 0.105942
    MaxOverProcs: 0.105972
    MeanOverCallCounts: 0.105942
  "MILO::driver::total setup and execution time": 
    MinOverProcs: 0.191244
    MeanOverProcs: 0.19141
    MaxOverProcs: 0.191561
    MeanOverCallCounts: 0.19141
  "MILO::function::decompose": 
    MinOverProcs: 0.000729084
    MeanOverProcs: 0.000751019
    MaxOverProcs: 0.000782967
    MeanOverCallCounts: 0.000751019
  "MILO::function::evaluate": 
    MinOverProcs: 0.00133014
    MeanOverProcs: 0.00136906
    MaxOverProcs: 0.00143909
    MeanOverCallCounts: 8.42498e-06
  "MILO::physics::getSideInfo()": 
    MinOverProcs: 3.00407e-05
    MeanOverProcs: 3.29018e-05
    MaxOverProcs: 3.60012e-05
    MeanOverCallCounts: 2.63214e-06
  "MILO::physics::setBCData()": 
    MinOverProcs: 0.000206947
    MeanOverProcs: 0.000221729
    MaxOverProcs: 0.000239134
    MeanOverCallCounts: 0.000221729
  "MILO::physics::setDirichletData()": 
    MinOverProcs: 5.29289e-05
    MeanOverProcs: 7.02143e-05
    MaxOverProcs: 8.79765e-05
    MeanOverCallCounts: 7.02143e-05
  "MILO::postprocess::computeError": 
    MinOverProcs: 0.00539207
    MeanOverProcs: 0.00546604
    MaxOverProcs: 0.00562692
    MeanOverCallCounts: 0.00546604
  "MILO::postprocess::writeSolution": 
    MinOverProcs: 0.00771213
    MeanOverProcs: 0.00786072
    MaxOverProcs: 0.00793791
    MeanOverCallCounts: 0.00786072
  "MILO::solver::linearSolver()": 
    MinOverProcs: 0.047626
    MeanOverProcs: 0.0476428
    MaxOverProcs: 0.0476599
    MeanOverCallCounts: 0.0238214
  "MILO::solver::projectDirichlet()": 
    MinOverProcs: 0.033663
    MeanOverProcs: 0.0337113
    MaxOverProcs: 0.0337479
    MeanOverCallCounts: 0.0337113
  "MILO::solver::setDirichlet()": 
    MinOverProcs: 1.21593e-05
    MeanOverProcs: 1.37687e-05
    MaxOverProcs: 1.50204e-05
    MeanOverCallCounts: 1.37687e-05
  "MILO::solver::setInitial()": 
    MinOverProcs: 0.021162
    MeanOverProcs: 0.0211765
    MaxOverProcs: 0.021188
    MeanOverCallCounts: 0.0211765
  "MILO::solver::setupFixedDOFs()": 
    MinOverProcs: 2.38419e-05
    MeanOverProcs: 2.56896e-05
    MaxOverProcs: 2.69413e-05
    MeanOverCallCounts: 2.56896e-05
  "MILO::solver::setupLinearAlgebra()": 
    MinOverProcs: 0.00157809
    MeanOverProcs: 0.00158048
    MaxOverProcs: 0.00158191
    MeanOverCallCounts: 0.00158048
  "MILO::thermal::volumeResidual() - evaluation of residual": 
    MinOverProcs: 0.00192046
    MeanOverProcs: 0.00193942
    MaxOverProcs: 0.0019691
    MeanOverCallCounts: 9.69708e-05
  "MILO::thermal::volumeResidual() - function evaluation": 
    MinOverProcs: 0.000691175
    MeanOverProcs: 0.000703275
    MaxOverProcs: 0.000715733
    MeanOverCallCounts: 3.51638e-05
  "MILO::workset::computeSolnVolIP - allocate/compute seeded": 
    MinOverProcs: 0.000258684
    MeanOverProcs: 0.000260293
    MaxOverProcs: 0.000261307
    MeanOverCallCounts: 8.67645e-06
  "MILO::workset::computeSolnVolIP - compute seeded sol at ip": 
    MinOverProcs: 0.00410461
    MeanOverProcs: 0.00411099
    MaxOverProcs: 0.00412393
    MeanOverCallCounts: 0.000137033
  "MILO::workset::reset*": 
    MinOverProcs: 0.000110149
    MeanOverProcs: 0.000110507
    MaxOverProcs: 0.000110865
    MeanOverCallCounts: 5.52535e-06
  "MueLu: ParameterListInterpreter (ParameterList)": 
    MinOverProcs: 0.035851
    MeanOverProcs: 0.0367033
    MaxOverProcs: 0.0375962
    MeanOverCallCounts: 0.0122344
  "STK_Interface::setupExodusFile(filename)": 
    MinOverProcs: 0.00085187
    MeanOverProcs: 0.00100666
    MaxOverProcs: 0.00107598
    MeanOverCallCounts: 0.00100666
  "STK_Interface::writeToExodus(timestep)": 
    MinOverProcs: 0.00668216
    MeanOverProcs: 0.00671434
    MaxOverProcs: 0.00673103
    MeanOverCallCounts: 0.00671434
  "UtilitiesBase::GetMatrixDiagonalInverse": 
    MinOverProcs: 0.000476837
    MeanOverProcs: 0.000619471
    MaxOverProcs: 0.000687838
    MeanOverCallCounts: 0.00020649
  "panzer::DOFManager::buildGlobalUnknowns": 
    MinOverProcs: 0.00236702
    MeanOverProcs: 0.00239503
    MaxOverProcs: 0.0024271
    MeanOverCallCounts: 0.00239503
  "panzer::DOFManager::buildGlobalUnknowns::build_ghosted_array": 
    MinOverProcs: 8.60691e-05
    MeanOverProcs: 8.93474e-05
    MaxOverProcs: 9.10759e-05
    MeanOverCallCounts: 8.93474e-05
  "panzer::DOFManager::buildGlobalUnknowns::build_local_ids": 
    MinOverProcs: 0.000163078
    MeanOverProcs: 0.00016427
    MaxOverProcs: 0.000165939
    MeanOverCallCounts: 0.00016427
  "panzer::DOFManager::buildGlobalUnknowns::build_orientation": 
    MinOverProcs: 0.000185013
    MeanOverProcs: 0.000186265
    MaxOverProcs: 0.000187874
    MeanOverCallCounts: 0.000186265
  "panzer::DOFManager::buildGlobalUnknowns::build_owned_vector": 
    MinOverProcs: 0.000200987
    MeanOverProcs: 0.00020802
    MaxOverProcs: 0.000226021
    MeanOverCallCounts: 0.00020802
  "panzer::DOFManager::buildGlobalUnknowns_GUN": 
    MinOverProcs: 0.000953197
    MeanOverProcs: 0.00097084
    MaxOverProcs: 0.00099206
    MeanOverCallCounts: 0.00097084
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_04 createOneToOne": 
    MinOverProcs: 0.000366926
    MeanOverProcs: 0.000391424
    MaxOverProcs: 0.000409842
    MeanOverCallCounts: 0.000391424
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_05 alloc_unique_mv": 
    MinOverProcs: 8.10623e-06
    MeanOverProcs: 8.82149e-06
    MaxOverProcs: 9.05991e-06
    MeanOverCallCounts: 8.82149e-06
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_06 export": 
    MinOverProcs: 0.000370979
    MeanOverProcs: 0.000394821
    MaxOverProcs: 0.000405073
    MeanOverCallCounts: 0.000394821
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_07-09 local_count": 
    MinOverProcs: 1.90735e-06
    MeanOverProcs: 2.74181e-06
    MaxOverProcs: 3.09944e-06
    MeanOverCallCounts: 2.74181e-06
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_10 prefix_sum": 
    MinOverProcs: 6.69956e-05
    MeanOverProcs: 9.37581e-05
    MaxOverProcs: 0.000133991
    MeanOverCallCounts: 9.37581e-05
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_13-21 gid_assignment": 
    MinOverProcs: 1.00136e-05
    MeanOverProcs: 1.10269e-05
    MaxOverProcs: 1.19209e-05
    MeanOverCallCounts: 1.10269e-05
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_23 final_import": 
    MinOverProcs: 2.5034e-05
    MeanOverProcs: 3.05176e-05
    MaxOverProcs: 4.1008e-05
    MeanOverCallCounts: 3.05176e-05
  "panzer::DOFManager::buildTaggedMultiVector": 
    MinOverProcs: 0.000353098
    MeanOverProcs: 0.000381947
    MaxOverProcs: 0.000417948
    MeanOverCallCounts: 0.000381947
  "panzer::DOFManager::buildTaggedMultiVector::allocate_tagged_multivector": 
    MinOverProcs: 4.79221e-05
    MeanOverProcs: 4.87566e-05
    MaxOverProcs: 5.00679e-05
    MeanOverCallCounts: 4.87566e-05
  "panzer::DOFManager::buildTaggedMultiVector::fill_tagged_multivector": 
    MinOverProcs: 4.48227e-05
    MeanOverProcs: 4.54783e-05
    MaxOverProcs: 4.60148e-05
    MeanOverCallCounts: 4.54783e-05
  "panzer::DOFManager::builderOverlapMapFromElements": 
    MinOverProcs: 0.000233889
    MeanOverProcs: 0.000264466
    MaxOverProcs: 0.000299931
    MeanOverCallCounts: 0.000264466
  "panzer::SquareQuadMeshFactory::buildUncomittedMesh()": 
    MinOverProcs: 0.000164986
    MeanOverProcs: 0.000167966
    MaxOverProcs: 0.000170946
    MeanOverCallCounts: 0.000167966
  "panzer::SquareQuadMeshFactory::completeMeshConstruction()": 
    MinOverProcs: 0.045965
    MeanOverProcs: 0.0459757
    MaxOverProcs: 0.0459869
    MeanOverCallCounts: 0.0459757
Call counts:
  "Belos: BlockGmresSolMgr total solve time": 
    MinOverProcs: 3
    MeanOverProcs: 3
    MaxOverProcs: 3
    MeanOverCallCounts: 3
  "Belos: DGKS[2]: Ortho (Inner Product)": 
    MinOverProcs: 22
    MeanOverProcs: 22
    MaxOverProcs: 22
    MeanOverCallCounts: 22
  "Belos: DGKS[2]: Ortho (Norm)": 
    MinOverProcs: 36
    MeanOverProcs: 36
    MaxOverProcs: 36
    MeanOverCallCounts: 36
  "Belos: DGKS[2]: Ortho (Update)": 
    MinOverProcs: 22
    MeanOverProcs: 22
    MaxOverProcs: 22
    MeanOverCallCounts: 22
  "Belos: DGKS[2]: Orthogonalization": 
    MinOverProcs: 14
    MeanOverProcs: 14
    MaxOverProcs: 14
    MeanOverCallCounts: 14
  "Belos: Operation Op*x": 
    MinOverProcs: 18
    MeanOverProcs: 18
    MaxOverProcs: 18
    MeanOverCallCounts: 18
  "Belos: Operation Prec*x": 
    MinOverProcs: 14
    MeanOverProcs: 14
    MaxOverProcs: 14
    MeanOverCallCounts: 14
  "Ifpack2::Chebyshev::apply": 
    MinOverProcs: 28
    MeanOverProcs: 28
    MaxOverProcs: 28
    MeanOverCallCounts: 28
  "Ifpack2::Chebyshev::compute": 
    MinOverProcs: 3
    MeanOverProcs: 3
    MaxOverProcs: 3
    MeanOverCallCounts: 3
  "MILO::assembly::computeJacRes() - gather": 
    MinOverProcs: 2
    MeanOverProcs: 2
    MaxOverProcs: 2
    MeanOverCallCounts: 2
  "MILO::assembly::computeJacRes() - insert": 
    MinOverProcs: 20
    MeanOverProcs: 20
    MaxOverProcs: 20
    MeanOverCallCounts: 20
  "MILO::assembly::computeJacRes() - physics evaluation": 
    MinOverProcs: 20
    MeanOverProcs: 20
    MaxOverProcs: 20
    MeanOverCallCounts: 20
  "MILO::assembly::computeJacRes() - total assembly": 
    MinOverProcs: 2
    MeanOverProcs: 2
    MaxOverProcs: 2
    MeanOverCallCounts: 2
  "MILO::assembly::createCells()": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "MILO::assembly::createWorkset()": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "MILO::assembly::dofConstraints()": 
    MinOverProcs: 2
    MeanOverProcs: 2
    MaxOverProcs: 2
    MeanOverCallCounts: 2
  "MILO::assembly::setDirichlet()": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "MILO::assembly::setInitial()": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "MILO::boundaryCell - build basis": 
    MinOverProcs: 2
    MeanOverProcs: 2.5
    MaxOverProcs: 3
    MeanOverCallCounts: 2.5
  "MILO::cell::computeJacRes() - fill local Jacobian": 
    MinOverProcs: 20
    MeanOverProcs: 20
    MaxOverProcs: 20
    MeanOverCallCounts: 20
  "MILO::cell::computeJacRes() - fill local residual": 
    MinOverProcs: 20
    MeanOverProcs: 20
    MaxOverProcs: 20
    MeanOverCallCounts: 20
  "MILO::cell::computeJacRes() - volume residual": 
    MinOverProcs: 20
    MeanOverProcs: 20
    MaxOverProcs: 20
    MeanOverCallCounts: 20
  "MILO::cell::computeSolAvg()": 
    MinOverProcs: 30
    MeanOverProcs: 30
    MaxOverProcs: 30
    MeanOverCallCounts: 30
  "MILO::cell::computeSolnFaceIP()": 
    MinOverProcs: 40
    MeanOverProcs: 40
    MaxOverProcs: 40
    MeanOverCallCounts: 40
  "MILO::cell::computeSolnVolIP()": 
    MinOverProcs: 30
    MeanOverProcs: 30
    MaxOverProcs: 30
    MeanOverCallCounts: 30
  "MILO::cell::constructor - build basis": 
    MinOverProcs: 10
    MeanOverProcs: 10
    MaxOverProcs: 10
    MeanOverCallCounts: 10
  "MILO::cell::constructor - build face basis": 
    MinOverProcs: 10
    MeanOverProcs: 10
    MaxOverProcs: 10
    MeanOverCallCounts: 10
  "MILO::cellMetaData::constructor()": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "MILO::driver::total run time": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "MILO::driver::total setup and execution time": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "MILO::function::decompose": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "MILO::function::evaluate": 
    MinOverProcs: 162
    MeanOverProcs: 162.5
    MaxOverProcs: 163
    MeanOverCallCounts: 162.5
  "MILO::physics::getSideInfo()": 
    MinOverProcs: 12
    MeanOverProcs: 12.5
    MaxOverProcs: 13
    MeanOverCallCounts: 12.5
  "MILO::physics::setBCData()": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "MILO::physics::setDirichletData()": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "MILO::postprocess::computeError": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "MILO::postprocess::writeSolution": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "MILO::solver::linearSolver()": 
    MinOverProcs: 2
    MeanOverProcs: 2
    MaxOverProcs: 2
    MeanOverCallCounts: 2
  "MILO::solver::projectDirichlet()": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "MILO::solver::setDirichlet()": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "MILO::solver::setInitial()": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "MILO::solver::setupFixedDOFs()": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "MILO::solver::setupLinearAlgebra()": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "MILO::thermal::volumeResidual() - evaluation of residual": 
    MinOverProcs: 20
    MeanOverProcs: 20
    MaxOverProcs: 20
    MeanOverCallCounts: 20
  "MILO::thermal::volumeResidual() - function evaluation": 
    MinOverProcs: 20
    MeanOverProcs: 20
    MaxOverProcs: 20
    MeanOverCallCounts: 20
  "MILO::workset::computeSolnVolIP - allocate/compute seeded": 
    MinOverProcs: 30
    MeanOverProcs: 30
    MaxOverProcs: 30
    MeanOverCallCounts: 30
  "MILO::workset::computeSolnVolIP - compute seeded sol at ip": 
    MinOverProcs: 30
    MeanOverProcs: 30
    MaxOverProcs: 30
    MeanOverCallCounts: 30
  "MILO::workset::reset*": 
    MinOverProcs: 20
    MeanOverProcs: 20
    MaxOverProcs: 20
    MeanOverCallCounts: 20
  "MueLu: ParameterListInterpreter (ParameterList)": 
    MinOverProcs: 3
    MeanOverProcs: 3
    MaxOverProcs: 3
    MeanOverCallCounts: 3
  "STK_Interface::setupExodusFile(filename)": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "STK_Interface::writeToExodus(timestep)": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "UtilitiesBase::GetMatrixDiagonalInverse": 
    MinOverProcs: 3
    MeanOverProcs: 3
    MaxOverProcs: 3
    MeanOverCallCounts: 3
  "panzer::DOFManager::buildGlobalUnknowns": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "panzer::DOFManager::buildGlobalUnknowns::build_ghosted_array": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "panzer::DOFManager::buildGlobalUnknowns::build_local_ids": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "panzer::DOFManager::buildGlobalUnknowns::build_orientation": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "panzer::DOFManager::buildGlobalUnknowns::build_owned_vector": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "panzer::DOFManager::buildGlobalUnknowns_GUN": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_04 createOneToOne": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_05 alloc_unique_mv": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_06 export": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_07-09 local_count": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_10 prefix_sum": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_13-21 gid_assignment": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_23 final_import": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "panzer::DOFManager::buildTaggedMultiVector": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "panzer::DOFManager::buildTaggedMultiVector::allocate_tagged_multivector": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "panzer::DOFManager::buildTaggedMultiVector::fill_tagged_multivector": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "panzer::DOFManager::builderOverlapMapFromElements": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "panzer::SquareQuadMeshFactory::buildUncomittedMesh()": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "panzer::SquareQuadMeshFactory::completeMeshConstruction()": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
