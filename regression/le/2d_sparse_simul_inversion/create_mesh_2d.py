#!/Users/dtseidl/anaconda/bin/python

import sys, os
import numpy as np
import pickle
# must build trilinos with shared libs
# for import to be successful
sys.path.append("/Users/dtseidl/local2/trilinos-seacas/lib")
from exodus import exodus

noise_level = 2 # 2% noise 
data_level = "dp1" #  dr, dp1, dp2 -- data rich/poor1/poor2
# dp1 and dp2 have different sparsity patterns
u_rms = np.loadtxt("u_rms.dat")
a = float(noise_level)/100*u_rms # standard deviation of measurement noise
# u_rms computed from paraview filter (u = u_y here)
#np.savetxt("noise_std.dat",np.array([a]))
#np.savetxt("noise_var.dat",np.array([a**2]))
np.random.seed(204)

# load sensor data
sensor_data = pickle.load( open( "sensor_data.p", "rb") )

mfile = "input_mesh_" + data_level + "_n" + str(noise_level) + ".exo"
rm_output=True
if (rm_output):
  os.system("rm " + mfile)
# mesh parameters
dim = 2
Lx = 1.0
Ly = 1.0
Nx = 20
Ny = 20
coord_names = ["x", "y"]
nblocks = 1
nnode_sets = 0
nside_sets = 4
side_set_names = ["bottom", "right", "top", "left"]
nblocks = 1
step = 1
t = 0.0
blkid = 1
blk_name = "eblock-0_0"
elem_type = "QUAD4"
nnode_per_elem = 4
nattr_per_elem = 0
# derived quantities
dx = Lx/Nx
dy = Ly/Ny
nx = np.linspace(0,Lx,Nx+1)
ny = np.linspace(0,Ly,Ny+1)
nz = np.zeros(nx.shape)
nnodes = (Nx+1)*(Ny+1)
nelem = Nx*Ny
# measurement noise
meas_noise_1 = np.random.normal(0.0,a,nelem)
meas_noise_2 = np.random.normal(0.0,a,nelem)
# optional title
mtitle="mesh"
# open new file for writing
mesh = exodus(mfile, mode='w', array_type="numpy", title=mtitle,
              numDims=dim, numNodes=nnodes, numElems=nelem,
              numBlocks=nblocks, numNodeSets=nnode_sets, numSideSets=nside_sets)
# time and block info
mesh.put_time(step,t)
mesh.put_elem_blk_info(blkid,elem_type,nelem,nnode_per_elem,nattr_per_elem)
mesh.put_elem_blk_name(blkid,blk_name)
# coordinates
mesh.put_coord_names(coord_names)
node_map = range(1,nnodes+1)
mesh.put_node_id_map(node_map)
cx = np.empty(nnodes)
cy = np.empty(nnodes)
cz = np.zeros(nnodes)
n = 0
for ycoor in ny:
  for xcoor in nx:
      cx[n] = xcoor
      cy[n] = ycoor
      n += 1
mesh.put_coords(cx,cy,cz)
# connectivity
conn_array = np.empty(nelem*nnode_per_elem,dtype=int)
spatial_node_map = np.empty((Ny+1,Nx+1),dtype=int)
n = 1
for ey in range(Ny+1):
  for ex in range(Nx+1):
    spatial_node_map[ey,ex] = n
    n += 1
cindex = 0
for ey in range(Ny):
  for ex in range(Nx):
    econn = [spatial_node_map[ey,ex], spatial_node_map[ey,ex+1],
             spatial_node_map[ey+1,ex+1], spatial_node_map[ey+1,ex]]
    for n in range(nnode_per_elem):
      conn_array[cindex] = econn[n]
      cindex += 1
mesh.put_elem_connectivity(blkid,conn_array)
# side sets
spatial_elem_map = np.empty((Ny,Nx),dtype=int)
e = 1
for ey in range(Ny):
  for ex in range(Nx):
    spatial_elem_map[ey,ex] = e
    e += 1
bottom_elems = spatial_elem_map[0,:]
right_elems = spatial_elem_map[:,-1]
top_elems = spatial_elem_map[-1,:]
left_elems = spatial_elem_map[:,0]
mesh.put_side_set_params(1,Nx,0)
mesh.put_side_set_params(2,Ny,0)
mesh.put_side_set_params(3,Nx,0)
mesh.put_side_set_params(4,Ny,0)
mesh.put_side_set(1,bottom_elems,np.ones(bottom_elems.shape,dtype=int))
mesh.put_side_set(2,right_elems,2*np.ones(right_elems.shape,dtype=int))
mesh.put_side_set(3,top_elems,3*np.ones(top_elems.shape,dtype=int))
mesh.put_side_set(4,left_elems,4*np.ones(left_elems.shape,dtype=int))
mesh.put_side_set_names(side_set_names)

# names of element fields
enames = ["numSensors", "sensor_1_Loc_x", "sensor_1_Loc_y", "sensor_1_Val_1", "sensor_1_Val_2"]
edata = {}
fullSensors = np.ones((Ny,Nx))
sparseSensors = np.zeros((Ny,Nx))
# edit to change sensor placement
if data_level == "dr":
  edata["numSensors"] = fullSensors.flatten()
else:
  if data_level == "dp1":
    pattern = np.array([2, 5, 8, 11, 14, 17])
  elif data_level == "dp2":
    pattern = np.array([1, 5, 9, 13, 17])
  sparseSensors[:,pattern] = 1
  edata["numSensors"] = sparseSensors.flatten()
  
  
edata["sensor_1_Loc_x"] = sensor_data["xCoor"]
edata["sensor_1_Loc_y"] = sensor_data["yCoor"]
# add noise here
edata["sensor_1_Val_1"] = sensor_data["d_x"] + meas_noise_1
edata["sensor_1_Val_2"] = sensor_data["d_y"] + meas_noise_2

# add element variables
elemvars = len(enames)
mesh.set_element_variable_number(elemvars)
for i in range(elemvars):
  ind = i + 1
  mesh.put_element_variable_name(enames[i],ind)
  mesh.put_element_variable_values(blkid,enames[i],step,edata[enames[i]])

mesh.close()

# for Bayesian inversion with Dakota ...
#if data_level == "dr":
#  np.savetxt("exp_data_" + data_level + "_n" + str(noise_level), \
#  np.append(edata["sensor_1_Val_2"], a**2*np.ones(meas_noise.shape)), \
#  newline=" ")
#else:
#  inds = np.nonzero(sparseSensors.flatten())[0]
#  np.savetxt("exp_data_" + data_level + "_n" + str(noise_level), \
#  np.append(edata["sensor_1_Val_2"][inds], a**2*np.ones(inds.size)), \
#  newline=" ")

           

