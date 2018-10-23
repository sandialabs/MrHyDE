#!/Users/dtseidl/anaconda/bin/python

import sys, os
import numpy as np
#import pickle
sys.path.append("/Users/dtseidl/local2/trilinos-seacas/lib")
from exodus import exodus
#import scipy.spatial as spt

# assumes a scalar data field for the inverse problem
# modification for a vector data field (needed for linear elasticity)
# would not be difficult

datfile = "output.exo"
outfile = "inverse_mesh.exo"

dat_field_name = "e"

include_node_sets = False

rm_output=True
if (rm_output):
  os.system("rm " + outfile)

# read in exodus information from data mesh
dat_mesh = exodus(datfile, mode='r', array_type="numpy")
timesteps = dat_mesh.get_times() # includes initial condition
steps = timesteps.size

dim = dat_mesh.num_dimensions()
coor_names = dat_mesh.get_coord_names()
coords = dat_mesh.get_coords()
if (dim == 2):
  coord_names = ["x", "y"]
elif (dim == 3):
  coord_names = ["x", "y", "z"]

nblocks = dat_mesh.num_blks()
if (include_node_sets):
  nnode_sets = dat_mesh.num_node_sets()
else:
  nnode_sets = 0
nside_sets = dat_mesh.num_side_sets()
blkid = dat_mesh.get_elem_blk_ids()[0]
blk_name = dat_mesh.get_elem_blk_name(blkid)
conn_array = dat_mesh.get_elem_connectivity(blkid)
nnode_per_elem = conn_array[2]
# seems like this should always be zero for milo problems
nattr_per_elem = 0
if (dim == 2):
  if (nnode_per_elem == 3):
    elem_type = "TRI3"
  elif (nnode_per_elem == 4):
    elem_type = "QUAD4"
elif (dim == 3):
  if (nnode_per_elem == 4):
    elem_type = "TETRA4"
  elif (nnode_per_elem == 8):
    elem_type = "HEX8"
nnodes = dat_mesh.num_nodes()
nelem = dat_mesh.num_elems()
node_map = dat_mesh.get_node_id_map()

# open new file for writing
mtitle="mesh"
mesh = exodus(outfile, mode='w', array_type="numpy", title=mtitle,
              numDims=dim, numNodes=nnodes, numElems=nelem,
              numBlocks=nblocks, numNodeSets=nnode_sets, numSideSets=nside_sets)
# block info
mesh.put_elem_blk_info(blkid,elem_type,nelem,nnode_per_elem,nattr_per_elem)
mesh.put_elem_blk_name(blkid,blk_name)
# coordinates
mesh.put_coord_names(coord_names)
mesh.put_node_id_map(node_map)
mesh.put_coords(coords[0],coords[1],coords[2])
# connectivity
mesh.put_elem_connectivity(blkid,conn_array[0])
# will need this for sensor data ...
#ec_tree = spt.KDTree(elem_centers)

# node sets
if (include_node_sets):
  node_set_names = dat_mesh.get_node_set_names()
  node_set_ids = dat_mesh.get_node_set_ids()
  mesh.put_node_set_names(node_set_names)
  for nsid in node_set_ids:
    ns_nodes = dat_mesh.get_node_set_nodes(nsid)
    ns_distfact = dat_mesh.get_node_set_dist_facts(nsid)
    mesh.put_node_set(nsid,ns_nodes)
    mesh.put_node_set_dist_fact(nsid,ns_distfact)

# side sets
side_set_names = dat_mesh.get_side_set_names()
side_set_ids = dat_mesh.get_side_set_ids()
mesh.put_side_set_names(side_set_names)

for ssid in side_set_ids:
  ssparams = dat_mesh.get_side_set_params(ssid)
  ssinfo = dat_mesh.get_side_set(ssid)
  mesh.put_side_set_params(ssid,ssparams[0],ssparams[1])
  mesh.put_side_set(ssid,ssinfo[0],ssinfo[1])

# add nodal data
nvar_names = ["problem_data"]
nvars = len(nvar_names)
mesh.set_node_variable_number(nvars)
# names of element fields
for step in range(steps):
  mesh.put_time(step+1,timesteps[step])
  for i in range(nvars):
    ind = i + 1
    if (step == 0):
      mesh.put_node_variable_name(nvar_names[i],ind)
    mesh.put_node_variable_values(nvar_names[i],step+1,dat_mesh.get_node_variable_values(dat_field_name,step+1))

mesh.close()
dat_mesh.close()
