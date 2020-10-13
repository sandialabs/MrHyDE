#!/usr/bin/python
from os import listdir
mydir="/Users/dtseidl/src/milo-2017/examples/thermal/weld_test_2d_ms/subgrid_data_4core"
files = listdir(mydir)
tfile = "fnames.txt"
output = open(tfile,'w')
for f in files:
  output.write("%s\n" % f)
output.close()
