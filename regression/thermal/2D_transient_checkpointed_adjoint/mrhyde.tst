#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys, os
import subprocess as sp
import string
import shutil
sys.path.append("../../scripts")
from mrhyde_test_support import *

# ==============================================================================
# Parsing input

desc = ''' Revolve checkpointed transient adjoint for the thermal problem.
       Checks that checkpointing gives the same gradient as storing the whole
       trajectory, at the cost Algorithm 799 predicts, without storing it.
       '''

its = mrhyde_test_support(desc)

#-------------------------------------------------------------------------------
# Problem Parameters

root = 'mrhyde'
gradtol = 1.0e-12    # checkpointing recomputes, so the gradient should be exact
num_steps = 20       # must match "number of steps" in the input files
budgets = [2, 3, 5]  # checkpoint budgets to sweep

# These comments are for testing with the runtest.py utility.
#TESTING active
#TESTING -n 1
#TESTING -k thermal,transient,adjoint,checkpointing

# ==============================================================================
status = 0

# ------------------------------
if its.opts.preprocess:
  if its.opts.verbose != 'none': print('---> Preprocessing %s' % (root))
  status += its.call('echo "  No preprocessing, yet."')

# p(m,s) from Griewank & Walther, Algorithm 799 eq. (3): the fewest extra
# forward steps needed to reverse m steps with s checkpoints.
def min_extra_forward_steps(m, s):
  depth = 0
  reachable = 1.0
  while reachable < m:
    depth += 1
    reachable = reachable*(depth + s)/depth
  return depth*m - round(reachable*depth/(s + 1.0))

def read_gradient(filename):
  with open(filename) as f:
    return [float(v) for v in f.read().split()]

# ------------------------------
# Reference: store the whole trajectory.

status += its.call('mpiexec -n 1 ../../mrhyde input_stored.yaml')
reference = read_gradient('grad_stored.scalar.0.dat')

# ------------------------------
# Checkpointed runs must reproduce it exactly.

for budget in budgets:
  status += its.call('mpiexec -n 1 ../../mrhyde input_ckpt%d.yaml > ckpt%d.log' % (budget, budget))

  gradient = read_gradient('grad_ckpt%d.scalar.0.dat' % budget)

  if len(gradient) != len(reference):
    print('  Failure: gradient length differs at %d checkpoints.' % budget)
    status += 1
  else:
    for i in range(len(reference)):
      error = abs(gradient[i] - reference[i])
      scale = max(abs(reference[i]), 1.0)
      if error/scale > gradtol:
        print('  Failure: gradient differs at %d checkpoints (component %d, error %.3e).'
              % (budget, i, error/scale))
        status += 1

  # The schedule should cost exactly num_steps + p(num_steps, budget) forward
  # solves, and should not leave the trajectory in full storage.
  solves = None
  stored = None
  for line in open('ckpt%d.log' % budget):
    if 'forward solves used' in line:
      solves = int(line.split(':')[1].split()[0])
    if 'states in full storage' in line:
      stored = int(line.split(':')[1].split()[0])

  expected = num_steps + min_extra_forward_steps(num_steps, budget)
  if solves != expected:
    print('  Failure: %d checkpoints cost %s solves, expected %d.'
          % (budget, solves, expected))
    status += 1

  if stored != 0:
    print('  Failure: %d checkpoints left %s states in full storage, expected 0.'
          % (budget, stored))
    status += 1

if status == 0:
  its.call('rm -rf *.dat ckpt*.log')

# ------------------------------
if its.opts.graphics and not status:
  if its.opts.verbose != 'none': print('---> Graphics %s' % (root))
  status += its.call('echo "  No graphics, yet."')

# ------------------------------
if its.opts.clean and not status:
  if its.opts.verbose != 'none': print('---> Clean %s' % (root))
  status += its.call('ichos_clean')

# ==============================================================================
if status == 0: print('Success.')
else:           print('Failure.')
sys.exit(status)
