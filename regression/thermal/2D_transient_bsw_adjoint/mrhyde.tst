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

desc = ''' Sketched-window checkpointing for the transient thermal adjoint.
       Two planned windows carry the low-rank middle of the trajectory, so
       their reversals cost no solves; a zero window budget must reduce to
       classic checkpointing exactly.
       '''

its = mrhyde_test_support(desc)

#-------------------------------------------------------------------------------
# Problem Parameters

root = 'mrhyde'
gradtol_windows = 1.0e-9    # window tolerance is 1e-8; the trajectory is rank 1
gradtol_classic = 1.0e-12   # no windows served means the classic exact path
classic_solves = 65         # 20 steps + p(20, 3)

# These comments are for testing with the runtest.py utility.
#TESTING active
#TESTING -n 1
#TESTING -k thermal,transient,adjoint,checkpointing,sketching

# ==============================================================================
status = 0

# ------------------------------
if its.opts.preprocess:
  if its.opts.verbose != 'none': print('---> Preprocessing %s' % (root))
  status += its.call('echo "  No preprocessing, yet."')

def read_gradient(filename):
  with open(filename) as f:
    return [float(v) for v in f.read().split()]

def read_report(logfile):
  used = predicted = windows = None
  for line in open(logfile):
    if 'forward solves used' in line:
      used = int(line.split(':')[1].split()[0])
      predicted = int(line.split('predicted')[1].split(')')[0])
    if 'committed windows' in line:
      windows = int(line.split(':')[1].split()[0])
  return used, predicted, windows

def compare(gradient, reference, tol, label):
  bad = 0
  if len(gradient) != len(reference):
    print('  Failure: gradient length differs (%s).' % label)
    return 1
  for i in range(len(reference)):
    error = abs(gradient[i] - reference[i])/max(abs(reference[i]), 1.0)
    if error > tol:
      print('  Failure: gradient differs (%s, component %d, error %.3e).'
            % (label, i, error))
      bad += 1
  return bad

# ------------------------------
# Reference: store the whole trajectory.

status += its.call('mpiexec -n 1 ../../mrhyde input_stored.yaml')
reference = read_gradient('grad_stored.scalar.0.dat')

# ------------------------------
# Two planned windows over the low-rank middle: reversals there are free.

status += its.call('mpiexec -n 1 ../../mrhyde input_bsw.yaml > bsw.log')
gradient = read_gradient('grad_bsw.scalar.0.dat')
status += compare(gradient, reference, gradtol_windows, 'windowed')

used, predicted, windows = read_report('bsw.log')
if windows != 2:
  print('  Failure: expected 2 committed windows, got %s.' % windows)
  status += 1
if used is None or used != predicted:
  print('  Failure: windowed run used %s solves, schedule predicted %s.'
        % (used, predicted))
  status += 1
if used is None or used >= classic_solves:
  print('  Failure: windowed run used %s solves; the windows saved nothing.'
        % used)
  status += 1

# ------------------------------
# Zero window budget: greedy opens nothing, so this must be the classic
# checkpointed adjoint exactly -- same gradient, same 65 solves.

status += its.call('mpiexec -n 1 ../../mrhyde input_bsw_nowin.yaml > nowin.log')
gradient = read_gradient('grad_nowin.scalar.0.dat')
status += compare(gradient, reference, gradtol_classic, 'zero budget')

used, predicted, windows = read_report('nowin.log')
if windows != 0:
  print('  Failure: zero budget opened %s windows.' % windows)
  status += 1
if used != classic_solves:
  print('  Failure: zero budget used %s solves, expected %d.'
        % (used, classic_solves))
  status += 1

if status == 0:
  its.call('rm -rf *.dat *.log')

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
