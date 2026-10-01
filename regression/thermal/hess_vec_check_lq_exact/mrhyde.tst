#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys, os
sys.path.append("../../scripts")
from mrhyde_test_support import *

# ==============================================================================
# Parsing input

desc = ''' LQ Hessian-vector identity check on a distributed-source thermal
control problem. Exact HessVec path (src_gate scaffolding). Asserts on the
[GRAD-CHECK], [HESSVEC-CHECK], [HV-*], [SECANT-IDENTITY] summary tags with
tolerance-based predicates (portable across BLAS/MPI implementations). '''

its = mrhyde_test_support(desc)

its.opts.verbose = True

root = 'mrhyde'

#TESTING active
#TESTING -n 4
#TESTING -k thermal,optimization,hessvec,lq

# ==============================================================================
status = 0

if its.opts.preprocess:
  status += its.call('echo "  No preprocessing, yet."')

status += its.call('mpiexec -n 4 ../../mrhyde >& mrhyde.log')
status += its.clean_log()

# Extract Hessian symmetry absolute error.
def _hess_sym_abs_err(text):
  m = re.search(r'Hessian symmetry check.*?\n.*?abs error\s*\n\s*'
                r'[-+0-9.eE]+\s+[-+0-9.eE]+\s+([-+0-9.eE]+)',
                text, re.DOTALL)
  if m is None:
    return None
  try:
    return float(m.group(1))
  except ValueError:
    return None

# Pull min Rayleigh value from both directions.
def _rayleigh_min(text):
  m = re.search(r'\[HV-RAYLEIGH\].*?<v1, H v1>\s*=\s*([-+0-9.eE]+).*?'
                r'<v2, H v2>\s*=\s*([-+0-9.eE]+)', text)
  if m is None:
    return None
  try:
    return min(float(m.group(1)), float(m.group(2)))
  except ValueError:
    return None

# Pull "relative =" scalar for a tagged line.
def _second_scalar(tag):
  def _e(text):
    m = re.search(re.escape(tag) + r'.*?relative\s*=\s*([-+0-9.eE]+)', text)
    if m is None:
      return None
    try:
      return float(m.group(1))
    except ValueError:
      return None
  return _e

checks = [
  ('[GRAD-CHECK] best rel_err < 1e-4',
   pred_scalar_lt('[GRAD-CHECK] best rel_err =', 1.0e-4)),

  ('[HESSVEC-CHECK] best rel_err < 1e-10',
   pred_scalar_lt('[HESSVEC-CHECK] best rel_err =', 1.0e-10)),

  ('[HV-ZERO] || H*0 || == 0',
   pred_scalar_lt('[HV-ZERO] || H*0 || =', 1.0e-30)),

  ('[HV-BILINEARITY] relative < 1e-12',
   pred_scalar_lt('[HV-BILINEARITY]', 1.0e-12, extractor=_second_scalar('[HV-BILINEARITY]'))),

  ('[HV-RAYLEIGH] min(<vi, H vi>) > 0',
   pred_scalar_gt('[HV-RAYLEIGH]', 0.0, extractor=_rayleigh_min)),

  ('[SECANT-IDENTITY] relative < 1e-12',
   pred_scalar_lt('[SECANT-IDENTITY]', 1.0e-12, extractor=_second_scalar('[SECANT-IDENTITY]'))),

  ('Hessian symmetry abs error < 1e-14',
   pred_scalar_lt('Hessian symmetry', 1.0e-14, extractor=_hess_sym_abs_err)),

  ('ROL terminates at Iteration Limit',
   pred_contains('Iteration Limit Exceeded')),
]

status += check_predicates('mrhyde.log', checks, verbose=True)

if its.opts.clean and not status:
  status += its.call('rm -f mrhyde.log')

# ==============================================================================
if status == 0: print('Success.')
else:           print('Failure.')
sys.exit(status)
