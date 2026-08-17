#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys, os
sys.path.append("../../scripts")
from mrhyde_test_support import *

# ==============================================================================
# Parsing input

desc = ''' LQ Hessian-vector identity check on a small transient Maxwell E-B
control problem. Exact HessVec path (src_gate scaffolding). Asserts on the
[HV-*] and Hessian-symmetry tags with tolerance-based predicates. Tolerances
are looser than the steady-state thermal test because the discrete adjoint
runs through the DIRK stage sequence and Belos linear tolerances propagate
into checkHessSym. '''

its = mrhyde_test_support(desc)

its.opts.verbose = True

root = 'mrhyde'

#TESTING active
#TESTING -n 3
#TESTING -k maxwell,optimization,hessvec,lq

# ==============================================================================
status = 0

if its.opts.preprocess:
  status += its.call('echo "  No preprocessing, yet."')

status += its.call('mpiexec -n 3 ../../mrhyde >& mrhyde.log')
status += its.clean_log()

# Extract Hessian symmetry relative error.
def _hess_sym_rel_err(text):
  m = re.search(r'Hessian symmetry check.*?\n.*?abs error\s*\n\s*'
                r'([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)',
                text, re.DOTALL)
  if m is None:
    return None
  try:
    a, b, err = float(m.group(1)), float(m.group(2)), float(m.group(3))
    ref = max(abs(a), abs(b), 1.0e-300)
    return err / ref
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
  # Primary exact-path assertion: b == 0 in S(v) = L v + b (homogeneity).
  ('[HV-ZERO] || H*0 || == 0',
   pred_scalar_lt('[HV-ZERO] || H*0 || =', 1.0e-30)),

  # Linearity of v -> H v; Belos linear tol propagates through DIRK stages.
  ('[HV-BILINEARITY] relative < 1e-10',
   pred_scalar_lt('[HV-BILINEARITY]', 1.0e-10, extractor=_second_scalar('[HV-BILINEARITY]'))),

  # Convex LQ: reduced Hessian PSD in the used inner product.
  ('[HV-RAYLEIGH] min(<vi, H vi>) > 0',
   pred_scalar_gt('[HV-RAYLEIGH]', 0.0, extractor=_rayleigh_min)),

  # Secant identity is exact on LQ; same Belos-floor limitation as bilinearity.
  ('[SECANT-IDENTITY] relative < 1e-12',
   pred_scalar_lt('[SECANT-IDENTITY]', 1.0e-12, extractor=_second_scalar('[SECANT-IDENTITY]'))),

  # Discrete transpose: symmetry to Belos floor, not O(dt).
  ('Hessian symmetry relative < 1e-10',
   pred_scalar_lt('Hessian symmetry', 1.0e-10, extractor=_hess_sym_rel_err)),

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
