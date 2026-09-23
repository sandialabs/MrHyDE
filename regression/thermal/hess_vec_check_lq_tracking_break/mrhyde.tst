#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys, os
sys.path.append("../../scripts")
from mrhyde_test_support import *

# ==============================================================================
# Parsing input

desc = ''' Tracking-target HessVec breakdown with exact path physics. 
Algebraic checks [HV-ZERO], [HV-BILINEARITY], checkHessSym, and
[SECANT-IDENTITY] are large, and [HV-RAYLEIGH] is negative. '''

its = mrhyde_test_support(desc)

its.opts.verbose = True

root = 'mrhyde'

#TESTING active
#TESTING -n 4
#TESTING -k thermal,optimization,hessvec,lq,breakdown

# ==============================================================================
status = 0

if its.opts.preprocess:
  status += its.call('echo "  No preprocessing, yet."')

status += its.call('mpiexec -n 4 ../../mrhyde >& mrhyde.log')
status += its.clean_log()

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
  ('[HV-ZERO] || H*0 || > 1e-5 (breakdown)',
   pred_scalar_gt('[HV-ZERO] || H*0 || =', 1.0e-5)),

  ('[HV-BILINEARITY] relative > 1e-3 (breakdown)',
   pred_scalar_gt('[HV-BILINEARITY]', 1.0e-3, extractor=_second_scalar('[HV-BILINEARITY]'))),

  ('Hessian symmetry abs error > 1e-5 (breakdown)',
   pred_scalar_gt('Hessian symmetry', 1.0e-5, extractor=_hess_sym_abs_err)),

  ('[SECANT-IDENTITY] relative > 1e-1 (breakdown)',
   pred_scalar_gt('[SECANT-IDENTITY]', 1.0e-1, extractor=_second_scalar('[SECANT-IDENTITY]'))),

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
