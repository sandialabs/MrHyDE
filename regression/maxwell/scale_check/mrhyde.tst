#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys, os
sys.path.append("../../scripts")
from mrhyde_test_support import *

# ==============================================================================
# Parsing input

desc = ''' Magnitude scan probe on a Maxwell E-B control problem. Reports each
objective and regularization term in unweighted and weighted forms; exits
after the scan (Iteration Limit: 0). Asserts on the magnitude-scan totals
plus iter-0 diagnostics with tolerance-based predicates (portable across
BLAS/MPI implementations). '''

its = mrhyde_test_support(desc)

its.opts.verbose = True

root = 'mrhyde'

#TESTING active
#TESTING -n 11
#TESTING -k maxwell,optimization,scale,scan

# ==============================================================================
status = 0

if its.opts.preprocess:
  status += its.call('echo "  No preprocessing, yet."')

status += its.call('mpiexec -n 11 ../../mrhyde >& mrhyde.log')
status += its.clean_log()

# Pull weighted column for a named scan row.
def _scan_row_weighted(term):
  def _e(text):
    pat = r'^\s*' + re.escape(term) + \
          r'\s+\S+' + \
          r'\s+([-+0-9.eE]+)' + \
          r'\s+([-+0-9.eE]+)' + \
          r'\s+([-+0-9.eE]+)\s*$'
    for line in text.splitlines():
      m = re.match(pat, line)
      if m:
        try:
          return float(m.group(3))
        except ValueError:
          return None
    return None
  return _e

# Pull weighted column from TOTAL row.
def _scan_total_weighted(text):
  # TOTAL row format: name, unweighted, weighted.
  for line in text.splitlines():
    m = re.match(r'^\s*TOTAL\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$', line)
    if m:
      try:
        return float(m.group(2))
      except ValueError:
        return None
  return None

# Parse iter-0 value and gnorm from ROL table.
def _iter0_gnorm(text):
  m = re.search(r'^\s*0\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+---',
                text, re.MULTILINE)
  if m is None:
    return None
  try:
    return float(m.group(2))
  except ValueError:
    return None

def _iter0_value(text):
  m = re.search(r'^\s*0\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+---',
                text, re.MULTILINE)
  if m is None:
    return None
  try:
    return float(m.group(1))
  except ValueError:
    return None

checks = [
  # Allow BLAS/MPI drift.
  ('EM Energy weighted ~ 2e-4 (+/-20%)',
   pred_scalar_rel('EM Energy weighted', 2.0e-4, 0.20,
                   extractor=_scan_row_weighted('EM Energy'))),

  ('RegObj weighted == 0',
   pred_scalar_lt('RegObj weighted', 1.0e-30,
                  extractor=_scan_row_weighted('RegObj'))),

  ('RegObj/l2reg weighted < 1e-20',
   pred_scalar_lt('RegObj/l2reg weighted', 1.0e-20,
                  extractor=_scan_row_weighted('RegObj/l2reg'))),

  ('RegObj/curlreg weighted < 1e-10',
   pred_scalar_lt('RegObj/curlreg weighted', 1.0e-10,
                  extractor=_scan_row_weighted('RegObj/curlreg'))),

  ('TOTAL weighted ~ 2e-4 (+/-20%)',
   pred_scalar_rel('TOTAL weighted', 2.0e-4, 0.20, extractor=_scan_total_weighted)),

  ('Iter-0 value < 1e-10',
   pred_scalar_lt('Iter-0 value', 1.0e-10, extractor=_iter0_value)),

  ('Iter-0 gnorm ~ 1.18 (+/-5%)',
   pred_scalar_rel('Iter-0 gnorm', 1.18, 0.05, extractor=_iter0_gnorm)),

  ('MAGNITUDE-SCAN ran',
   pred_contains('[MAGNITUDE-SCAN] probe at seeded random ctrl')),

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
