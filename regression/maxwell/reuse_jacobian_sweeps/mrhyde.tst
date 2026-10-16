#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys, os, re
sys.path.append("../../scripts")
from mrhyde_test_support import *

# ==============================================================================
# Parsing input

desc = ''' Direct solves with 'reuse Jacobian across sweeps' against the same deck
without it. The LQ Maxwell checks run forward, adjoint and both incremental sweeps,
so the kept forward and adjoint factorizations are both reused; the exact-path
numbers must not move. The mass run adds 'print mass', whose setup leaves a mass
matrix in the state context that must not be kept as the Jacobian. '''

its = mrhyde_test_support(desc)

its.opts.verbose = True

root = 'mrhyde'

#TESTING active
#TESTING -n 3
#TESTING -k maxwell,optimization,hessvec,direct

# ==============================================================================
status = 0

if its.opts.preprocess:
  status += its.call('echo "  No preprocessing, yet."')

logs = {}
for name in ('off', 'on', 'mass'):
  log = 'mrhyde_%s.log' % name
  status += its.call('mpiexec -n 3 ../../mrhyde input_%s.yaml >& %s' % (name, log))
  its.clean_log(log)   # advisory IOSS filter; its sed call returns nonzero on BSD
  logs[name] = open(log, errors='replace').read()
status += its.call('rm -f mass.mm lumped_mass.mm')

NUM = r'([-+]?\d+\.\d+[eE][-+]?\d+)'
ROW = re.compile(r'^\s+' + r'\s+'.join([NUM] * 4) + r'\s*$', re.M)

# Second column of a check table: grad'*dir and norm(Hess*vec) are exact-path values.
def _table_column(text, header, nxt):
  i = text.find(header)
  j = text.find(nxt, i + 1)
  if i < 0 or j < 0:
    return []
  return [float(m.group(2)) for m in ROW.finditer(text[i:j])]

def _exact_numbers(text):
  out = _table_column(text, 'Gradient check', 'Hessian-vector check')
  out += _table_column(text, 'Hessian-vector check', 'Hessian symmetry check')
  m = re.search(r'Hessian symmetry check.*?abs error\s*\n\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)',
                text, re.DOTALL)
  if m:
    out += [float(m.group(1)), float(m.group(2))]
  m = re.search(r'<v1, H v1>\s*=\s*([-+0-9.eE]+).*?<v2, H v2>\s*=\s*([-+0-9.eE]+)', text)
  if m:
    out += [float(m.group(1)), float(m.group(2))]
  m = re.search(r'^\s*0\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+---', text, re.M)
  if m:
    out += [float(m.group(1)), float(m.group(2))]
  return out

# Parse "relative =" scalar for a tag.
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
  ('[HV-ZERO] || H*0 || == 0',
   pred_scalar_lt('[HV-ZERO] || H*0 || =', 1.0e-30)),
  ('[HV-BILINEARITY] relative < 1e-10',
   pred_scalar_lt('[HV-BILINEARITY]', 1.0e-10, extractor=_second_scalar('[HV-BILINEARITY]'))),
  ('[SECANT-IDENTITY] relative < 1e-12',
   pred_scalar_lt('[SECANT-IDENTITY]', 1.0e-12, extractor=_second_scalar('[SECANT-IDENTITY]'))),
  ('ROL terminates at Iteration Limit',
   pred_contains('Iteration Limit Exceeded')),
]
status += check_predicates('mrhyde_on.log', checks, verbose=True)

# Every run factors the same matrices, so a reused factorization changes nothing.
a = _exact_numbers(logs['off'])
for name in ('on', 'mass'):
  b = _exact_numbers(logs[name])
  if len(a) != 32 or len(b) != len(a):
    print('FAIL  exact-path numbers: read %d and %d of 32' % (len(a), len(b)))
    status += 1
    continue
  worst = max(abs(x - y) / max(abs(x), 1.0e-300) for x, y in zip(a, b))
  ok = worst <= 1.0e-12
  print('%s  32 exact-path numbers, %s against off: max rel diff %.3e <= 1e-12'
        % ('pass' if ok else 'FAIL', name, worst))
  if not ok:
    status += 1

if its.opts.clean and not status:
  status += its.call('rm -f mrhyde_off.log mrhyde_on.log mrhyde_mass.log')

# ==============================================================================
if status == 0: print('Success.')
else:           print('Failure.')
sys.exit(status)
