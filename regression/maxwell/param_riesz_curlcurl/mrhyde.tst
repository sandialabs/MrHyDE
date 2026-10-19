#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys, os, re
sys.path.append("../../scripts")
from mrhyde_test_support import *

# ==============================================================================
# Parsing input

desc = ''' Turns the parameter-space Riesz map on and checks the vector space it
defines. The same transient Maxwell E-B control case runs twice, once with the
Euclidean inner product and once with the regularizer as the metric, so the
gates compare two decks in one binary rather than recorded constants. '''

its = mrhyde_test_support(desc)

its.opts.verbose = True

root = 'mrhyde'

#TESTING active
#TESTING -n 3
#TESTING -k maxwell,optimization,riesz,hcurl,parameter

# ==============================================================================
status = 0

if its.opts.preprocess:
  status += its.call('echo "  No preprocessing, yet."')

DECKS = ['euclid', 'riesz']
logs = {}
for name in DECKS:
  log = 'mrhyde_%s.log' % name
  status += its.call('mpiexec -n 3 ../../mrhyde input_%s.yaml >& %s' % (name, log))
  its.clean_log(log)   # advisory IOSS filter; its sed call returns nonzero on BSD
  logs[name] = open(log, errors='replace').read()

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

# Minimum Rayleigh value from both directions.
def _rayleigh_min(text):
  m = re.search(r'\[HV-RAYLEIGH\].*?<v1, H v1>\s*=\s*([-+0-9.eE]+).*?'
                r'<v2, H v2>\s*=\s*([-+0-9.eE]+)', text)
  if m is None:
    return None
  try:
    return min(float(m.group(1)), float(m.group(2)))
  except ValueError:
    return None

# ROL's gradient norm at iteration 0, which is measured in the metric's norm.
def _iter0_gnorm(text):
  m = re.search(r'^\s*0\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+---', text, re.M)
  if m is None:
    return None
  try:
    return float(m.group(2))
  except ValueError:
    return None

# One named line of ROL's checkVector. The other twelve are absolute norms whose
# size tracks cond(P) and the vector scale, so they do not make a gate.
def _vcheck_line(label):
  def _e(text):
    m = re.search(re.escape(label) + r'[^0-9+-]*([-+0-9.eE]+)', text)
    if m is None:
      return None
    try:
      return float(m.group(1))
    except ValueError:
      return None
  return _e

common = [
  # apply() is the unweighted duality pairing, so it disagrees with dot() unless
  # dual() applies the same operator. Nothing else below can see that, and a
  # broken dual() takes this to O(1) rather than to something merely larger.
  ('checkVector apply and dual < 1e-10',
   pred_scalar_lt('apply and dual', 1.0e-10,
                  extractor=_vcheck_line('Consistency of apply and dual:'))),

  # A Riesz map must leave the gradient alone: checkGradient pairs primal with dual.
  ('[GRAD-CHECK] best rel_err < 1e-3',
   pred_scalar_lt('[GRAD-CHECK] best rel_err =', 1.0e-3)),

  ('[HV-ZERO] || H*0 || == 0',
   pred_scalar_lt('[HV-ZERO] || H*0 || =', 1.0e-30)),

  ('[HV-RAYLEIGH] min(<vi, H vi>) > 0',
   pred_scalar_gt('[HV-RAYLEIGH]', 0.0, extractor=_rayleigh_min)),

  ('[SECANT-IDENTITY] relative < 1e-12',
   pred_scalar_lt('[SECANT-IDENTITY]', 1.0e-12, extractor=_second_scalar('[SECANT-IDENTITY]'))),

  ('ROL terminates at Iteration Limit',
   pred_contains('Iteration Limit Exceeded')),
]

for name in DECKS:
  print('--- %s ---' % name)
  status += check_predicates('mrhyde_%s.log' % name, common, verbose=True)

# Reflexivity measures P^-1 P - I and is reported, not gated: it is an absolute
# norm, so it carries cond(P) times the vector scale.
for name in DECKS:
  v = _vcheck_line('Reflexivity. Consistency error:')(logs[name])
  print('      %-7s reflexivity %s' % (name, 'not found' if v is None else '%.3e' % v))

# The metric is the trust-region preconditioner, so an inert one leaves every
# number above unchanged.
g_euclid = _iter0_gnorm(logs['euclid'])
g_riesz = _iter0_gnorm(logs['riesz'])
if g_euclid is None or g_riesz is None or g_euclid == 0.0:
  print('FAIL  gnorm ratio: could not read iteration 0 from both logs')
  status += 1
else:
  ratio = g_riesz / g_euclid
  moved = (ratio < 0.5 or ratio > 2.0)
  print('%s  gnorm ratio outside [0.5, 2.0]: %.4e / %.4e = %.4e'
        % ('pass' if moved else 'FAIL', g_riesz, g_euclid, ratio))
  if not moved:
    status += 1

m = re.search(r'\[RieszMap\] P = ([-+0-9.eE]+)\*M \+ ([-+0-9.eE]+)\*S', logs['riesz'])
if m is None:
  print('FAIL  [RieszMap] line not found in the riesz log')
  status += 1
else:
  l2w, semiw = float(m.group(1)), float(m.group(2))
  ok = l2w > 0.0 and semiw > 0.0
  print('%s  [RieszMap] P = %.3e*M + %.3e*S, both weights > 0' % ('pass' if ok else 'FAIL', l2w, semiw))
  if not ok:
    status += 1

if its.opts.clean and not status:
  status += its.call('rm -f mrhyde_euclid.log mrhyde_riesz.log')

# ==============================================================================
if status == 0: print('Success.')
else:           print('Failure.')
sys.exit(status)
