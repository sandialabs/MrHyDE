#!/usr/bin/env python3
#-------------------------------------------------------------------------------

from __future__ import print_function
import optparse
import subprocess as sp
import sys, os, shutil
import struct

exec_sh = None
if shutil.which("/bin/bash") is not None:
  exec_sh = "/bin/bash"

# ==============================================================================

def syscmd(cmd, status=0, logfile=None, verbose=False, ignore_status=False):

  internal_status = 0

  # start the command
  if verbose: print(cmd)
  p = sp.Popen(cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE, executable=exec_sh)

  stdout = ''
  stderr = ''

  if verbose == True:
    with p.stdout:
      for line in iter(p.stdout.readline, b''): 
        print(line.decode('ascii'), end='')
  else:
    stdout, stderr = p.communicate()
  
  # wait for the command to finish and grab the exit status
  internal_status = p.wait()

  if stderr: print(stderr)
  if logfile:
    f = open(logfile, 'w')
    f.writelines(stdout)
    f.close()
  if not ignore_status:
    status += internal_status
    if internal_status != 0:
      print('  ==> Execution failed with status = %i!\n' %(internal_status))
      sys.exit(status)

  return status

# ==============================================================================
class mrhyde_test_support:
  """Class to help support mrhyde tests"""
  def __init__( self, description = 'MrHyDE testing script.', \
                      number_spatial_dimensions = 2 ):

    p = optparse.OptionParser(description)

    p.add_option("-n", dest="nprocs", default=None, \
                     action="store", type="int", metavar="nprocs", \
                     help="number of processors")

    p.add_option("-r", "--run", dest="run", default=False, \
                     action="store_true", \
                     help='''run the test (same as -ped). This is the
                             default option if none are given.''')
    p.add_option("-p", "--preprocess", dest="preprocess", default=False, \
                     action="store_true", help="run preprocess for this test")
    p.add_option("-e", "--execute", dest="execute", default=False, \
                     action="store_true", help="execute this test")
    p.add_option("-d", "--diff", dest="diff", default=False, \
                     action="store_true", help="run the difference test")
    p.add_option("-b", "--baseline", dest="baseline", default=False, \
                     action="store_true", help="baseline the test")
    p.add_option("", "--64", dest="mode_64", default=False, \
                     action="store_true", help="running 64 bit")
    p.add_option("", "--32", dest="mode_32", default=False, \
                     action="store_true", help="running 32 bit")
    p.add_option("-y", "--cray", dest="cray", default=False, \
                     action="store_true", help="running on cray")
    p.add_option("-g", "--graphics", dest="graphics", default=False, \
                     action="store_true", help="generate graphics for test")
    p.add_option("-c", "--clean", dest="clean", default=False, \
                     action="store_true", \
                     help="clean up test, if there are no failures")
    p.add_option("-v", "--verbose", dest="verbose", default=False, \
                     action="store_true", \
                     help='''echo out ALL screen text''')
    p.add_option("-q", "--quiet", dest="quiet", default=False, \
                     action="store_true", \
                     help='''echo NO screen text''')


    self.opts, self.args = p.parse_args()

    found_proc = False
    if self.opts.preprocess: found_proc = True
    if self.opts.execute:    found_proc = True
    if self.opts.diff:       found_proc = True
    if self.opts.baseline:   found_proc = True
    if self.opts.graphics:   found_proc = True
    if self.opts.clean:      found_proc = True
    if self.opts.run or not found_proc:
       found_proc = True
       self.opts.preprocess = True
       self.opts.execute    = True
       self.opts.diff       = True

    # error if both options are supplied: --32 and --64
    if self.opts.mode_32 and self.opts.mode_64:
       print('Error: cannot specify both --32 and --64 bit mode')
       sys.exit(0)
    # if neither option is set, default to 32 bit mode
    if False == self.opts.mode_32 and False == self.opts.mode_64:
       self.opts.mode_32 = True;

    if self.opts.verbose == True and self.opts.quiet == True:
       self.opts.quiet = False

    self.nsd = number_spatial_dimensions

  def which(self, program):
    def is_exe(fpath):
        return os.path.exists(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None

  def is_32bit(self):
    return self.opts.mode_32

  def is_64bit(self):
    return self.opts.mode_64

  def set_cray(self):
    self.opts.cray = True

  # This is the main routine for running MyHyDE tests
  def call(self, cmd, logfile=None, ignore_status=False):
    status = 0

    # if on cray, replace mpiexec with aprun
    if self.opts.cray == True:
      if (cmd.find('mpiexec') == -1):
        # if env is set, skip past env variables before inserting aprun
        # otherwise aprun doesn't set env variables and tests fail
        if (cmd.find('env') != -1):
          index = cmd.rfind('=')
          new_cmd = cmd.find(' ', index)
          cmd = cmd[0:new_cmd+1] + 'aprun -q ' + cmd[new_cmd+1:]
        else:
          # no environment set, prepend aprun to requested command
          cmd = 'aprun -q ' + cmd
      else:
        # replace mpiexec with quiet aprun
        cmd = cmd.replace('mpiexec', 'aprun -q')

    if self.opts.verbose == True: print('---> ' + cmd)
    elif self.opts.quiet == True: pass
    else:                         print('  ' + cmd)

    syscmd(cmd, status, logfile, self.opts.verbose, ignore_status)

    return status

  # This is the main routine for cleaning mrhyde.log files after they are generated
  def clean_log(self, logfile='mrhyde.log'):
    status = 0
    # grab a hostname from the user's environment
    hostname = os.getenv('HOSTNAME')    
    # if there is a hostname, 
    if hostname != None:
      # on weaver, there's often a large amount of garbage that is printed to stdout, and consequently the logfile
      # this deletes the first 11 lines of garbage and additionally any lines that contain 'weaver'
      if hostname.find('weaver') != -1:
        status += os.system('sed -i \'1,11d;\' ' + logfile)
        status += os.system('sed -i \'/weaver/d\' ' + logfile)

    # IOSS may provide garbage to stdout which will interfere with logfile validation
    status += os.system('sed -i \'/IOSS/d\' ' + logfile)
    
    # return the sum of the exit codes from the shell commands
    return status

  def wrap_cmd(self, exe, root, np=None, args='', env=''):
    cmd = ''
    if (os.environ.has_key('PBS_NODEFILE') or \
        os.environ.has_key('SLURM_JOB_NODELIST')) and \
        self.opts.nprocs == None:
      cmd = '%s mpiexec p%s.exe %s %s' % (env,exe,args,root)
    elif self.opts.nprocs == None:
      cmd = '%s %s.exe %s %s' % (env,exe,args,root)
    else:
      if np is None:
        cmd = '%s mpiexec -n %i p%s.exe %s %s' % (env,self.opts.nprocs,exe,args,root)
      else:
        # user has overridden nprocs, use their value instead
        cmd = '%s mpiexec -n %i p%s.exe %s %s' % (env,np,exe,args,root)
    return cmd

  def mrhyde(self, root, args=''):
    status = 0
    log = '%s.log' % (root)
    cmd = self.wrap_cmd('mrhyde', root, self.opts.nprocs, args)
    status += self.call(cmd, log)
    return status

  def mrhyde_diff(self, aeps, reps, ref, test, root):
    status = 0
    log = '%s.log' % (root)
    cmd = self.wrap_cmd('mrhyde_diff',root,self.opts.nprocs, \
        '-aeps %g -reps %g -r1 %s.ref -r2 %s.rst'%(aeps,reps,ref,test))
    status += self.call(cmd, log)
    return status

  def mrhyde_opt(self, root, args=''):
    status = 0
    log = '%s.log' % (root)
    cmd = self.wrap_cmd('mrhyde_opt', root, self.opts.nprocs, args);
    status += self.call(cmd, log)
    return status

  def mrhyde_clean(self, root):
    status = self.call('mrhyde_clean %s'%root)
    return status

  def mkinp(self, root, physics, porder, Nt):
    ''' Create a input file for use with graph weights
    '''

    status = 0
    lines = []
    lines.append('eqntype  = %i\n' % (physics))
    lines.append('inttype  = 3\n')
    lines.append('p        = %i\n' % (porder))
    lines.append('Nt       = %i\n' % (Nt))
    lines.append('Ntout    = %i\n' % (Nt))
    lines.append('ntout    = 1\n')
    lines.append('dt       = 0.0025\n')
    lines.append('bmesh    = 1\n')

    mode = 'w'
    f = open('%s.inp' %(root), mode)
    f.writelines(lines)
    f.close()
    return status

  def mkcrv(self, root, nelems):
    ''' Create a curve file
    '''
    status = 0

    # setup to write binary file
    bmode = 'wb'
    fb = open('%s.cv' %(root), bmode)

    lines = []
    lines.append('** Curved Sides **\n\n')
    lines.append('1 Number of curve type(s)\n\n')
    # binary write number of curve types
    fb.write(struct.pack('i',1))
    if self.nsd == 2:
      lines.append('Straight\n')
      # binary write curve type, number of bytes in string
      fb.write(struct.pack('i',8))
      fb.write('Straight')
    elif self.nsd == 3:
      lines.append('Straight3d\n')
      # binary write curve type, number of bytes in string
      fb.write(struct.pack('i',10))
      fb.write('Straight3d')
    else:
      print('Error: Can not determine curve type (nsd=%i).' % (nsd))
      status = 1
    lines.append('skewed\n\n')
    # binary write user curve type name
    fb.write(struct.pack('i',6))
    fb.write('skewed')
    lines.append('%i Number of curved side(s)\n\n' %(nelems))
    # binary write number of arguments
    fb.write(struct.pack('i',0))
    # binary write number of curved sides
    fb.write(struct.pack('i',nelems))
    # write displacements
    # write lengths
    for elem_id in xrange(nelems):
      lines.append('%i 0 skewed\n' %(int(elem_id)))

    # binary write sides
    # write two ints for each side of each element
    for elem_id in xrange(nelems):
      fb.write(struct.pack('i',0))
      fb.write(struct.pack('i',0))

    fb.close()

    mode = 'w'
    f = open('%s.crv' %(root), mode)
    f.writelines(lines)
    f.close()


# ==============================================================================
# Summary-tag health checks
#
# Byte-diffing mrhyde.log against a gold file is not portable across BLAS/MPI
# implementations - low-order digits drift by roundoff. For FD-check /
# magnitude-scan tests, assert on the summary tags they already emit with
# tolerance-based predicates instead.
# ==============================================================================

import re

def extract_scalar(log_text, tag, group=1):
  """Pull the first floating-point number after `tag` on the same line.

  Example: extract_scalar(text, '[GRAD-CHECK] best rel_err =') matches
  '[GRAD-CHECK] best rel_err = 2.12e-05   at h = 1.00e-07'
  and returns 2.12e-05.
  """
  pat = re.escape(tag) + r'\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)'
  m = re.search(pat, log_text)
  if m is None:
    return None
  try:
    return float(m.group(group))
  except ValueError:
    return None

def extract_scalars(log_text, pattern):
  """Pull every floating-point number matched by the first capture group of
  `pattern`. Returns a list of floats (possibly empty)."""
  vals = []
  for m in re.finditer(pattern, log_text):
    try:
      vals.append(float(m.group(1)))
    except (ValueError, IndexError):
      pass
  return vals

def check_predicates(log_path, checks, verbose=True):
  """Run a list of (name, predicate) checks against a log file.

  `predicate` is a callable that takes the full log text and returns either:
    - (True, message)  -> pass
    - (False, message) -> fail

  Prints per-check outcomes when verbose. Returns 0 if all pass, else 1.
  """
  try:
    with open(log_path, 'r') as f:
      log_text = f.read()
  except (OSError, IOError) as e:
    print('  CHECK-FAIL: cannot open %s: %s' % (log_path, e))
    return 1

  all_ok = True
  print('  --- summary-tag checks on %s ---' % log_path)
  for name, pred in checks:
    ok, msg = pred(log_text)
    tag = 'PASS' if ok else 'FAIL'
    if verbose or not ok:
      print('  [%s] %s: %s' % (tag, name, msg))
    if not ok:
      all_ok = False
  return 0 if all_ok else 1


# Convenience predicate builders. Each returns a function suitable for
# check_predicates(); they close over the tag name and the tolerance.

def pred_scalar_lt(tag, upper, extractor=None):
  """Assert extract_scalar(text, tag) < upper."""
  def _p(text):
    v = (extractor(text) if extractor is not None else extract_scalar(text, tag))
    if v is None:
      return False, 'tag not found (%r)' % tag
    return (v < upper), 'value=%.3e < %.3e ? -> %s' % (v, upper, 'yes' if v < upper else 'no')
  return _p

def pred_scalar_gt(tag, lower, extractor=None):
  """Assert extract_scalar(text, tag) > lower."""
  def _p(text):
    v = (extractor(text) if extractor is not None else extract_scalar(text, tag))
    if v is None:
      return False, 'tag not found (%r)' % tag
    return (v > lower), 'value=%.3e > %.3e ? -> %s' % (v, lower, 'yes' if v > lower else 'no')
  return _p

def pred_scalar_in(tag, lower, upper, extractor=None):
  """Assert lower <= extract_scalar(text, tag) <= upper."""
  def _p(text):
    v = (extractor(text) if extractor is not None else extract_scalar(text, tag))
    if v is None:
      return False, 'tag not found (%r)' % tag
    return (lower <= v <= upper), 'value=%.3e in [%.3e, %.3e] ? -> %s' \
      % (v, lower, upper, 'yes' if (lower <= v <= upper) else 'no')
  return _p

def pred_scalar_rel(tag, expected, rel_tol, extractor=None):
  """Assert |extract_scalar(text, tag) - expected| / |expected| <= rel_tol."""
  def _p(text):
    v = (extractor(text) if extractor is not None else extract_scalar(text, tag))
    if v is None:
      return False, 'tag not found (%r)' % tag
    denom = abs(expected) if expected != 0.0 else 1.0
    rel = abs(v - expected) / denom
    ok = rel <= rel_tol
    return ok, 'value=%.6g vs expected=%.6g (rel=%.3e <= %.3e ? -> %s)' \
      % (v, expected, rel, rel_tol, 'yes' if ok else 'no')
  return _p

def pred_contains(needle, description=None):
  """Assert the string `needle` appears in the log."""
  def _p(text):
    ok = needle in text
    return ok, 'contains %r ? -> %s' % (needle if description is None else description,
                                        'yes' if ok else 'no')
  return _p
