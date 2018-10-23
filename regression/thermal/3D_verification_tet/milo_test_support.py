#!/usr/bin/env python
#-------------------------------------------------------------------------------

import optparse
import subprocess as sp
import sys, os
import struct

# ==============================================================================

def syscmd(cmd, status=0, logfile=None, verbose=False, ignore_status=False):

  internal_status = 0

  if verbose: print cmd
  p = sp.Popen(cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)

  stdout = ''
  stderr = ''
  if verbose == True:
    # if len(stdout) > 0: print stdout
    while True:
      out = p.stdout.read(1)
      if out == '' and p.poll() != None:
        break
      if out != '':
        sys.stdout.write(out)
        sys.stdout.flush()
        stdout += out

    stderr = p.stderr.read()
  else:
    stdout, stderr = p.communicate()
  internal_status = p.wait()

  if stderr: print stderr
  if logfile:
    f = open(logfile, 'w')
    f.writelines(stdout)
    f.close()
  if not ignore_status:
    status += internal_status
    if internal_status != 0:
      print '  ==> Execution failed with status = %i!\n' %(internal_status)
      sys.exit(status)

  return status

# ==============================================================================
class milo_test_support:
  """Class to help support milo tests"""
  def __init__( self, description = 'MILO testing script.', \
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
       print 'Error: cannot specify both --32 and --64 bit mode'
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

    if self.opts.verbose == True: print '---> ' + cmd
    elif self.opts.quiet == True: pass
    else:                         print '  ' + cmd

    syscmd(cmd, status, logfile, self.opts.verbose, ignore_status)

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

  def milo(self, root, args=''):
    status = 0
    log = '%s.log' % (root)
    cmd = self.wrap_cmd('milo', root, self.opts.nprocs, args)
    status += self.call(cmd, log)
    return status

  def milo_diff(self, aeps, reps, ref, test, root):
    status = 0
    log = '%s.log' % (root)
    cmd = self.wrap_cmd('milo_diff',root,self.opts.nprocs, \
        '-aeps %g -reps %g -r1 %s.ref -r2 %s.rst'%(aeps,reps,ref,test))
    status += self.call(cmd, log)
    return status

  def milo_opt(self, root, args=''):
    status = 0
    log = '%s.log' % (root)
    cmd = self.wrap_cmd('milo_opt', root, self.opts.nprocs, args);
    status += self.call(cmd, log)
    return status

  def milo_clean(self, root):
    status = self.call('milo_clean %s'%root)
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
      print 'Error: Can not determine curve type (nsd=%i).' % (nsd)
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

    return status
