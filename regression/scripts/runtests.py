#!/usr/bin/env python
#-------------------------------------------------------------------------------
# Simple interactive python script to run DGM tests.
#
# Authors:       Curtis C. Ober, William F. Spotz, K. Noel Belcourt
# Created:       2009/05/08
# Last modified: 2011/10/18
#-------------------------------------------------------------------------------

import sys
import os
import optparse
import platform
import copy
import stat
import shlex
import time
import string
import xml.dom.minidom
from subprocess import Popen, PIPE
import select
import glob
import getpass
from socket import gethostname

#-------------------------------------------------------------------------------

def getList(option, opt, value, parser):
  setattr(parser.values, option.dest, value.split(','))

def optional_arg(arg_default):
  def func(option,opt_str,value,parser):
    if parser.rargs and not parser.rargs[0].startswith('-S'):
      val=parser.rargs[0]
      parser.rargs.pop(0)
    else:
      val=arg_default
    setattr(parser.values,option.dest,val)
  return func

version = "runtests.py  Version 0.05 2011-10-18"

description = """
NAME
      runtests.py - test execution tool
      (version """ + version + """)

SYNOPSIS

      runtests.py [OPTIONS]

      This script searches for tests, runs them and reports back a
      pass/fail status for each test.  Tests are identified by an
      extension (default: '.tst'), and should be a script that return
      the status (e.g., $status) which is reported on output.

DESCRIPTION

      This script searches for tests starting at the top of the directory
      tree and recursively searches down from there.  The top-level
      directory can be the execution directory (default), or can be
      specified through a command line argument (-d).

      Each test should be a script that executes the desired commands
      for the problem.  The test should keep track of the status for
      all important commands, and return the status in the exit command.
      This status will be reported as pass (status=0) or fail (status!=0).

DESIRED NEW FEATURES
      * Be able to specific multiple directories with the -d option.
      * Filter tests based on
        - Grep for words within the *.inp file.

AUTHOR(S)
      Curtis Ober,   Sandia National Laboratories, ccober@sandia.gov
      Bill Spotz,    Sandia National Laboratories, wfspotz@sandia.gov
      Noel Belcourt, Sandia National Laboratories, kbelco@sandia.gov
"""

#===============================================================================

class Test:
  description = "A class for individual tests."
  def __init__(self, fullpath, startingDir):
      self.fullpath = fullpath
      self.fname = self.fullpath[len(startingDir)+1:]
      self.active = False
      self.print_keywords = False
      self.include_keywords = []
      self.exclude_keywords = []
      self.machines = []
      self.machinesExclude = []
      self.selected = True
      self.nprocs = 0
      self.avgRuntime = 0  # in hours
      self.forward_nodes = 0
      self.inversion_nodes = 0
      self.unknownOpts = []
      self.status = 1
      self.statusStr = 'inactive'
      self.skipped = 1
      self.jobid = 0
      self.runtime = 0
      self.expected_runtime = '00:10:00'
      self.test_args = ''
      self.stdout = ''
      self.stderr = ''
      self.p = ''
      self.stmt = ''
      self.index = 0
      self.starttime = 0

      # strip down to just non-numeric hostname
      hostname = (gethostname().split('.')[0]).split('-')[0]
      hostname = hostname.rstrip('1234567890')

      if not os.path.exists(self.fullpath):
        print('%s does not exist!') %(self.fullpath)
        self.status = 1
        self.statusStr = '!exist'
        self.skipped = 1
      elif not os.access(self.fullpath, os.X_OK):
        print('%s not executable!') %(self.fullpath)
        self.status = 1
        self.statusStr = '!exec'
        self.skipped = 1
      else:
        fin = open(self.fullpath, 'r')
        lines = fin.readlines()
        fin.close()

        for line in lines:
          words = line.split()
          if len(words) == 0: continue
          if words[0] == '#TESTING':
            words.remove('#TESTING')
            if words[0].lower() == 'active':    # Determine if active
              self.active = True
              self.statusStr = 'active'
              self.skipped = 0
            elif words[0] == '-k':              # Collect include keywords
              klist = words[1].split(',')
              for newkeyword in klist:
                if self.include_keywords.count(newkeyword) == 0:
                  self.include_keywords.append(newkeyword)
            elif words[0] == '-K':              # Collect exclude keywords
              klist = words[1].split(',')
              for newkeyword in klist:
                if self.exclude_keywords.count(newkeyword) == 0:
                  self.exclude_keywords.append(newkeyword)
            elif words[0] == '-m':              # Collect machines
              mlist = words[1].split(',')
              for newmachine in mlist:
                if self.machines.count(newmachine) == 0:
                  self.machines.append(newmachine)
            elif words[0] == '-M':              # Collect exclude machines
              mlist = words[1].split(',')
              for newmachine in mlist:
                if self.machinesExclude.count(newmachine) == 0:
                  self.machinesExclude.append(newmachine)
            elif words[0] == '-n':              # Determine number of processors
              self.nprocs = int(words[1])
            elif words[0] == '-t':              # Determine time test runs, in hours
              self.avgRuntime = int(words[1])
            elif words[0] == '--fn':            # Number of forward nodes
              self.forward_nodes = int(words[1])
            elif words[0] == '--in':            # Number of inversion nodes
              self.inversion_nodes = int(words[1])
            else:
              nl = line.strip()
              self.unknownOpts.append(nl[nl.find(' '):].strip())

        # Check if test has specified a host to run on or not run on.
        foundMatch = False
        for mword in self.machinesExclude:
          if mword == hostname:
            foundMatch = True                  # Host in exclude list
        if foundMatch and self.active:         # Exclude test explicitly
          self.selected = False
          self.statusStr = 'filtered'
          self.skipped = 1
        if not foundMatch and len(self.machines) > 0:
          for mword in self.machines:
            if hostname.find(mword) > -1:
              foundMatch = True                # Host in include list
          if not foundMatch and self.active:   # Exclude test; didn't match host
            self.selected = False
            self.statusStr = 'filtered'
            self.skipped = 1

  def matchesKeywords(self,inputKeywords):
    if inputKeywords == []: return
    for sword in inputKeywords:
      if self.include_keywords.count(sword) > 0:
        return True
    return False

  def matchKeywords(self,inputKeywords):
    if not self.active or inputKeywords == []: return
    foundMatch = False
    for sword in inputKeywords:
      # Handle case of tuple (e.g. -k short,Trilinos)
      klist = sword.split(',')
      if 1 < len(klist):
        foundMatch = True
        for kword in klist:
          if self.include_keywords.count(kword) == 0:
            foundMatch = False
      elif self.include_keywords.count(sword) > 0:
          foundMatch = True
      if foundMatch: return
    if not foundMatch:
      self.selected = False
      self.statusStr = 'filtered'
      self.skipped = 1

  def excludeKeywords(self,inputKeywordsExclude):
    if not self.active or inputKeywordsExclude == []: return
    foundMatch = False
    for sword in inputKeywordsExclude:
      # Handle case of tuple (e.g. -K short,Trilinos)
      klist = sword.split(',')
      if 1 < len(klist):
        foundMatch = True
        for kword in klist:
          if self.exclude_keywords.count(kword) == 0:
            foundMatch = False
      elif self.exclude_keywords.count(sword) > 0 or self.include_keywords.count(sword) > 0:
        foundMatch = True
      if foundMatch:
        self.selected = False
        self.statusStr = 'filtered'
        self.skipped = 1
    if foundMatch:
      self.selected = False
      self.statusStr = 'filtered'
      self.skipped = 1

  def forceActivate(self):
    if not self.active:
      self.active = True
      self.selected = True
      self.statusStr = 'forced'
      self.skipped = 0

  def matchProcessorRange(self,inputRange):
    if inputRange == []: return
    if len(inputRange) != 2:
      print('Length of processor range should be 2: ')
      print inputRange
      sys.exit(1)
    inRange = False
    np_min = int(inputRange[0])
    np_max = int(inputRange[1])
    if (self.nprocs == 0) or \
       ( (np_min <= self.nprocs) and (self.nprocs <= np_max) ):
      inRange = True
    if not inRange and self.active:
      self.selected = False
      self.statusStr = 'filtered'
      self.skipped = 1

# Searches for tests
def findTests(opts,dirname,names):
  # This search finds tests by simply asking does the filename
  # have the extension.
  for name in names:
    if name.endswith(opts.extension):
      opts.listOfTestFileNames.append(os.path.join(dirname,name))

class xml_document:
  description = "A class for xml documentation."
  def __init__(self, opts, startingDir, listOfTests):
    # Setup XML output info
    self.impl = xml.dom.minidom.getDOMImplementation()
    self.doc = self.impl.createDocument(None, "testsuite", None)
    self.root = self.doc.documentElement
    self.root.setAttribute('name', 'dgm')
    self.execDir = os.getcwd()
    self.startingDir = startingDir
    self.opts = opts
    self.skipped = 0
    self.listOfTests = listOfTests
    self.list_length = len(listOfTests)
    self.failed = 0
    self.totalstarttime = time.time()

  def __del__(self):
    self.root.setAttribute("failures", str(self.failed))
    self.root.setAttribute("errors", "0")
    self.root.setAttribute("skipped", "0")
    self.root.setAttribute("ignored", "0")
    self.root.setAttribute("time", str(time.time() - self.totalstarttime))
    xmlString = self.doc.toprettyxml()
    os.chdir(self.execDir)
    # The next two lines handle writing of the TEST-<filename>.xml file
    # don't comment these out because it will break the nightly regression.
    f = open('TEST-%s.xml' % (self.opts.package), 'w')
    f.write(xmlString)

  def testOutput(self,test,runtime):
    # Add results to XML
    elmt = self.doc.createElement('testcase')
    # Note: want to make sure this is not empty and doesn't end with a '.'
    (Shead, Stail) = os.path.split(self.startingDir)
    (Thead, Ttail) = os.path.split(test.fname)
    (head, tail) = os.path.split(test.fullpath)
    classname = os.path.join(Stail.replace('.',''), Thead.replace('.',''))
    classname = self.opts.package + '.' + classname
    classname = classname.rstrip('.')
    elmt.setAttribute('classname', classname)
    elmt.setAttribute('name', tail)
    elmt.setAttribute('time', str(runtime))
    if ( len(test.stdout) ):
      e = self.doc.createElement('system-out')
      e.appendChild(self.doc.createTextNode("\n"+test.stdout))
      elmt.appendChild(e)
    if ( len(test.stderr) ):
      e = self.doc.createElement('system-err')
      e.appendChild(self.doc.createTextNode("\n"+test.stderr))
      elmt.appendChild(e)
    if test.status:
      e = self.doc.createElement('failure')
      elmt.appendChild(e)
    if test.skipped:
      e = self.doc.createElement('skipped')
      elmt.appendChild(e)
    self.root.appendChild(elmt)
    self.skipped += test.skipped
    if self.opts.printKeywords :
      stmt2 = '%4i/%i %10s%8.2fs  np=%s    %55s    %s' \
              % (test.index+1, self.list_length, \
                 test.statusStr, runtime, test.nprocs, test.fname[0:-11], test.include_keywords)
      print stmt2 + ' '*(max(0,len(test.stmt)-len(stmt2)))
    else :
      stmt2 = '%4i/%i %10s%8.2fs  np=%s    %55s' \
              % (test.index+1, self.list_length, \
                 test.statusStr, runtime, test.nprocs, test.fname[0:-11])
      print stmt2 #+ ' '*(max(0,len(test.stmt)-len(stmt2)))
    

#===============================================================================

def serial_testing(opts,listOfTests,doc):
  passed = 0
  failed = 0
  skipped = 0
  for test in listOfTests:
    starttime = time.time()
    #if not opts.simpleReporting:
    #  tm = time.localtime()
    #  stmt = 'running on %s procs - %2i:%02i:%02i %s' \
    #       % (test.nprocs, tm.tm_hour, tm.tm_min, tm.tm_sec, test.fname)
    #  print stmt,
    #  sys.stdout.flush()
    # Unknown TESTING options are passed to test
    for option in test.unknownOpts:
      test.test_args += ' ' + option
    # split test path and add test options
    (head, tail) = os.path.split(test.fullpath)
    if head == '': head = '.'
    cmd = ["./" + tail]
    if opts.testArgs: cmd += shlex.split(opts.testArgs)
    if opts.computer != None: cmd += ['--computer=' + opts.computer]
    # default is 32 bit so only add if 64 bit
    if opts.mode_64: cmd += ['--64']
    if test.test_args: cmd += shlex.split(test.test_args)
    if opts.verbose==1: print 'Executing the command %s' % (cmd)
    # launch command and wait for completion
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, cwd=head)
    test.stdout, test.stderr = p.communicate()
    test.status = p.wait()
    # get completion status
    if test.status:
      failed += 1
      test.statusStr = 'fail'
    else:
      passed += 1
      test.statusStr = 'pass'
    # report test results
    #if not opts.simpleReporting:
    #  print '\b' * (len(stmt)+1),
    #  sys.stdout.flush()
    # including test time
    endtime = time.time()
    runtime = endtime-starttime
    doc.testOutput(test,runtime)
  return (passed,failed,skipped)

def launch_test(opts,test,tail):
  # Unknown TESTING options are passed to test
  for option in test.unknownOpts:
    test.test_args += ' ' + option
  # split test path and add test options
  (head, tail) = os.path.split(test.fullpath)
  if head == '': head = '.'
  cmd = ["./" + tail]
  if opts.testArgs: cmd += shlex.split(opts.testArgs)
  if opts.computer != None: cmd += ['--computer=' + opts.computer]
  # default is 32 bit so only add if 64 bit
  if opts.mode_64: cmd += ['--64']
  if test.test_args: cmd += shlex.split(test.test_args)
  tm = time.localtime()
  test.stmt = 'running on %s procs - %2i:%02i:%02i %s' \
       % (test.nprocs, tm.tm_hour, tm.tm_min, tm.tm_sec, test.fname)
  # launch command
  if opts.verbose==1 : print 'Executing the command %s' % test.fullpath
  p = Popen(cmd, stdout=PIPE, stderr=PIPE, cwd=head)
  return p

def find_test(listOfTests, n):
  if not listOfTests: return
  for a in listOfTests:
    if a.nprocs <= n:
      del listOfTests[listOfTests.index(a)]
      return a

def find_test_id(listOfTests, id):
  if not listOfTests: return
  for a in listOfTests:
    if a.jobid == id:
      return a

def smp_testing(opts,listOfTests,doc):
  n = 4096
  tail = ''
  passed = 0
  failed = 0
  skipped = 0
  subprocs = {} # map of process file descriptors
  rfds = []
  n_running = 0
  n_avail = opts.smp_cores
  done = False
  while not done:
    # fill available cores
    while listOfTests and 0 < n_avail:
      t = find_test(listOfTests,n_avail)
      if t:
        n_running += t.nprocs
        n_avail -= t.nprocs
        t.starttime = time.time()
        t.p = launch_test(opts,t,tail)
        subprocs[t.p.stdout.fileno()] = t
        subprocs[t.p.stderr.fileno()] = t
        rfds.append(t.p.stdout.fileno())
        rfds.append(t.p.stderr.fileno())
      else:
        break
    # if no more room or no more tests, wait for tests to complete
    test_terminated = False
    while not test_terminated and 0 < n_running:
      rl, wl, el = select.select(rfds,[],[],5)
      for fd in rl:
        if fd not in subprocs: continue
        test = subprocs[fd]
        if fd == test.p.stdout.fileno():
          chunk = os.read(fd,n)
          test.stdout += chunk
          while len(chunk) == n:
            chunk = os.read(fd,n)
            test.stdout += chunk
        elif fd == test.p.stderr.fileno():
          chunk = os.read(fd,n)
          test.stderr += chunk
          while len(chunk) == n:
            chunk = os.read(fd,n)
            test.stderr += chunk
      done_tests = []
      for key in subprocs:
        test = subprocs[key]
        if -1 == test.p.pid: continue
        pid, status = os.waitpid(test.p.pid,os.WNOHANG)
        if pid == test.p.pid:
          test.p.pid = -1
          done_tests.append(test)
          test_terminated = True
          test.status = status
          # get completion status
          if test.status:
            failed += 1
            doc.failed += 1
            test.statusStr = 'fail'
          else:
            passed += 1
            test.statusStr = 'pass'
          # remove descriptors from select
          del rfds[rfds.index(test.p.stdout.fileno())]
          del rfds[rfds.index(test.p.stderr.fileno())]
          n_running -= test.nprocs
          n_avail += test.nprocs
          endtime = time.time()
          runtime = endtime-test.starttime
          doc.testOutput(test,runtime)
      for test in done_tests:
        # delete process from dictionary
        del subprocs[test.p.stdout.fileno()]
        del subprocs[test.p.stderr.fileno()]
    if not listOfTests and n_running == 0: done = True
  return (passed,failed,skipped)

def batch_testing(opts,listOfTests,doc):
  # write salloc testing script
  f = open('srun.dat', 'w')
  f.write('%d\n'%len(listOfTests))
  for a in listOfTests:
    f.write('%s %s %d\n'%(a.fullpath,a.expected_runtime,a.nprocs))
  f.close()

  passed = 0
  failed = 0
  skipped= 0

  # request 16 nodes (128 cores) and wait for an allocation
  head = os.getcwd()
  # old WC ID is FY139321 (2013)
  # old WC ID is FY140238 (2014)
  # old WC ID is FY150006 (2015)
  cmd = ['salloc'] + ['--account=FY150006'] + ['-N1'] + ['--time=00:10:00'] + ['python'] + ['/home/kbelco/srun.py']
  print 'cmd = %s\n'%cmd
  print 'head = %s\n'%head
  p = Popen(cmd, stdout=PIPE, stderr=PIPE, cwd=head)
  stdout, stderr = p.communicate()
  status = p.wait()
  print 'stdout = %s\n'%stdout
  print 'stderr = %s\n'%stderr
  print 'status = %d\n'%status
  jobid = os.getenv('SLURM_JOB_ID')
  print 'jobid = %s\n'%jobid
  return (passed,failed,skipped)

def queryStdout_redsky(opts, test, Pstdout, hostname):
  job_state = 'U'
  if 0 == len(Pstdout):
    return job_state
  if hostname == 'curie':
    # toss first line of output
    loc_begin = Pstdout.find('\n')
    Pstdout = Pstdout[loc_begin+1:]
    begin = Pstdout.find('job_state') + 12
    job_state = Pstdout[begin:begin+1]
    if job_state == 'C':
      begin = Pstdout.find('resources_used.walltime')
      if 0 < begin:
        begin += 26
        walltime = Pstdout[begin:begin+8]
        sec = float(walltime[0:2]) * 3600
        sec += float(walltime[3:5]) * 60
        test.runtime = sec + float(walltime[6:8])
      # check test exit status
      # exit_status = -3
      begin = Pstdout.find('exit_status = ') + 14
      stat = int(Pstdout[begin:begin+3])
      if 0 != stat: job_state = ' F'
      # if opts.verbose==1:
        # print 'curie: job_state: ***%s*** job_status: ***%d***'%(job_state,stat)
      test.status = stat
  else:
    job_state = Pstdout[1:3]
    if job_state == 'CD':
      test.status = 0
    test.runtime = float(Pstdout[4:14])
  return job_state

def queue_results(opts, doc, listOfTests, hostname):
  passed = 0
  failed = 0
  skipped = 0
  completed_tests = []
  done = False
  # format of 'qstat -a' job info
  # 69557.sdb            kbelco   batch    3d_small_non-ort  19036     1  16    --  01:00 C 00:01
  # columns in qstat -a output for the relevant fields
  idb = 0  # job id [begin, end]
  ide = 22
  usb = 24 # user id [begin, end]
  use = 35
  stb = 99 # status [begin, end]
  ste = 100
  uname = getpass.getuser()
  done = False
  while not done:
    completed_tests = []
    time.sleep(3)
    # qstat -a to get all test status, find only completed tests
    if hostname == 'curie':
      s = ['qstat'] + ['-a']
      # parse result for our jobs
      p = Popen(s, stdout=PIPE, stderr=PIPE)
      Pstdout, Pstderr = p.communicate()
      status = p.wait()
      # list of completed tests
      # 1st field is test job id
      # 10th field is completion status
      # print 'stdout from qstat -a :', Pstdout
      # print 'stderr from qstat -a :', Pstderr
      # header is 5 lines that must be skipped
      begin = 0
      end = Pstdout.find('\n')
      # print Pstdout[begin:end]
      begin = end + 1
      Pstdout = Pstdout[begin:]
      begin = 0
      end = Pstdout.find('\n')
      # print Pstdout[begin:end]
      begin = end + 1
      Pstdout = Pstdout[begin:]
      begin = 0
      end = Pstdout.find('\n')
      # print Pstdout[begin:end]
      begin = end + 1
      Pstdout = Pstdout[begin:]
      begin = 0
      end = Pstdout.find('\n')
      # print Pstdout[begin:end]
      begin = end + 1
      Pstdout = Pstdout[begin:]
      begin = 0
      end = Pstdout.find('\n')
      # print Pstdout[begin:end]
      # print out first job info
      begin = end + 1
      Pstdout = Pstdout[begin:]
      begin = 0
      end = Pstdout.find('\n')
      while end != -1:
        user = Pstdout[usb:use];
        first_space = user.find(' ')
        if 0 < first_space:
          user = user[0:first_space]
        if uname == user:
          status = Pstdout[stb:ste]
          if 'C' == status:
            id = Pstdout[idb:ide]
            first_space = id.find(' ')
            if (0 < first_space):
              id = id[0:first_space]
            # check if this is one of our tests
            test = find_test_id(listOfTests, id)
            if test != None and test.statusStr == '?':
              completed_tests.append(test)
          elif 'R' == status:
            id = Pstdout[idb:ide]
        begin = end + 1
        end = Pstdout.find('\n')
        Pstdout = Pstdout[begin:]
    elif hostname == 'skybridge':
      # query sqlog for tests whose completion status we do not know
      for test in listOfTests:
        if test.statusStr == '?':
          completed_tests.append(test)
    # loop over completed tests getting completion status
    for test in completed_tests:
      statusStr = '?'
      (Shead, Stail) = os.path.split(test.fullpath)
      if hostname == 'curie':
        s = ['qstat'] + ['-f'] + ['%s' % (test.jobid)]
      else:
        s = ['sqlog'] + ['--no-header'] + ['--format=state,runtime_s'] + ['--jobids=%s' % (test.jobid)]
      p = Popen(s, stdout=PIPE, stderr=PIPE, cwd=Shead)
      Pstdout, Pstderr = p.communicate()
      status = p.wait()
      job_state = queryStdout_redsky(opts, test, Pstdout, hostname)
      if hostname == 'curie' and 'C' == job_state:
        passed += 1
        statusStr = 'pass'
      elif 'CD' == job_state:
        passed += 1
        statusStr = 'pass'
      elif ' F' == job_state:
        failed += 1
        statusStr = 'fail'
      elif 'CA' == job_state:
        # job was canceled
        failed += 1
        statusStr = 'fail'
      elif 'NF' == job_state:
        # node failed
        failed += 1
        statusStr = 'fail'
      elif 'TO' == job_state:
        # job timed out
        failed += 1
        statusStr = 'fail'
      if statusStr != '?':
        # read log file if present
        loc = Stail.find('.tst')
        str = '%s/%s.log' % (Shead,Stail[0:loc])
        Pstdout = ''
        if os.path.exists(str):
          f = open(str, 'r')
          lines = f.readlines()
          f.close()
          Pstdout = '\ncontents of %s\n' % (str)
          Pstdout += "".join(lines)
        # read slurm output file if present
        if hostname != 'curie':
          str2 = '%s/slurm-%s.out' % (Shead,test.jobid)
          if os.path.exists(str2):
            f = open(str2, 'r')
            lines = f.readlines()
            f.close()
            Pstdout += '\ncontents of %s\n' % (str2)
            Pstdout += "".join(lines)
        test.stdout = Pstdout
        test.stderr = Pstderr
        test.statusStr = statusStr
        doc.testOutput(test,test.runtime)
      else:
        time.sleep(0.1)
      sys.stdout.flush()
    # terminate loop if every test has been processed
    done = True
    for t in listOfTests:
      if t.statusStr == '?':
        done = False;
        break;
  return (passed,failed,skipped)

def queue_testing(opts, listOfTests, doc, hostname):
  # first clean test directories
  clean_test_directories(opts, listOfTests)
  cores_per_node = 8
  if hostname == 'glory' or hostname == 'skybridge' or hostname == 'curie':
    cores_per_node = 16
  partition = 'ec'
  # if hostname == 'glory' : partition = 'ecis'
  # setup for queue submission
  home = os.getenv('HOME')
  testdir = home + '/test_scripts/'
  # cleanup previous tests
  if os.path.exists(testdir):
    os.system('rm %s*' % (testdir))
  else:
    os.system('mkdir %s' % (testdir))
    os.chmod(s, 0755)
  for test in listOfTests:
    (head, tail) = os.path.split(test.fullpath)
    nodes = 1
    if cores_per_node < int(test.nprocs):
      nodes = int(test.nprocs) / cores_per_node
      nodes += int(test.nprocs) % cores_per_node
    # if head == '' or head[0] == '.': head = os.getcwd() + head[1:]
    # write submission script
    s = testdir + '%s' % (tail)
    f = open(s, 'w')
    f.write('#!/bin/sh\n')
    if hostname == 'curie':
      f.write('#PBS -l nodes=%s:ppn=16,walltime=%s\n'%(str(nodes),test.expected_runtime))
      f.write('#PBS -N %s\n'%tail)
      f.write('cd $PBS_O_WORKDIR\n')
      f.write('pwd\n')
      f.write('source /lscratch2/kbelco/'+opts.product+'/regression/jenkins/'+opts.product+'-env-setup-curie.bsh\n')
    f.write('set -o pipefail\n')
    f.write('cd %s\n' % (head))
    f.write('./'+tail)
    if test.nprocs != '?' and 1 <= int(test.nprocs):
      f.write(' -n ')
      f.write('%s' % (test.nprocs))
    else:
      test.nprocs = '1'
    if opts.testArgs: f.write(' %s'%(opts.testArgs))
    # default is 32 bit so only add if 64 bit
    if opts.mode_64: f.write(' --64')
    loc = tail.find('.tst')
    f.write(' 2>&1 | tee %s.noel\n'%(tail))
    f.write('status=$?\n')
    f.write('cp %s.noel %s.log\n'%(tail,tail[:loc]))
    f.write('rm -f %s.noel\n'%(tail))
    f.write('exit $status\n')
    f.close()
    os.chmod(s, 0755)
    # submit using sbatch or qsub on curie
    qsub_cmd = ['sbatch'] + ['--partition=' + partition] + ['--time=' + test.expected_runtime] + ['--job-name=' + tail] + ['--account=FY150006'] + ['--nodes=' + str(nodes)] + [s]
    if hostname == 'curie':
      qsub_cmd = ['qsub'] + ['-A FY150006'] + [s]
    p = Popen(qsub_cmd, stdout=PIPE, stderr=PIPE, cwd=head)
    test_stdout, test_stderr = p.communicate()
    status = p.wait()
    if hostname == 'curie':
      loc = 0
      result = test_stdout[loc:loc+20]
      test.jobid = result[0:20]
      first_space = test.jobid.find('b')
      test.jobid = test.jobid[0:first_space+1]
    else:
      loc = test_stdout.find('Submitted batch job ')
      result = test_stdout[loc+20:loc+28]
      test.jobid = result[0:8]
    if opts.verbose==1 : print 'Executing the command %s with jobid %s' % (qsub_cmd, test.jobid)
    test.statusStr = '?'
  # wait for tests to complete
  return queue_results(opts, doc, listOfTests, hostname)

def clean_test_directories(opts,listOfTests):
  for t in listOfTests:
    # split test path and add test options
    (head, tail) = os.path.split(t.fullpath)
    if head == '': head = '.'
    cmd = ["./" + tail]
    # add the clean option
    cmd += [ '-c' ]
    # launch clean command
    if opts.verbose==1 : print 'Executing the command %s' % cmd
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, cwd=head)

def cray_testing(opts,listOfTests):
  # first clean test directories
  clean_test_directories(opts, listOfTests)
  # just forwards to smp testing
  return smp_testing(opts,listOfTests,doc)

def main():

  # strip down to just non-numeric hostname
  hostname = (gethostname().split('.')[0]).split('-')[0]
  hostname = hostname.rstrip('1234567890')

  # Define the command line options
  p = optparse.OptionParser(description)

  p.add_option("-a", dest="testArgs", default='', \
                     action="store", type="string", \
                     help='''Specify the arguments to pass to the tests.''')

  p.add_option("-b", dest="batch", default=False, \
                     action="store_true", \
                     help='''For redsky, jobs run with srun via salloc.''')

  p.add_option("-c", dest="computer", default=None, \
                     action="store", \
                     help='''Computer to run on (e.g.  SandiaCray, 
                     SandiaSkybridge, MacPro, Linux)''')

  p.add_option("-d", dest="startingDir", default='.', \
                     action="store", type="string", \
                     help='''Specify the path to top of directory tree to
                             search for tests.  Without this option, the
                             current directory is used.''')

  p.add_option("-e", dest="extension", default='.tst', \
                     action="store", type="string", \
                     help='''Override the default extension, ".tst",
                             with the specified extension.''')

  p.add_option("-f", "--force", dest="force", default=False, \
                     action="store_true", \
                     help='''force all tests to be run inspite of
                             inactive defines.  Filtering is still
                             honored.''')

  p.add_option("-i", dest="info", default=False, action="store_true", \
                     help='''Report information on tests and do not
                             execute them.''')

  p.add_option("--include-all", dest="includeAll", default=False, \
                     action="store_true", \
                     help=''''Inactive' and keyword-filtered tests are
                             included in the runtests.py output report
                             and tallying, but are not run.  By default
                             these tests are excluded from the reports
                             and tallying.''')

  p.add_option("-k", dest="include_keywords", default=[], \
                     action="append", type="string", \
                     help='''List of keywords required by tests to run.
                             Multiple keywords are deliminated by commas
                             without intervening spaces.

                             Example with multiple -k options:
                             e.g. -k acoustic -k short selects tests that are
                                  either acoustic or short tests.

                             Example with comma separated -k option:
                             e.g. -k elastic,Trilinos selects tests that
                                  are both elastic and Trilinos.

                             Composite example showing both comma separated
                             and multiple -k options:
                             e.g. -k acoustic,short -k Trilinos,elastic 
                             selects tests that have both the acoustic and
                             short keywords, and selects additional tests that 
                             have both the elastic and Trilinos keywords.''')

  p.add_option("-K", dest="exclude_keywords", default=[], \
                     action="append", type="string", \
                     help='''List of keywords to exclude tests from running.
                             Multiple keywords are deliminated by commas
                             without intervening spaces.

                             Example with multiple -K options:
                             e.g. -K Trilinos -K short excludes tests that are
                                  either Trilinos tests or short tests.

                             Example with comma separated -K option:
                             e.g. -K Trilinos,short only excludes a test if 
                                  that test has both the Trilinos and short 
                                  keywords.

                             Composite example with both -k and -K options:
                             e.g. -k acoustic,short -K Trilinos,Rol  will first
                                  select tests that are both acoustic and short, 
                                  and will then remove any of those tests that 
                                  have both the Trilinos and Rol keywords.''')

  p.add_option("--list-keywords", dest="listKeywords", default=False, \
                     action="store_true", \
                     help='''List the keywords of each test.''')

  p.add_option("-m", dest="machines", default=[], \
                     action="callback", type="string", callback=getList, \
                     help='''List of machines to run on.  Multiple machines
                             are deliminated by commas (",") without spaces.
                             This option is primarily used in tests to
                             specify which machines to run on.''')

  p.add_option("-M", "--skip-machines", dest="machinesExclude", default=[], \
                     action="callback", type="string", callback=getList, \
                     help='''List of machines to exclude tests from running.
                             Multiple machines are deliminated by commas
                             (",") without spaces.  This option is primarily
                             used in tests to specify which machines not to
                             run on.''')

  p.add_option("-n", dest="nprocs", default=0, \
                     action="store", type="int", metavar="nprocs", \
                     help='''Override the number of processors with this
                             value in each and every test.  If set to zero,
                             the test value will be run with the value from
                             PBS or SLURM.  If PBS or SLURM is not available
                             the problems are run in serial.  Tests can
                             specify the number of processors to use by
                             including the comment line, "#TESTING -n <int>",
                             within the tst file.''')

  p.add_option("--nrange", dest="nrange", default=[], \
                     action="callback", type="string", callback=getList, \
                     help='''Filter tests based on whether test's nprocs
                             falls within specified processor range.  For
                             example, '--nrange 2,8' will run tests which
                             satisfy 2 <= n <= 8. Minimum and maximum values
                             are deliminated by commas (",") without spaces.
                             Default will run all tests.''')

  p.add_option("-N", dest="nonrecursive", default=False, action="store_true", \
                     help='''Do not recursively look for tests from the
                             starting directory (-d).  Just run tests
                             specified in the starting directory.''')

  p.add_option("-p", dest="package", default='package', \
                     action="store", type="string", \
                     help='''Specify the name of the xml output package.''')

  p.add_option("-P", dest="product", default='MrHyDE', \
                     action="store", type="string", \
                     help='''Specify the name of the product were building.''')

  p.add_option("-q", dest="queue", default=False, \
                     action="store_true", \
                     help='''For clusters, submit jobs to default queue.''')

  # If this option is present, it is ignored and not passed to the
  # individual tests as this is the default testing mode.
  p.add_option("--32", dest="mode_32", default=False, action="store_true",\
                     help='''Use 32bit executables''')

  # If this option is present, it is passed to each individual test along
  # with the -a pass-through arguments.  This is done so tests can query
  # if they're running in 64 bit mode which would permit them to use
  # different reference files for diff'ing test results.  Many of our
  # output files have Json headers that encode the SIZE and ORDINAL
  # so we must use the correct executable to diff the matching output
  # files, if they're mismatched, the test may fail.
  p.add_option("--64", dest="mode_64", default=False, action="store_true",\
                     help='''Use 64bit executables''')

  p.add_option("-s", dest="simpleReporting", default=False,action="store_true",\
                     help='''Use simple reporting.  This gives output
                             only after test is finished.  Useful for
                             nightly regression reports where regular
                             output is hard to read.''')

  p.add_option("-S", dest='smp_cores', action='callback', \
                     callback=optional_arg(0), \
                     help='''For SMP architectures, specify number of
                             cores to run tests on.  The default is to
                             run tests serially (sequentially).  If no
                             argument is passed to -S, we interrogate
                             the system for the number of available
                             cores to use ($DGM_DIST/util/ncpus).''')

  p.add_option("-t", dest="listOfTestFileNames", default=[], \
                     action="callback", type="string", callback=getList, \
                     help='''List of tests to run relative to -d. Multiple
                             tests are deliminated by commas (",") without
                             spaces.''')

  p.add_option("-T", dest="testFileName", default=None, \
                     action="store", type="string", \
                     help='''Name of file which contains a list of tests
                             to run relative to -d. One test per line.''')

  p.add_option("-v", dest="verbose", default=False, action="store_true", \
                     help='''Echo the commands to output.''')
                     
  p.add_option("--print-keywords", dest="printKeywords", default=False, \
                     action="store_true", \
                     help='''Print the keywords with results of each test.''')
                     
  # Parse the command line options
  opts, args = p.parse_args()

  ######################
  # Option sanity checks
  ######################

  # Script takes no arguments
  if args != []:
    print "Error - Currently there are no arguments to runtests.py. Only options."
    sys.exit(1)

  # Remove any trailing slash from starting directory
  if opts.startingDir[-1] == "/":
    opts.startingDir = opts.startingDir[:-1]

  # If test file names are specified, change to full relative path
  if opts.listOfTestFileNames != []:
    tmpList = [ os.path.join(opts.startingDir,fname) for
                fname in opts.listOfTestFileNames ]
    opts.listOfTestFileNames = tmpList

  # error if both options are supplied: --32 and --64
  if opts.mode_32 and opts.mode_64:
    print 'Error: cannot specify both --32 and --64 bit mode'
    sys.exit(0)
  # if neither option is set, default to 32 bit mode
  if False == opts.mode_32 and False == opts.mode_64:
    opts.mode_32 = True;

  # Get tests from file
  if opts.testFileName != None:
    fin = open(opts.testFileName, 'r')
    fnames = fin.readlines()
    fin.close()
    tmpList=[os.path.join(opts.startingDir,fname.strip()) for fname in fnames]
    opts.listOfTestFileNames = opts.listOfTestFileNames + tmpList

  # Fill out the list of test file names if none specified
  if opts.listOfTestFileNames == []:
    # Non-recursive case
    if opts.nonrecursive:
      sname = os.path.join(opts.startingDir,'*%s' % opts.extension)
      opts.listOfTestFileNames = glob.glob(sname)
    # Recursive case
    else:
      if os.path.exists(opts.startingDir):
        os.path.walk(opts.startingDir, findTests, opts)
      if opts.listOfTestFileNames == []:
        print "Did not find any tests with the extension, '%s'." \
              % opts.extension
        sys.exit(1)

  # Convert list of test file names to list of Test objects
  listOfTests = []
  for fname in opts.listOfTestFileNames:
    a = Test(fname, opts.startingDir)
    a.matchKeywords(opts.include_keywords)
    a.excludeKeywords(opts.exclude_keywords)
    a.matchProcessorRange(opts.nrange)
    if (opts.force): a.forceActivate()
    # scale nprocs by avgRuntime is it's been set in test script
    if 0 < a.avgRuntime: a.nprocs *= int(a.avgRuntime)
    # set expected_runtime,
    if a.nprocs != None and a.matchesKeywords(["long"]) == True:
      a.expected_runtime = '00:00:10'
    elif a.matchesKeywords(["medium"]) == True:
      a.expected_runtime = '00:00:10'
    # add test
    listOfTests.append(a)

  # get number of SMP cores
  if opts.smp_cores != None:
    # convert to integral type
    opts.smp_cores = int(opts.smp_cores)

  # if no argument to -S on Cray, error out as
  # this only asks the head node how many cores it has
  # require user to input value on Cray
  if opts.testArgs.find('y') != -1 and opts.smp_cores == 0:
    print 'Error: pass -S an argument for number of cores to use on Cray'
    sys.exit(-2)

  # User passed -S without value, query machine for number of cores
  if opts.smp_cores == 0:
    dgm_dist = os.getenv('DGM_DIST')
    if None != dgm_dist:
      ncpus = dgm_dist + '/util/ncpus'
      if os.path.exists(ncpus):
        p = Popen(dgm_dist+'/util/ncpus', stdout=PIPE, stderr=PIPE)
        Pstdout, Pstderr = p.communicate()
        status = p.wait()
        opts.smp_cores = int(Pstdout)

  # Remove tests that are not active, not selected, or too many cores
  if not opts.includeAll:
    tempListOfTests = copy.copy(listOfTests)
    for a in tempListOfTests:
      # skip inactive, not selected or tests that are too big
      if (not a.active) or (not a.selected) or \
        (1 < opts.smp_cores and opts.smp_cores < a.nprocs):
        listOfTests.remove(a)
      else:
        # Initialize test arguments
        a.test_args = ''
        if opts.nprocs == 0:
          if a.nprocs != None:
            if 0 == a.nprocs: a.nprocs = 1
            option = '-n %s' % (a.nprocs)
            a.test_args += ' ' + option
          else:
            a.nprocs = '?'
        else:
          option = '-n %s' % (opts.nprocs)
          a.test_args += ' ' + option
          a.nprocs = opts.nprocs

  # order tests by number of cores (longest to shortest)
  listOfTests.sort(key=lambda k: k.nprocs)
  listOfTests.reverse()

  # set test index
  for a in listOfTests:
    a.index = listOfTests.index(a)

  # Initialize test suite
  execDir = os.getcwd()
  startingDirA = os.path.abspath(opts.startingDir)
  passed  = 0
  failed  = 0
  skipped = 0
  print
  print time.asctime()
  print 'Test Results from Directory: ' + startingDirA
  print 'Total number of test(s): %i' % (len(listOfTests))
  if opts.info: print ' Nprocs    Test Name'
  print '-----------------------------------------------------------------------------------------------'

  # user wants info on tests, don't run tests
  if opts.info:
    for a in listOfTests:
      print '    %d\t%s'%(a.nprocs,a.fname)
    sys.exit(0)

  allKeywordsInclude = []
  allKeywordsExclude = []
  allMachines = []
  allMachinesExclude = []
  if opts.listKeywords:
    for a in listOfTests:
      if 0 < len(a.include_keywords):
        print '  %s -k' % a.fname,
        for k in a.include_keywords:
          allKeywordsInclude.append(k)
          print ' %s' % (k),
        print
      if 0 < len(a.exclude_keywords):
        print '  %s -K' % a.fname,
        for k in a.exclude_keywords:
          allKeywordsExclude.append(k)
          print ' %s' % (k),
        print
      if 0 < len(a.machines):
        print '  %s -m' % a.fname,
        for k in a.machines:
          allMachines.append(k)
          print ' %s' % (k),
        print
      if 0 < len(a.machinesExclude):
        print '  %s -M' % a.fname,
        for k in a.machinesExclude:
          allMachinesExclude.append(k)
          print ' %s' % (k),
        print
      sys.stdout.flush()

  if opts.listKeywords:
    print
    uniqueKeywords = list(set(allKeywordsInclude))
    uniqueKeywords.sort()
    if 0 < len(uniqueKeywords):
      print 'Unique include keywords (-k):',
      for k in uniqueKeywords:
        print ' %s' % (k),
      print
    uniqueKeywordsExclude = list(set(allKeywordsExclude))
    uniqueKeywordsExclude.sort()
    if 0 < len(uniqueKeywordsExclude):
      print 'Unique exclude keywords (-K):',
      for k in uniqueKeywordsExclude:
        print ' %s' % (k),
      print
    uniqueMachines = list(set(allMachines))
    uniqueMachines.sort()
    if 0 < len(uniqueMachines):
      print 'Unique machines (-m):',
      for m in uniqueMachines:
        print ' %s' % (m),
      print
    uniqueMachinesExclude = list(set(allMachinesExclude))
    uniqueMachinesExclude.sort()
    if 0 < len(uniqueMachinesExclude):
      print 'Unique exclude machines (-M):',
      for m in uniqueMachinesExclude:
        print ' %s' % (m),
      print
    print
    sys.stdout.flush()
    return

  # Loop over test objects
  totalstarttime = time.time()

  # construct xml document
  doc = xml_document(opts, os.path.abspath(opts.startingDir), listOfTests)

  # different types of ways to run tests on different systems
  # serial (sequential), smp, batch, alloc, cray
  if hostname == 'redsky' or hostname == 'glory' or hostname == 'skybridge' or hostname == 'curie':
    # ensure we don't enable both batch and queue mode
    if opts.batch and opts.queue:
      print 'Error: cant request both batch and queue testing mode'
      sys.exit(-1)
    if opts.batch:
      passed, failed, skipped = batch_testing(opts,listOfTests,doc)
    elif opts.queue:
      passed, failed, skipped = queue_testing(opts,listOfTests,doc, hostname)
    else:
      if opts.smp_cores != None and 0 < opts.smp_cores:
        passed, failed, skipped = smp_testing(opts,listOfTests,doc)
      else:
        passed, failed, skipped = serial_testing(opts,listOfTests,doc)
  elif hostname == 'curie' or hostname == 'muzia':
    passed, failed, skipped = cray_testing(opts,listOfTests,doc)
  else:
    if opts.smp_cores != None and 0 < opts.smp_cores:
      # user specified -S with or without explicit number of cores
      passed, failed, skipped = smp_testing(opts,listOfTests,doc)
    else:
      passed, failed, skipped = serial_testing(opts,listOfTests,doc)

  # Summary output
  print '-----------------------------------------------------------------------------------------------'
  print ' Pass: %i    Fail: %i    Skipped: %i    Total: %i' \
        % (passed, failed, skipped, doc.list_length)

  totalendtime = time.time()
  totalruntime = totalendtime-totalstarttime
  print
  print 'Total Runtime: %10.2fs' % (totalruntime)
  print time.asctime()

  sys.exit(0)

#-------------------------------------------------------------------------------

if __name__ == "__main__":
  main()
