# To glob *.cpp files and if a directory change is desired
import glob, os
# To modify the file in the most logical way I know
import fileinput

# Glob for all cpp files in current directory
# os.chdir(".") # use this to change directory if needed
filenames = glob.glob("*.cpp")
print(filenames)

for filename in filenames:
  print('Opening',filename,'...')

  # Dump the file to "lines"
  f = open(filename,"r")
  lines = f.readlines()
  f.close()

  # Create an empty set (avoids duplicates) of parameters and start filling it
  functionnames = set()
  for line in lines:
    if 'fs.get' in line:
      #print(line,end='')
      # The first double-quoted word after fs.get is the input parameter (dependent on syntax in each input deck)
      functionnames.add(line.split('fs.get')[1].split('"')[1].split('"')[0])

  # Print as a sanity check
  #print(functionnames)
  #print('Found', len(functionnames), 'function names!')

  headerfilename = filename.replace("cpp","hpp")

  # NOTE: There's a lot of logic below to check if a class already
  # has some documention. If so, it'll print a warning so some touchups
  # can be done by hand.
  hasClassDoc = False
  inClassDoc = False
  alreadyPrintedDoc = False
  for line in fileinput.input(headerfilename, inplace=1):
    # If we see /**, it's probably documentation.
    if '/**' in line and not alreadyPrintedDoc:
      hasClassDoc = True
      inClassDoc = True
    # We've exited documentation
    if '*/' in line:
      inClassDoc = False
    # If we see { before a class declaration, and we're not on a class line, then it's not a class documentation block
    if '{' in line and not alreadyPrintedDoc and not inClassDoc and not 'class' in line:
      hasClassDoc = False
    # If the next line is a class declaration, let's start dumping documentation
    # NOTE: to avoid duplicating documentation in files, check if "hasClassDoc" is true
    if 'class' in line and not alreadyPrintedDoc and not inClassDoc:
      alreadyPrintedDoc = True
      print('  /**')
      print('   * \\brief %s physics class.' % filename.replace(".cpp",""))
      print('   *')
      print('   * This class computes volumetric residuals for the physics described by the following weak form:')
      print('   * \\f{eqnarray*}')
      print('   *   \\dots')
      print('   * \\f}')
      print('   * Where the unknown ___ is the ___.')
      print('   * The following functions may be specified in the input.yaml file:')
      # Print the potential functions that can be specified in the function manager
      for functionname in functionnames:
        print('   *   - "%s" is the %s.' % (functionname,functionname))
      print('   */')
    # keep original lines (need end='' to avoid double line breaks since the strings we print contain '\n')
    print(line,end='')
  # Warn in case we already documented the class
  if hasClassDoc:
    print('WARNING: File',headerfilename,'may already have doxygen ')


