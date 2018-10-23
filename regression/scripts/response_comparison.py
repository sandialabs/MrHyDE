import sys
import re

# arguments:
# argv[1] = name of output file
# argv[2] = response name
# argv[3] = response value
# argv[4] (optional) = tolerance (defaults to 1e-6)

try:
  print 'len(sys.argv) = '+str(len(sys.argv))
  if not (len(sys.argv)==4 or len(sys.argv)==5):
    raise Exception('Failure')

  print "opening \"" + sys.argv[1] +"\""
  f = open(sys.argv[1]);

  valEst = None;
 
  valPat = re.compile('.*Response "'+sys.argv[2]+'" = (.*)$')
 
  for line in f:
    valMatch = valPat.match(line)

    if valMatch:
      valEst = float(valMatch.group(1))

  if valEst==None:
    raise Exception(sys.argv[2] + " response not found!")

  valExact = float(sys.argv[3])
  error = abs(valEst - valExact)
  tolerance = 1.0e-6

  print 'Response = '+sys.argv[2]
  print '  True value     = '+str(valExact)
  print '  Computed value = '+str(valEst)
  print '  Error          = '+str(error)
  print '  Tolerance      = '+str(tolerance)

  if len(sys.argv)==5:
    tolerance = float(sys.argv[4])

  if error>tolerance:
    raise Exception('Error does not meet tolerance')

except Exception, e:
  print "Test Failed"
  print e
else:
  print "Test Passed"
