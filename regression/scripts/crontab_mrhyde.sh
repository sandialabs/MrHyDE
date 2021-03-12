#!/bin/bash

# Crontab e
# crontab -e 

#00 00 * * * kinit -k -t bartv.keytab bartv;/scratch/bartv/regression-mrhyde/crontab_mrhyde.sh >& crontab.out

#now=$(date +"%m_%d_%Y")
#cd /scratch/bartv/regression-mrhyde
#. ~bartv/.bashrc

/scratch/bartv/regression-mrhyde/regression-mrhyde.sh 

# sender="bartv@sandia.gov"
# receiver="bartv@sandia.gov"
# subject="MILO Regression Testing Results"

# echo "MILO Regression Testing" mail $receiver -s "$subject" | mutt -a "/scratch/bartv/regression-mrhyde/regression-"${now}.out"
cat /scratch/bartv/regression-mrhyde/mrhyde/regression/runtests-opt.out | mail -s "MILO regression" bartv@sandia.gov
cat /scratch/bartv/regression-mrhyde/mrhyde/regression/runtests-opt.out | mail -s "MILO regression" tmwilde@sandia.gov
cat /scratch/bartv/regression-mrhyde/mrhyde/regression/runtests-opt.out | mail -s "MILO regression" dtseidl@sandia.gov
