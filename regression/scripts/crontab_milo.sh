#!/bin/bash

# Crontab e
# crontab -e 

#00 00 * * * kinit -k -t bartv.keytab bartv;/scratch/bartv/regression-milo/crontab_milo.sh >& crontab.out

#now=$(date +"%m_%d_%Y")
#cd /scratch/bartv/regression-milo
#. ~bartv/.bashrc

/scratch/bartv/regression-milo/regression-milo.sh 

# sender="bartv@sandia.gov"
# receiver="bartv@sandia.gov"
# subject="MILO Regression Testing Results"

# echo "MILO Regression Testing" mail $receiver -s "$subject" | mutt -a "/scratch/bartv/regression-milo/regression-"${now}.out"
cat /scratch/bartv/regression-milo/milo/regression/runtests-opt.out | mail -s "MILO regression" bartv@sandia.gov
cat /scratch/bartv/regression-milo/milo/regression/runtests-opt.out | mail -s "MILO regression" tmwilde@sandia.gov
cat /scratch/bartv/regression-milo/milo/regression/runtests-opt.out | mail -s "MILO regression" dtseidl@sandia.gov
