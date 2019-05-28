#!/bin/env python
#
import optparse
import pickle
import glob
import sys
import subprocess
import os 
from os.path import exists,join,basename

splitDescriptionFile = 'description.pkl'
rootResDirs = ['res','trees']
hline = '-'*80

usage = 'usage: %prog <splitpath>'
parser = optparse.OptionParser(usage)
parser.add_option('-f','--force', dest='force', action='store_true', default=False)

(opt, args) = parser.parse_args()

if len(args) != 1:
    parser.error('Wrong number of arguments')

splitPath = args[0]
# Target directory must exist
if not exists(splitPath):
    parser.error('Directory \''+splitPath+'\' does not exits')

# Split sample description file must exist
ssDescPath = join(splitPath,splitDescriptionFile)
if not exists(ssDescPath):
    parser.error('Split sample description \''+ssDescPath+'\' does not exits')

# Load the description
ssDesc = pickle.load(open(ssDescPath,'r'))

# Loop over the result directories
for resDir in rootResDirs:

    if not exists(resDir):
        os.makedirs(resDir)

    x = {}
    allResFiles = set()
    # Loop over components
    for ss in ssDesc['components']:
        # List files in the result directory
        z = glob.glob(join(splitPath,ss,resDir,'*.root'))

        x[ss] = z
        allResFiles |= set(map(basename, z))

    print allResFiles
    anyMissing = False;

    # Check for missing files
    for ss,files in x.iteritems():
        basefiles = set(map(basename, files))
        if allResFiles != basefiles:
            print 'ERROR: missing file(s) in sample',ss,': ',' '.join(allResFiles-basefiles)
            anyMissing = True

    if anyMissing and not opt.force:
        sys.exit(-1)

    for f in allResFiles:
        merged = join(resDir,f)

        cmd = 'hadd -f '+merged+' '+' '.join([ join(splitPath,ss,resDir,f) for ss in ssDesc['components']])

        print hline
        print 'Merge Command:'
        print cmd
        print
        subprocess.call(cmd,shell=True)
