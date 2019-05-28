#!/bin/env python
import os
import shutil
import argparse 

import subprocess
import sys
import glob
import math
import pickle


print 'Python version', sys.version_info
if sys.version_info < (2, 7):
    raise "Must use python 2.7 or greater. Have you forgotten to do cmsenv?"

path = os.path.abspath(os.path.dirname(__file__))

hline = '-'*80

def debug(text):
    print(">>> DEBUG: ")
    print(text)
    print(">>> >>>>>")

parser = argparse.ArgumentParser()
parser.add_argument('-g','--gdb', dest='gdb', action='store_true', default=False)
parser.add_argument('-i', '--input', dest='inputdir', action='store', required=True,  help='input directory for rootfiles')
parser.add_argument('-o', '--output', dest='outputdir', action='store', required=True, help='output directory (relative to this python file)')
parser.add_argument('-n', '--name', dest='name', action='store', default='sample', help='sample save name')
parser.add_argument('-f', '--filter', dest='filter', action='store', default='*', help='glob-style filter for root files in inputfile')
parser.add_argument('-d', '--debug', dest='debug', action='store_true', default=False)
parser.add_argument('-t', '--timing', dest='timing', action='store_true', default=False)
parser.add_argument('-c', '--save-cuts', dest='cuts', action='store_true', default=False)
parser.add_argument('-b', '--build', dest='build', action='store_true', default=False, help='rebuild cpp files before running')
parser.add_argument('--dry', dest='dryrun', action='store_true', default=False, help="don't run analysis")

args = parser.parse_args()

if args.build:
    os.system("cd {0}; cd ../..; scram b -j 10; cd {1}".format(path, path))

inputdir = os.path.join(path, args.inputdir)
outputdir = os.path.join(path, args.outputdir)

filefilter = str(args.filter)
if not filefilter.endswith(".root"):
    filefilter += ".root"

# get list of samples, write to text file
samplenames = glob.glob(os.path.join(inputdir, filefilter))
samplefile = os.path.join(inputdir, "samples.txt")
with open(samplefile, "w+") as sf:
    for samplename in samplenames:
        sf.write(samplename)

if not os.path.exists(outputdir):
    os.makedirs(outputdir)

if len(samplenames) == 0:
    raise Exception('Not enough samples!')

cmd = 'SVJAnalysis ' + ' '.join([samplefile, args.name, outputdir, '1' if args.debug else '0', '1' if args.timing else '0', '1' if args.cuts else '0'])

print hline

if args.gdb:
    cmd = 'gdb --args ' + cmd

print "Running command: " + cmd

if not args.dryrun:
    subprocess.call(cmd,shell=True)
else:
    print "(Dryrun: finished)"