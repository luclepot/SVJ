#!/bin/env python
import os
import shutil
import argparse 

import subprocess
import sys
import glob
import math
import pickle


LOG_PREFIX = "SVJAnalysis :: "

def log(msg=""):
    print(LOG_PREFIX + str(msg))

def error(msg, code=1):
    log(">>>> ERROR")
    log(msg)
    exit(int(code))

hline = '-'*80
log(hline)
log("PYTHON")

# def range_input(s):
#     try:
#         return tuple(map(int, s.split(',')))
#     except:
#         raise argpparse.ArgumentTypeError("argument '{}' is not of form int,int !".format(s))

if sys.version_info < (2, 7):
    # os.system('cmsenv')
    # os.execl(sys.executable, 'python', os.path.basename(__file__))
    # exit(0)
    error("Must use python 2.7 or greater. Have you forgotten to do cmsenv?")

path = os.path.abspath(os.path.dirname(__file__))


def smartpath(s):
    if s.startswith('~'):
        return s
    return os.path.abspath(s)

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', dest='inputdir', action='store', required=True,  help='input directory for rootfiles')
parser.add_argument('-o', '--output', dest='outputdir', action='store', required=True, help='output directory (relative to this python file)')
parser.add_argument('-n', '--name', dest='name', action='store', default='sample', help='sample save name')
parser.add_argument('-f', '--filter', dest='filter', action='store', default='*', help='glob-style filter for root files in inputfile')
# parser.add_argument('-r', '--range', dest='range', action='store', default=(-1,-1), type=range_input, help='subset of tree values to parse')
parser.add_argument('-s', '--subset', dest='subset', action='store', default=-1, type=int, help='n events to run, out of total')
parser.add_argument('-d', '--debug', dest='debug', action='store_true', default=False, help='enable debug output')
parser.add_argument('-t', '--timing', dest='timing', action='store_true', default=False, help='enable timing output')
parser.add_argument('-c', '--save-cuts', dest='cuts', action='store_true', default=False, help='save cut values')
parser.add_argument('-b', '--build', dest='build', action='store_true', default=False, help='rebuild cpp files before running')
parser.add_argument('-z', '--dry',  dest='dryrun', action='store_true', default=False, help='don\'t run analysis code')
parser.add_argument('-g', '--gdb', dest='gdb', action='store_true', default=False, help='run with gdb debugger :-)')

print sys.argv

argscmd = None
# if len(sys.argv) > 1 and sys.argv[1].lower() == "test":
#     argscmd = "-o outtest -i intest -f qcd* -t -d -c -n test".split()
#     log("running default test")
#     if len(sys.argv) > 2:
#         argscmd = argscmd + ["-b"]
#     if len(sys.argv) > 3:
#         argscmd = argscmd + ["-s", sys.argv[3]]

args = parser.parse_args(argscmd)

if args.build:
    log("Attemping to build rootfile...")
    log("Build Output:")
    if os.system("cd {0}; cd ../..; scram b -j 10; cd {1}".format(path, path)):
        log()
        error("Build failed! Aborting")

inputdir = smartpath(args.inputdir) 
outputdir = smartpath(args.outputdir)

filefilter = str(args.filter)
if not filefilter.endswith(".root"):
    filefilter += ".root"

# get list of samples, write to text file
criteria = os.path.join(inputdir, filefilter)
samplenames = glob.glob(criteria)
samplefile = os.path.join(outputdir, "{}_filelist.txt".format(args.name))

if len(samplenames) == 0:
    error("No samples found matching glob crieria '{}'".format(criteria))

if not os.path.exists(outputdir):
    log("making directory '{}'".format(outputdir))
    os.makedirs(outputdir)

with open(samplefile, "w+") as sf:
    for samplename in samplenames:
        sf.write(samplename + '\n')

cmd = 'SVJAnalysis ' + ' '.join([samplefile, args.name, outputdir, '1' if args.debug else '0', '1' if args.timing else '0', '1' if args.cuts else '0', str(int(args.subset))])


if args.gdb:
    cmd = 'gdb --args ' + cmd

log("Running command: " + cmd)

if args.dryrun:
    log("(Dryrun: finished)")
    log()
    log(hline)
else:
    log()
    log(hline)
    subprocess.call(cmd,shell=True)
