#!/bin/env python
import os
import shutil
import optparse 

import subprocess
import sys
import glob
import math
import pickle

from os.path import join,exists

print 'Python version', sys.version_info
if sys.version_info < (2, 7):
    raise "Must use python 2.7 or greater. Have you forgotten to do cmsenv?"

workDir = 'work'
fileListDir = join(workDir,'files')
splitDir = 'split'
splitDescriptionFile = 'description.pkl'
resDirs = ['res','trees','txt']
hline = '-'*80

#localPath = "/t3home/decosa/SVJ/CMSSW_8_0_20/src/SVJ/SVJAnalysis/"
localPath = "/mnt/t3nfs01/data01/shome/grauco/SVJAnalysis/CMSSW_8_0_20/src/SVJ/SVJAnalysis/"
#localPath = "/mnt/t3nfs01/data01/shome/grauco/SVJAnalysis/CMSSW_8_0_20/src/SVJ/SVJAnalysis/"
#t3Path = '//pnfs/psi.ch/cms/trivcat/store/user/grauco/SVJ/v0/'
t3Path = '/pnfs/psi.ch/cms/trivcat/store/user/grauco/SVJ/'
t3Ls = 'xrdfs t3dcachedb03.psi.ch ls -u'

samples = []

#samples.append("SVJ_mZprime-1000_mDark-20_rinv-0.3_alpha-0.2")
'''
samples.append("QCD_Pt_1000to1400")
samples.append("QCD_Pt_120to170")
samples.append("QCD_Pt_170to300")   
samples.append("QCD_Pt_1400to1800")
samples.append("QCD_Pt_1800to2400")
samples.append("QCD_Pt_2400to3200")
samples.append("QCD_Pt_300to470")
samples.append("QCD_Pt_3200toInf")
samples.append("QCD_Pt_470to600")
samples.append("QCD_Pt_600to800")
samples.append("QCD_Pt_800to1000")
samples.append("QCD_Pt_80to120")
'''
'''
samples.append("TTJets")

samples.append("WJetsToLNu_HT100to200")
samples.append("WJetsToLNu_HT1200to2500")
samples.append("WJetsToLNu_HT200to400")
samples.append("WJetsToLNu_HT2500toInf")
samples.append("WJetsToLNu_HT400to600")
samples.append("WJetsToLNu_HT600to800")
samples.append("WJetsToLNu_HT800to1200")
samples.append("ZJetsToNuNu_Zpt100to200")
samples.append("ZJetsToNuNu_Zpt200toInf")
'''

samples.append("SVJ_mZprime1000_mDark20_rinv03_alpha02")
samples.append("SVJ_mZprime2000_mDark20_rinv03_alpha02")
samples.append("SVJ_mZprime3000_mDark100_rinv03_alpha02")
samples.append("SVJ_mZprime3000_mDark1_rinv03_alpha02")
samples.append("SVJ_mZprime3000_mDark20_rinv01_alpha02")
samples.append("SVJ_mZprime3000_mDark20_rinv03_alpha01")
samples.append("SVJ_mZprime3000_mDark20_rinv03_alpha02")
samples.append("SVJ_mZprime3000_mDark20_rinv03_alpha05")
samples.append("SVJ_mZprime3000_mDark20_rinv03_alpha1")
samples.append("SVJ_mZprime3000_mDark20_rinv05_alpha02")
samples.append("SVJ_mZprime3000_mDark20_rinv07_alpha02")
samples.append("SVJ_mZprime3000_mDark50_rinv03_alpha02")
samples.append("SVJ_mZprime4000_mDark20_rinv03_alpha02")


splitMap = {}
splitMap["TT"] = 30


def makedirs(subdirs,base=None):

    # Calculate the full path
    if base is not None:
        subdirs = [ join(base,sd) for sd in subdirs]

    # Loop over subdirs
    for sd in subdirs:
        # Check for existence
        if exists(sd): continue
        # Make directory and partents
        os.makedirs(sd)

def writeFileList(sample, files, opt):
    # Save it to a semi-temp file
    sampleFileList = join(fileListDir,sample+'_'+opt.sys+'.txt')
    print 'File list:',sampleFileList
    with open(sampleFileList,'w') as sl:
        sl.write('\n'.join(files))

    return sampleFileList

def run(sample, cmd, opt):

    print hline
    if opt.gdb:
        cmd = 'gdb --args '+cmd
    elif opt.t3batch:
        jid = '%s_%s' % (sample,opt.sys)
        cmd = 'qexe.py -w '+workDir+' '+jid+' -- '+cmd
    print cmd

    if opt.dryrun:
        print 'Dry Run (command will not be executed)'
        return

    subprocess.call(cmd,shell=True)

systematics = ["noSys", "jesUp", "jesDown", "jerUp", "jerDown", "jmsUp", "jmsDown", "jmrUp", "jmrDown"]

usage = 'usage: %prog'
parser = optparse.OptionParser(usage)

parser.add_option('-s', '--sys', dest='sys', default = 'noSys', choices=systematics, help='Systematics: '+' '.join(systematics))
parser.add_option('--sync', dest='sync', type='string', default = 'noSync', help='Synchro exercise')
parser.add_option('-g','--gdb', dest='gdb', action='store_true', default=False)
parser.add_option('-n','--dryrun', dest='dryrun', action='store_true', default=False)
parser.add_option('-m','--mode', dest='mode', default='t3se', choices=['local','t3se'])
parser.add_option('--t3batch', dest='t3batch', action='store_true', default=True)

isData="MC"
#isData="DATA"
(opt, args) = parser.parse_args()

if opt.sys not in systematics:
    parser.error('Please choose an allowed value for sys: "noSys", "jesUp", "jesDown", "jerUp", "jerDown", "jmsUp", "jmsDown", "jmrUp", "jmrDown"')


# Create working area if it doesn't exist
if not exists(fileListDir):
    os.makedirs(fileListDir)
    

for s in samples:
    if (s.startswith("JetHT") or s.startswith("SingleMu") or s.startswith("SingleEl") or  s.startswith("MET")): isData="DATA"
    print s
    print str(opt.sync)
    ## cmd = "ttDManalysis "+ s + " " + path + s  + " " + opt.channel + " " + opt.sys + " " + opt.sync + " " + isData
    ## print cmd
    ## os.system(cmd)

    if opt.mode == 'local':
        sPath = join(localPath,s,'*.root')
        
        print sPath
        # print ' '.join([lLs,sPath])
        # Get the complete list of files
        # listing = subprocess.check_output(lLs.split()+[sPath])
        allFiles = glob.glob(sPath)
        print 'Sample',s,'Files found',len(allFiles)

    elif opt.mode == 't3se':
        # Build the full path of sample files
        sT3Path = join(t3Path,s)
        print ' '.join([t3Ls,sT3Path])

        # Get the complete list of files
        listing = subprocess.check_output(t3Ls.split()+[sT3Path])
        allFiles = listing.split()
        print 'Sample',s,'Files found',len(allFiles)



    if not (s in splitMap and opt.t3batch):

        if not opt.t3batch:
            print 'WARNING: Batch mode not active: Sample',s,'will not be split even if it appears in the splitMap'

        # Save it to a semi-temp file
        sampleFileList = join(fileListDir,s+'_'+opt.sys+'.txt')
        print 'File list:',sampleFileList
        with open(sampleFileList,'w') as sl:
            sl.write('\n'.join(allFiles))


        cwd = os.getcwd()

        makedirs(resDirs,cwd)
        cmd = 'SVJAnalysis '+ s + ' ' + sampleFileList  + ' ' + opt.sys + ' ' + opt.sync + ' ' + isData + ' ' +  cwd
        
        run(s,cmd,opt)

    else:

        # 
        nsplit = splitMap[s]

        if len(allFiles) < nsplit:
            nsplit = len(allFiles)
            print 'Warning: split level larger than number of files. Splitlevel set to nfiles (',nsplit,')'

        # Pretty format
        ndigits = int(math.floor(math.log10(nsplit))+1)
        nfmt = '%%s_%%s_j%%0%dd' % ndigits

        # Calculate the size of each file block
        chunkSize = len(allFiles)/nsplit
        # The reminder is the number of jobs that get one file more
        nExtra = len(allFiles)%nsplit

        print ' * Split level',nsplit
        print ' * Files per job (base):',chunkSize
        print ' * Jobs with 1 extra file:',nExtra

        resRootDir = join(splitDir,'%s_%s' % (s,opt.sys))

        print ' * Result root folder',resRootDir

        subsamples = {}
        lastFileIdx = 0
        for k in xrange(nsplit):
            print nfmt, s, opt.sys, k
            name = nfmt % (s,opt.sys,k)
            wd = join(resRootDir,name)
            # Start from last idx
            firstFileIdx = lastFileIdx
            # add chunk size. add 1 if k<nExtra
            lastFileIdx = firstFileIdx+chunkSize+(k<nExtra)
            files = allFiles[firstFileIdx:lastFileIdx]

            subsamples[name] = (wd,files)

        # Cleanup previous result areas
        if exists(resRootDir):
            print 'Cleaning directory',resRootDir
            shutil.rmtree(resRootDir)

        # Recreate the target directory
        os.makedirs(resRootDir)

        ssDesc = { 'sample':s, 'components': subsamples.keys() }
        pickle.dump(ssDesc, open(join(resRootDir,splitDescriptionFile),'w') )

        for ss in sorted(subsamples):
            # Fetch output directory and file list
            wd,files = subsamples[ss]
            # Make the subdirectories
            makedirs(resDirs,wd)

            sampleFileList = writeFileList(ss, files,opt)

            cmd = 'SVJAnalysis '+ s + ' ' + sampleFileList  + ' ' + opt.sys + ' ' + opt.sync + ' ' + isData + ' ' +  wd

            run(ss,cmd,opt)
