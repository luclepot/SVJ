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

localPath = os.path.abspath(os.path.dirname(__file__))

t3Path = '/pnfs/psi.ch/cms/trivcat/store/user/grauco/SVJ/v16/Skims_METFilters/JERup/2017'

t3Ls = 'xrdfs t3dcachedb03.psi.ch ls -u'

samples = []
samples.append("SVJ_mZprime2600_mDark20_rinv03_alphapeak")

def sample_comment():
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

    samples.append("TTJets")
    samples.append("TTJets_DiLept")
    samples.append("TTJets_DiLept_genMET-150")
    samples.append("TTJets_HT1200to2500")
    samples.append("TTJets_HT2500toInf")
    samples.append("TTJets_HT600to800")
    samples.append("TTJets_HT800to1200")
    samples.append("TTJets_SingleLeptFromT")
    samples.append("TTJets_SingleLeptFromT_genMET-150")
    samples.append("TTJets_SingleLeptFromTbar")
    samples.append("TTJets_SingleLeptFromTbar_genMET-150")
    samples.append("WJetsToLNu_HT100to200")
    samples.append("WJetsToLNu_HT1200to2500")
    samples.append("WJetsToLNu_HT200to400")
    samples.append("WJetsToLNu_HT2500toInf")
    samples.append("WJetsToLNu_HT400to600")
    samples.append("WJetsToLNu_HT600to800")
    samples.append("WJetsToLNu_HT800to1200")
    samples.append("ZJetsToNuNu_HT100to200")
    samples.append("ZJetsToNuNu_HT1200to2500")
    samples.append("ZJetsToNuNu_HT200to400")
    samples.append("ZJetsToNuNu_HT2500toInf")
    samples.append("ZJetsToNuNu_HT400to600")
    samples.append("ZJetsToNuNu_HT600to800")
    samples.append("ZJetsToNuNu_HT800to1200")
    '''
    '''

    samples.append("Run2016B")
    samples.append("Run2016C")
    samples.append("Run2016D")
    samples.append("Run2016E")
    samples.append("Run2016F")
    samples.append("Run2016G")
    samples.append("Run2016H")

    '''
    '''
    samples.append("Run2017B")
    samples.append("Run2017C")
    samples.append("Run2017D")
    samples.append("Run2017E")
    samples.append("Run2017F")
    '''
    '''
    samples.append("SVJ_mZprime3000_mDark1_rinv03_alphapeak")
    samples.append("SVJ_mZprime3000_mDark5_rinv03_alphapeak")
    samples.append("SVJ_mZprime3000_mDark10_rinv03_alphapeak")
    samples.append("SVJ_mZprime3000_mDark30_rinv03_alphapeak")
    samples.append("SVJ_mZprime3000_mDark40_rinv03_alphapeak")
    samples.append("SVJ_mZprime3000_mDark50_rinv03_alphapeak")
    samples.append("SVJ_mZprime3000_mDark60_rinv03_alphapeak")
    samples.append("SVJ_mZprime3000_mDark70_rinv03_alphapeak")
    samples.append("SVJ_mZprime3000_mDark80_rinv03_alphapeak")
    samples.append("SVJ_mZprime3000_mDark90_rinv03_alphapeak")
    #samples.append("SVJ_mZprime3000_mDark100_rinv03_alphapeak")

    samples.append("SVJ_mZprime3000_mDark20_rinv0_alphapeak")
    samples.append("SVJ_mZprime3000_mDark20_rinv01_alphapeak")
    samples.append("SVJ_mZprime3000_mDark20_rinv02_alphapeak")
    samples.append("SVJ_mZprime3000_mDark20_rinv04_alphapeak")
    samples.append("SVJ_mZprime3000_mDark20_rinv05_alphapeak")
    samples.append("SVJ_mZprime3000_mDark20_rinv06_alphapeak")
    samples.append("SVJ_mZprime3000_mDark20_rinv07_alphapeak")
    samples.append("SVJ_mZprime3000_mDark20_rinv08_alphapeak")
    samples.append("SVJ_mZprime3000_mDark20_rinv09_alphapeak")
    samples.append("SVJ_mZprime3000_mDark20_rinv1_alphapeak")

    samples.append("SVJ_mZprime3000_mDark20_rinv03_alphalow")
    samples.append("SVJ_mZprime3000_mDark20_rinv03_alpha02")
    samples.append("SVJ_mZprime3000_mDark20_rinv03_alphahigh")

    samples.append("SVJ_mZprime500_mDark20_rinv03_alphapeak")
    samples.append("SVJ_mZprime600_mDark20_rinv03_alphapeak")
    samples.append("SVJ_mZprime700_mDark20_rinv03_alphapeak")
    samples.append("SVJ_mZprime800_mDark20_rinv03_alphapeak")
    samples.append("SVJ_mZprime900_mDark20_rinv03_alphapeak")
    samples.append("SVJ_mZprime1000_mDark20_rinv03_alphapeak")
    samples.append("SVJ_mZprime1100_mDark20_rinv03_alphapeak")
    samples.append("SVJ_mZprime1200_mDark20_rinv03_alphapeak")
    samples.append("SVJ_mZprime1300_mDark20_rinv03_alphapeak")
    samples.append("SVJ_mZprime1400_mDark20_rinv03_alphapeak")
    samples.append("SVJ_mZprime1500_mDark20_rinv03_alphapeak")
    samples.append("SVJ_mZprime1600_mDark20_rinv03_alphapeak")
    samples.append("SVJ_mZprime1700_mDark20_rinv03_alphapeak")
    samples.append("SVJ_mZprime1800_mDark20_rinv03_alphapeak")
    samples.append("SVJ_mZprime1900_mDark20_rinv03_alphapeak")
    samples.append("SVJ_mZprime2000_mDark20_rinv03_alphapeak")
    samples.append("SVJ_mZprime2100_mDark20_rinv03_alphapeak")
    samples.append("SVJ_mZprime2200_mDark20_rinv03_alphapeak")
    samples.append("SVJ_mZprime2300_mDark20_rinv03_alphapeak")
    samples.append("SVJ_mZprime2400_mDark20_rinv03_alphapeak")
    samples.append("SVJ_mZprime2500_mDark20_rinv03_alphapeak")
    samples.append("SVJ_mZprime2600_mDark20_rinv03_alphapeak")
    samples.append("SVJ_mZprime2700_mDark20_rinv03_alphapeak")
    samples.append("SVJ_mZprime2800_mDark20_rinv03_alphapeak")
    samples.append("SVJ_mZprime2900_mDark20_rinv03_alphapeak")
    samples.append("SVJ_mZprime3000_mDark20_rinv03_alphapeak")
    samples.append("SVJ_mZprime3100_mDark20_rinv03_alphapeak")

    samples.append("SVJ_mZprime3200_mDark20_rinv03_alphapeak")
    samples.append("SVJ_mZprime3300_mDark20_rinv03_alphapeak")
    samples.append("SVJ_mZprime3400_mDark20_rinv03_alphapeak")
    samples.append("SVJ_mZprime3500_mDark20_rinv03_alphapeak")
    samples.append("SVJ_mZprime3600_mDark20_rinv03_alphapeak")
    samples.append("SVJ_mZprime3700_mDark20_rinv03_alphapeak")
    samples.append("SVJ_mZprime3800_mDark20_rinv03_alphapeak")
    samples.append("SVJ_mZprime3900_mDark20_rinv03_alphapeak")
    samples.append("SVJ_mZprime4000_mDark20_rinv03_alphapeak")
    samples.append("SVJ_mZprime4100_mDark20_rinv03_alphapeak")
    samples.append("SVJ_mZprime4200_mDark20_rinv03_alphapeak")
    samples.append("SVJ_mZprime4300_mDark20_rinv03_alphapeak")
    samples.append("SVJ_mZprime4400_mDark20_rinv03_alphapeak")
    samples.append("SVJ_mZprime4500_mDark20_rinv03_alphapeak")
    '''
    return 0

def debug(text):
    print(">>> DEBUG: ")
    print(text)
    print(">>> >>>>>")

splitMap = {}

#splitMap["Run2016B"] = 20

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
        cmd = 'gdb --args ' + cmd
    elif opt.t3batch:
        jid = '%s_%s' % (sample,opt.sys)
        cmd = 'qexe.py -w '+workDir+' '+jid+' -- ' + cmd
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
parser.add_option('-y', '--year', dest='year', type='string',  help='Year')
parser.add_option('-g','--gdb', dest='gdb', action='store_true', default=False)
parser.add_option('-n','--dryrun', dest='dryrun', action='store_true', default=False)
parser.add_option('-m','--mode', dest='mode', default='t3se', choices=['local','t3se'])
parser.add_option('--t3batch', dest='t3batch', action='store_true', default=False)
parser.add_option('-t', '--tree-name', dest='tree_name', action='store', type='string', help='name of tree to run', default='tree')
parser.add_option('-f', '--filter', dest='filter', action='store', type='string', help='filter for root files', default='*')

isData="MC"
#isData="DATA"

(opt, args) = parser.parse_args()

if opt.sys not in systematics:
    parser.error('Please choose an allowed value for sys: "noSys", "jesUp", "jesDown", "jerUp", "jerDown", "jmsUp", "jmsDown", "jmrUp", "jmrDown"')

# Create working area if it doesn't exist

if not exists(fileListDir):
    os.makedirs(fileListDir)
    

for s in samples:
    if (s.startswith("JetHT") or s.startswith("SingleMu") or s.startswith("SingleEl") or  s.startswith("MET")):
        isData="DATA"
    
    print s
    print str(opt.sync)
    
    ## cmd = "ttDManalysis "+ s + " " + path + s  + " " + opt.channel + " " + opt.sys + " " + opt.sync + " " + isData
    ## print cmd
    ## os.system(cmd)

    if opt.mode == 'local':

        sPath = join(localPath,s,opt.filter)
        
        allFiles = glob.glob(sPath)
        # print ' '.join([lLs,sPath])
        # Get the complete list of files
        # listing = subprocess.check_output(lLs.split()+[sPath])

        print 'Sample',s,'Files found',len(allFiles)

    elif opt.mode == 't3se':
        # Build the full path of sample files
        sT3Path = join(t3Path,s)
        print ' '.join([t3Ls,sT3Path])

        # Get the complete list of files
        listing = subprocess.check_output(t3Ls.split()+[sT3Path])
        allFiles = listing.split()
        print 'Sample',s,'Files found',len(allFiles)

    if len(allFiles) == 0:
        raise Exception('Not enough samples!')

    if not (s in splitMap and opt.t3batch):

        if not opt.t3batch:
            print 'WARNING: Batch mode not active: Sample',s,'will not be split even if it appears in the splitMap'

        # Save it to a semi-temp file
        sampleFileList = join(fileListDir,s+'_'+opt.sys+'.txt')
        print 'DEBUG: writing'
        print 'File list:',sampleFileList
        with open(sampleFileList,'w') as sl:
            sl.write('\n'.join(allFiles))


        cwd = os.getcwd()

        makedirs(resDirs,cwd)
        cmd = 'SVJAnalysis '+ s + ' ' + sampleFileList  + ' ' + opt.sys + ' ' + opt.sync + ' ' + isData + ' ' +  cwd + ' ' + opt.year + ' ' + opt.tree_name
        
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

            cmd = 'SVJAnalysis '+ s + ' ' + sampleFileList  + ' ' + opt.sys + ' ' + opt.sync + ' ' + isData + ' ' +  wd + ' ' + opt.tree_name

            run(ss,cmd,opt)
