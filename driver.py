import os
import sys
from argparse import ArgumentParser, ArgumentError
from glob import glob

BASE_COMMAND = 'env -i HOME=$HOME bash -i -c "<CMD>"'
LOG_PREFIX = "Driver :: "
ERROR_PREFIX = LOG_PREFIX + "ERROR: "

### helper functions

def log(s):
    logbase(s, LOG_PREFIX)

def error(s):
    logbase(s, ERROR_PREFIX)

def logbase(s, prefix):
    if isinstance(s, str):
        for line in s.split('\n'):
            print prefix + str(line)
    else:
        print prefix + str(line)

def _range_input(s):
    try:
        return tuple(map(int, s.strip().strip(')').strip('(').split(',')))
    except:
        raise ArgumentError("-r input not in format: int,int")

def _smartpath(s):
    if s.startswith('~'):
        return s
    return os.path.abspath(s)

def _check_for_default_file(pathoptions, name, pattern, suffix="txt"):
    fname = "{0}_{1}.{2}".format(name, pattern, suffix)
    
    if isinstance(pathoptions, str):
        pathoptions = [pathoptions]
    
    for path in pathoptions:
        spath = os.path.join(path, fname)
        if os.path.exists(spath):
            return spath

    error("file wih pattern '{0}' does not exist in any of: \n\t{1}".format(fname, "\n\t".join(pathoptions)))
    error("quiting")
    sys.exit(1)

def condor_submit(
    cmd,
    outputdir,
    name,
    setup_cmd = None,
):
    log("writing queue commands")
    run_script = os.path.join(outputdir, "{0}_submit.sh".format(name))
    run_submit = os.path.join(outputdir, "{0}_condor.submit".format(name))
    with open(run_script, 'w+') as f:
        f.write("#!/bin/bash\n")
        f.write("echo 'RUNNING NOW'\n")
        f.write("\n")
        # f.write(cmd + "\n")
        for line in cmd.split("; "):
            f.write("{}\n".format(line))
    with open(run_submit, 'w+') as f:
        f.write("executable = {}\n\n".format(run_script))
        f.write("universe = vanilla\n")
        f.write("getenv = True\n")
        f.write("log = {0}\n".format(os.path.join(outputdir, "{0}.log".format(name))))
        f.write("output = {0}\n".format(os.path.join(outputdir, "{0}.out".format(name))))
        f.write("error = {0}\n".format(os.path.join(outputdir, "{0}.err".format(name))))
        f.write("should_transfer_files = YES\n")
        f.write("when_to_transfer_output = ON_EXIT_OR_EVICT\n")
        f.write("transfer_input_files = {0}\n".format(run_script))
        # f.write("transfer_output_files = Data\n")
        f.write("request_cpus = 4\n")
        # f.write("request_disk = 20MB\n")
        # f.write("request_memory = 5MB\n\n")
        f.write("+JobFlavour = \"workday\"\n")
        f.write("queue\n")
    os.system("chmod +rwx {}".format(run_script))
    condor_cmd = "condor_submit {}; condor_q".format(run_submit)
    if setup_cmd is not None:
        condor_cmd = "{0}; ".format(setup_cmd) + condor_cmd
    os.system(condor_cmd)
    sys.exit(0)

def local_submit(
    master_command,
):
    os.system(master_command)
    sys.exit(0)

### parser setup

def setup_parser():
    parser = ArgumentParser()

    # add subparsers
    subparsers = parser.add_subparsers(dest='COMMAND')
    select = subparsers.add_parser("select", help="base selection of events")
    convert = subparsers.add_parser("convert", help="convert raw root data to h5 files based on selection output")
    train = subparsers.add_parser("train", help="training command for module")
    
    # general criteria
    for subparser in [select, convert, train]:
        subparser.add_argument('-i', '--input', dest="inputdir", action="store", type=_smartpath, help="input dir path", required=True)
        subparser.add_argument('-o', '--output', dest="outputdir", action="store", type=_smartpath, help="output dir path", required=True)
        subparser.add_argument('-n', '--name', dest='name', action='store', default='sample', help='sample save name')
        subparser.add_argument('-j', '--batch-job', dest='batch', action='store', default=None, help='attempt to run as a batch job on the indicated service')
        subparser.add_argument('-z', '--dry',  dest='dryrun', action='store_true', default=False, help='don\'t run analysis code')
    
    # selection args
    select.add_argument('-f', '--filter', dest='filter', action='store', default='*', help='glob-style filter for root files in inputfile')
    select.add_argument('-r', '--range', dest='range', action='store', default=(-1,-1), type=_range_input, help='subset of tree values to parse')
    select.add_argument('-d', '--debug', dest='debug', action='store_true', default=False, help='enable debug output')
    select.add_argument('-t', '--timing', dest='timing', action='store_true', default=False, help='enable timing output')
    select.add_argument('-c', '--save-cuts', dest='cuts', action='store_true', default=False, help='save cut values')
    select.add_argument('-b', '--build', dest='build', action='store_true', default=False, help='rebuild cpp files before running')
    select.add_argument('-g', '--gdb', dest='gdb', action='store_true', default=False, help='run with gdb debugger :-)')
    # conversion args
    convert.add_argument('-d', '--dr', dest='DR', action='store', type=float, default=0.8, help='dr parameter for jet finding')
    convert.add_argument('-c', '--constituents', dest='NC', action='store', type=int, default=100, help='number of jet constituents to save')
    convert.add_argument('-r', '--range', dest='range', action='store', type=_range_input, default=(-1,-1), help='range of data to parse')
    # training arg

    return parser

def select_main(inputdir, outputdir, name, batch, filter, range, debug, timing, cuts, build, dryrun, gdb):
    log("running command 'select'")
    
    inputdir = _smartpath(inputdir)
    outputdir = _smartpath(outputdir)
    
    ffilter = str(filter)
    rng = range

    if not ffilter.endswith(".root"):
        ffilter += ".root"

    # get list of samples, write to text file
    criteria = os.path.join(inputdir, ffilter)
    samplenames = glob(criteria)
    samplefile = os.path.join(outputdir, "{0}_filelist.txt".format(name))

    if len(samplenames) == 0:
        error("No samples found matching glob crieria '{0}'".format(criteria))
        sys.exit(1)

    if not os.path.exists(outputdir):
        log("making ouput directory '{0}'".format(outputdir))
        os.makedirs(outputdir)

    with open(samplefile, "w+") as sf:
        log("writing samplefile to file '{0}'".format(samplefile))
        for samplename in samplenames:
            sf.write(samplename + '\n')

    path = os.path.abspath(os.path.dirname(__file__))
    setup_command = "source {0}".format(os.path.join(path, "selection/setup.sh"))

    if build:
        setup_command += "; cd {0}; cd ../..; scram b -j 10; cd {1}".format(path, path)
        
    run_command = 'cd {0}; ../../bin/sl*/SVJselection '.format(path) + ' '.join([samplefile, name, outputdir] + list(map(lambda x: str(int(x)), [debug, timing, cuts, rng[0], rng[1]])))
    master_command = setup_command + "; " + run_command

    if dryrun:
        log("DRYRUN: command is:")
        log(master_command)
        sys.exit(0)
    
    if batch is not None:
        if batch == "condor":
            condor_setup = "; ".join([
                "source /cvmfs/cms.cern.ch/cmsset_default.sh",
                "export SCRAM_ARCH=slc7_amd64_gcc530",
                "eval `scramv1 runtime -sh`; ",
            ])
            condor_submit(condor_setup + master_command, outputdir, name, setup_command)
        else:
            raise ArgumentError("unrecognized batch platform '{0}'".format(batch))

    master_command = BASE_COMMAND.replace("<CMD>", master_command)

    local_submit(master_command)

def convert_main(inputdir, outputdir, name, batch, range, DR, NC, dryrun):
    log("running command 'convert'")
    filespec = _check_for_default_file([inputdir, outputdir], name, 'filelist')
    spath = _check_for_default_file([inputdir, outputdir], name, 'selection')
    setup_command = "source " + os.path.abspath("conversion/setup.sh")
    python_command = "python " + os.path.abspath("conversion/h5converter.py")
    python_command += " " + " ".join(map(str, [inputdir, outputdir, filespec, spath, name, DR, NC, range[0], range[1]]))

    master_command = BASE_COMMAND.replace("<CMD>", "; ".join([setup_command, python_command]))

    if dryrun:
        log("DRYRUN: command is:")
        log(master_command)
        sys.exit(0)


    if batch is not None:
        if batch == "condor":
            condor_setup = ""
            condor_submit(condor_setup + master_command, outputdir, name, setup_command)
        else:
            raise ArgumentError("unrecognized batch platform '{0}'".format(batch))
        
    os.system(master_command)
    sys.exit(0)

def train_main(inputdir, outputdir, name, batch, filter):
    log("running command 'train'")
    sys.exit(0)

if __name__=="__main__":
    log("setting up driver")
    parser = setup_parser()
    argv = sys.argv[1:]

    if len(argv) > 0:
        args = parser.parse_args(argv)
    else:
        parser.print_help()
        sys.exit(0)

    args_dict = vars(args)
    cmd = args_dict["COMMAND"]
    del args_dict["COMMAND"]
    # args_dict = { var: vars(args)[var] for var in vars(args) if "COMMAND" not in var }

    if cmd == "select":
        select_main(**args_dict)
    if cmd == "convert":
        convert_main(**args_dict)
    if cmd == "train":
        train_main(**args_dict) 