import ROOT as rt
import sys

LOG_PREFIX = "Signal Selection :: "
ERROR_PREFIX = LOG_PREFIX + "ERROR: "

def log(s):
    logbase(s, LOG_PREFIX)

def error(s):
    logbase(s, ERROR_PREFIX)

def logbase(s, prefix):
    if not isinstance(s, str):
        s = str(s)
    for line in s.split('\n'):
        print prefix + str(line)


if __name__=="__main__":
    args = sys.argv
    (_, dummy_filepath, input_filepath, name, rmin, rmax) = args
    
    with open(input_filepath) as fread:
        files = map(lambda x: x.strip('\n'), fread.readlines())
    
    with open(dummy_filepath, 'w+') as fwrite:
        log("Found {0} files for selection; beginning now!!")
        log("----------------------------------------------")
        for f in files:
            tf = rt.TFile(f)
            if tf.GetListOfKeys().Contains("Delphes"):
                from tqdm import tqdm 

                log("selecting in file {0}".format(f))

                tree = tf.Get("Delphes")
                n_events = int(tree.GetEntries())

                to_write = []

                for i in tqdm(range(n_events)):
                    tree.GetEntry(i)

                    # enforce dijet, and write to file
                    if tree.Jet_size > 1:
                        to_write.append(i)

                if len(to_write) > 0:
                    fwrite.write(f)
                    fwrite.write(': ')
                    fwrite.write(' '.join(map(str, to_write)))
                    fwrite.write('\n')
            else:
                error("selection on file {0} failed; no delphes tree in file.".format(f))


                