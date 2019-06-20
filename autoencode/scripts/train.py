# import autoencodeSVJ.models as models
# import autoencodeSVJ.utils as utils
# import autoencodeSVJ.trainer as trainer

def _get_args():
    import argparse

    def figsize(s):
        try:
            slist = list(map(int, s.split(",")))
            assert len(slist) == 2
            return tuple(slist)
        except:
            raise argparse.ArgumentTypeError("input '{}' is not of the required format <int>,<int>".format(s))

    import sys
    parser = argparse.ArgumentParser("python plot.py")

    parser.add_argument("-n", "--name", help="name for the training instance", required=True)
    sub = parser.add_subparsers(dest="PLOT_COMMAND")

    reps = sub.add_parser("reps", help="plot reps for the model specified")
    data = sub.add_parser("data", help="plot input/output data for model specified")
    loss = sub.add_parser("loss", help="plot loss statistics for model specified")
    
    reps.add_argument("-d", "--data", help="path to data to parse", required=True)
    data.add_argument("criteria", help="glob criteria for data subsets to plot")
    reps.add_argument("-s", "--figsize", default=(7,5), type=figsize)
    reps.add_argument("-y", "--yscale", default="linear")
    reps.add_argument("-x", "--xscale", default="linear")
    reps.add_argument("-c", "--columns", default=4, type=int)
    # reps.add_argument("")
    
    data.add_argument("-d", "--data", help="path to data to parse", required=True)
    data.add_argument("criteria", help="glob criteria for data subsets to plot")
    data.add_argument("-s", "--figsize", default=(7,5), type=figsize)
    data.add_argument("-y", "--yscale", default="linear")
    data.add_argument("-x", "--xscale", default="linear")
    data.add_argument("-c", "--columns", default=4, type=int)

    loss.add_argument("criteria", help="glob criteria for losses to plot")
    loss.add_argument("-y", "--yscale", default="linear")
    loss.add_argument("-x", "--xscale", default="linear")
    loss.add_argument("-s", "--figsize", default=(7,5), type=figsize)
    loss.add_argument("-c", "--columns", default=4, type=int)

    return parser.parse_args(sys.argv[1:])

def make_parser():
    import argparse

    def figsize(s):
        try:
            slist = list(map(int, s.split(",")))
            assert len(slist) == 2
            return tuple(slist)
        except:
            raise argparse.ArgumentTypeError("input '{}' is not of the required format <int>,<int>".format(s))

    def node_arch(s):
        try:
            slist = list(map(int, s.split(",")))
            assert len(slist) > 2
            return tuple(slist)
        except:
            raise argparse.ArgumentTypeError("input '{}' is not of the required format <int>,...,<int>,...,<int>".format(s))

    import sys

    p = argparse.ArgumentParser("python train.py")

    p.add_argument("NAME")
    subp = p.add_subparsers("COMMAND")

    make = subp.add_parser("make")
    train = subp.add_parser("train")
    plot = subp.add_parser("plot")
    

    make.add_argument("-a", "--arch", type=node_arch, required=True, help="node architecture for autoencoder")
    make.add_argument("")

    train.add_argument("-e", "--epochs", type=int)

    plot_types = plot.add_subparsers()

if __name__ == "__main__":
    args = get_args()
    print args

