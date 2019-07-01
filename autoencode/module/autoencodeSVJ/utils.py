import numpy as np 
from collections import OrderedDict as odict
from operator import mul
import h5py
import glob
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as prep
import pandas as pd
import os
import traceback
import matplotlib.pyplot as plt
import glob
import pandas as pd
import prettytable
from StringIO import StringIO
from enum import Enum

plt.rcParams['figure.figsize'] = (10,10)
plt.rcParams.update({'font.size': 18})

class logger:
    """
    good logging parent class.
    Usage:
        When creating a class, subclass this class. 
        You can override the _LOG_PREFIX, _ERROR_PREFIX,
        and VERBOSE member variables either in the derived
        class __init__, or when calling the logger __init__.

        Ex:
        class <classname>(logger):
            def __init__(self, ..., verbose):
                ...
                logger.__init__(self, VERBOSE=verbose, LOG_PREFIX="MY class log prefix")
                ...
            OR
            def __init__(self, ..., verbose):
                logger.__init__(self)
                self.VERBOSE = verbose
                self._LOG_PREFIX = "my log prefix: "

        From there, you can use the logging functions as members of your class.
        i.e. self.log("log message") or self.error("log message")
        
        To get the log as a string, pass the 'string=True' argument to 
        either of the 'log' or 'error' member functions
                
    """


    def __init__(
        self,
        LOG_PREFIX = "logger :: ",
        ERROR_PREFIX = "ERROR: ",
        VERBOSE = True,
    ):
        self._LOG_PREFIX = LOG_PREFIX
        self._ERROR_PREFIX = ERROR_PREFIX
        self.VERBOSE = VERBOSE

    def log(
        self, 
        s,
        string=False,
    ):
        if string:
            if self.VERBOSE:
                return self._log_str(s, self._LOG_PREFIX)
            return ''
        if self.VERBOSE: 
            self._log_base(s, self._LOG_PREFIX)

    def error(
        self,
        s,
        string=False,
    ):
        if string:
            return self._log_str(s, self._LOG_PREFIX + self._ERROR_PREFIX)    
        self._log_base(s, self._LOG_PREFIX + self._ERROR_PREFIX)

    def _log_base(
        self,
        s,
        prefix
    ):
        if isinstance(s, basestring):
            for line in s.split('\n'):
                print prefix + str(line)
        else:
            print prefix + str(s)

    def _log_str(
        self,
        s,
        prefix
    ):
        out = ''
        if isinstance(s, basestring):
            for line in s.split('\n'):
                out += prefix + str(line) + '\n'
        else:
            out += prefix + str(line) + '\n'
        return out

class data_table(logger):
    
    class NORM_TYPES(Enum):
        MinMaxScaler = 0
        StandardScaler = 1
        RobustScaler = 2

    _RDICT_NORM_TYPES = dict(map(lambda x: (x.value, x.name), NORM_TYPES))

    TABLE_COUNT = 0

    """
    wrapper for the pandas data table. 
    allows for quick variable plotting and train/test/splitting.
    """

    def __init__(
        self,
        data,
        headers=None,
        name=None,
        verbose=1,
    ):
        logger.__init__(self, "data_table :: ", verbose)
        self.name = name or "untitled {}".format(data_table.TABLE_COUNT)    
        data_table.TABLE_COUNT += 1
        if headers is not None:
            self.headers = headers
        elif isinstance(data, pd.DataFrame):
            self.headers = data.columns 
        elif isinstance(data, data_table):
            data = data.data
            self.headers = data.headers
        else:
            self.headers = ["dist " + str(i + 1) for i in range(data.shape[1])]
        self.data = data        
        assert len(self.data.shape) == 2, "data must be matrix!"
        assert len(self.headers) == self.data.shape[1], "n columns must be equal to n column headers"
        assert len(self.data) > 0, "n samples must be greater than zero"
        self.df = pd.DataFrame(self.data, columns=self.headers)
        self.scaler = None

    def norm(
        self,
        data=None,
        norm_type=0,
        **scaler_args
    ):
        if isinstance(norm_type, str):
            norm_type = getattr(self.NORM_TYPES, norm_type)
        elif isinstance(norm_type, int):
            norm_type = getattr(self.NORM_TYPES, self._RDICT_NORM_TYPES[norm_type])
        
        assert isinstance(norm_type, self.NORM_TYPES)

        self.scaler = getattr(prep, norm_type.name)(**scaler_args)
        self.scaler.fit(self.df)
        
        if data is None:
            data = self
        
        assert isinstance(data, data_table), "data must be data_table type"

        ret = data_table(self.scaler.transform(data.df), headers = self.headers, name="'{}' normed to '{}'".format(data.name,self.name))
        return ret

    def inorm(
        self,
        data=None, 
        rng=None,
        norm_type=0,
        **scaler_args
    ):

        if rng is None:
            rng = (0,1)

        if isinstance(norm_type, str):
            norm_type = getattr(self.NORM_TYPES, norm_type)
        elif isinstance(norm_type, int):
            norm_type = getattr(self.NORM_TYPES, self._RDICT_NORM_TYPES[norm_type])
        
        assert isinstance(norm_type, self.NORM_TYPES)

        self.scaler = getattr(prep, norm_type.name)(rng, **scaler_args)
        self.scaler.fit(self.df)
        
        if data is None:
            data = self
        
        assert isinstance(data, data_table), "data must be data_table type"

        ret = data_table(self.scaler.inverse_transform(data.df), headers=self.headers, name="'{}' inv_normed to '{}'".format(data.name,self.name))
        return ret
        
    def __getattr__(
        self,
        attr,
    ):
        if hasattr(self.df, attr):
            return self.df.__getattr__(attr)
        else:
            raise AttributeError, "no dataframe or data_table attribute matching '{}'".format(attr)

    def __str__(
        self,
    ):
        return self.df.__str__()

    def __repr__(
        self,
    ):
        return self.df.__repr__()

    def split_by_column_names(
        self,
        column_list_or_criteria,
    ):
        match_list = None
        if isinstance(column_list_or_criteria, str):
            match_list = [c for c in self.headers if glob.fnmatch.fnmatch(c, column_list_or_criteria)]
        else:
            match_list = list(column_list_or_criteria)

        other = [c for c in self.headers if c not in match_list]

        t1,t2 = self.df.drop(other,axis=1),self.df.drop(match_list,axis=1)

        
        return data_table(t1, headers=match_list, name=self.name), data_table(t2, headers=other, name=self.name)
        
    def train_test_split(
        self,
        test_fraction=0.25,
        random_state=None,
        shuffle=True
    ):
        dtrain, dtest = train_test_split(self, test_size=test_fraction, random_state=random_state)
        return (data_table(np.asarray(dtrain), np.asarray(dtrain.columns), "train"),
            data_table(np.asarray(dtest), np.asarray(dtest.columns), "test"))
    
    def plot(
        self,
        others=[],
        values="*",
        bins=32,
        rng=None,
        cols=4,
        ticksize=8,
        fontsize=25,
        normed=0,
        figloc="upper right",
        figsize=(30,15),
        alpha=0.7,
        xscale="linear",
        yscale="linear"
    ):
        if isinstance(values, str):
            values = [key for key in self.headers if glob.fnmatch.fnmatch(key, values)]
        if not hasattr(values, "__iter__"):
            values = [values]
        for i in range(len(values)):
            if isinstance(values[i], int):
                values[i] = self.headers[values[i]]
        
        if not isinstance(others, list) or isinstance(others, tuple):
            others = [others]

        for i in range(len(others)):
            if not isinstance(others[i], data_table):
                others[i] = data_table(others[i], headers=self.headers)

        n = len(values)
        rows = self._rows(cols, n)

        if n < cols:
            cols = n
            rows = 1

        plot_data = [self[v] for v in values]
        plot_others = [[other[v] for v in values] for other in others]

        if rng is None:
            rmax = np.max([d.max().values for d in ([self] + others)], axis=0)
            rmin = np.min([d.min().values for d in ([self] + others)], axis=0)
            rng =  np.array([rmin, rmax]).T
        elif len(rng) == 2 and all([not hasattr(r, "__iter__") for r in rng]):
            rng = [rng for i in range(len(plot_data))]

        weights = None

        self.log("plotting distrubution(s) for table(s) {}".format([self.name,] + [o.name for o in others]))
        plt.rcParams['figure.figsize'] = figsize

        for i in range(n):
            ax = plt.subplot(rows, cols, i + 1)
            # weights = np.ones_like(plot_data[i])/float(len(plot_data[i]))
            ax.hist(plot_data[i], bins=bins, range=rng[i], histtype='step', normed=normed, label=self.name, weights=weights, alpha=alpha)
            for j in range(len(others)):
                # weights = np.ones_like(plot_others[j][i])/float(len(plot_others[j][i]))
                ax.hist(plot_others[j][i], bins=bins, range=rng[i], histtype='step', label=others[j].name, normed=normed, weights=weights, alpha=alpha)

            plt.xlabel(plot_data[i].name + " {}-scaled".format(xscale), fontsize=fontsize)
            plt.ylabel("{}-scaled".format(yscale), fontsize=fontsize)
            plt.xticks(size=ticksize)
            plt.yticks(size=ticksize)
            plt.yscale(yscale)
            plt.xscale(xscale)
            plt.gca().spines['left']._adjust_location()
            plt.gca().spines['bottom']._adjust_location()


        handles,labels = ax.get_legend_handles_labels()
        plt.figlegend(handles, labels, loc=figloc)
        plt.tight_layout(pad=0.01, w_pad=0.01, h_pad=0.01)
        plt.show()
    
    def _rows(
        self,
        cols,
        n
    ):
        return n/cols + bool(n%cols)

    def split_to_jets(
        self,
    ):
        return split_to_jets(self)

class data_loader(logger):
    """
    data loader / handler/merger for h5 files with the general format
    of this repository
    """

    def __init__(
        self,
        name,
        verbose=True
    ):
        logger.__init__(self)
        self.name = name
        self._LOG_PREFIX = "data_loader :: "
        self.VERBOSE = verbose
        self.samples = odict()
        self.sample_keys = None
        self.data = odict()
        self.size = 0
        self.shapes = []
        self.table = None

    def add_sample(
        self,
        sample_path,
    ):
        filepath = smartpath(sample_path)
        assert os.path.exists(filepath)
        if filepath not in self.samples:
            self.log("Adding sample at path '{}'".format(filepath))
            self.samples[filepath] = h5py.File(filepath)
            if self.sample_keys is None:
                self.sample_keys = set(self.samples[filepath].keys())
            else:
                self.sample_keys = self.sample_keys.intersection(set(self.samples[filepath].keys()))

            self._update_data(self.samples[filepath], self.samples[filepath].keys())
            try:
                self.table = self.make_table(self.name, "*data", "*names")
            except:
                self.error(traceback.format_exc())
                self.error("Couldn't make datatable with added elements!")
                self.table = None

    def get_dataset(
        self,
        keys=None,
    ):
        if keys is None:
            raise AttributeError("please choose data keys to add to sample dataset! \noptions: {0}".format(list(self.sample_keys)))
        assert hasattr(keys, "__iter__") or isinstance(keys, basestring), "keys must be iterable of strings!"

        if isinstance(keys, str):
            keys = [keys]

        data_dict = self.data.copy()
        shapes = []
        for key in data_dict:
            keep = False
            for subkey in keys:
                if glob.fnmatch.fnmatch(key, subkey):
                    keep = True
            if not keep:
                del data_dict[key]
            else:
                shapes.append(data_dict[key].shape)

        types = [v.dtype.kind for v in data_dict.values()]
        is_string = [t == 'S' for t in types]
        if any(is_string):
            if all(is_string):
                return np.concatenate(data_dict.values()), data_dict
            raise AttributeError, "Cannot ask for mixed type datasets! types: {0}".format(types)

        self.log("Grabbing dataset with keys {0}".format(list(data_dict.keys())))

        samples = set([x.shape[0] for x in data_dict.values()])
        assert len(samples) == 1
        sample_size = samples.pop()

        sizes = [reduce(mul, x.shape[1:], 1) for x in data_dict.values()]
        splits = [0,] + [sum(sizes[:i+1]) for i in range(len(sizes))]

        dataset = np.empty((sample_size, sum(sizes)))

        for i, datum in enumerate(data_dict.values()):
            dataset[:,splits[i]:splits[i + 1]] = datum.reshape(datum.shape[0], sizes[i])
        # self.log("Dataset shape: {0}".format(dataset.shape))
        return dataset, data_dict

    def save(
        self,
        filepath,
        force=False,
    ):
        """saves the current sample sets as one h5 file to the filepath specified"""
        filepath = smartpath(filepath)
        if not filepath.endswith('.h5'):
            filepath += '.h5'

        if os.path.exists(filepath) and not force:
            self.error("Path '{0}' already contains data. Use the 'force' argument to force saving".format(filepath))
            return 1

        f = h5py.File(filepath, "w")
        for key,data in self.data.items():
            f.create_dataset(key, data=data)

        self.log("Saving current dataset to file '{0}'".format(filepath))
        f.close()
        return 0

    # def plot(
    #     self,
    #     *args,
    #     **kwargs
    # ):
    #     return self.table.plot(*args, **kwargs)

    def make_table(
        self,
        name=None,
        value_keys="*data",
        header_keys="*names"
    ):
        values, vdict = self.get_dataset(value_keys)
        headers, hdict = None if header_keys is None else self.get_dataset(header_keys) 
        name = name or self.name

        assert len(values.shape) == 2, "data must be 2-dimensional and numeric!"
        assert values.shape[1] == headers.shape[0], "data must have the same number of columns as there are headers!!"

        return data_table(values, headers, name)

    # def norm_min_max(
    #     self,
    #     dataset,
    #     ab=(0,1)
    # ):
    #     if isinstance(dataset, basestring) or isinstance(dataset, list):
    #         dataset = self.get_dataset(dataset)
    #     a,b = ab
    #     rng = dataset.min(axis=0), dataset.max(axis=0)
    #     return (b-a)*(dataset - rng[0])/(rng[1] - rng[0]) + a, rng

    # def inorm_min_max(
    #     self,
    #     dataset,
    #     rng,
    #     ab=(0,1)
    # ):
    #     a,b = ab
    #     return ((dataset - a)/(b - a))*(rng[1] - rng[0]) + rng[0]
        
    # def norm_mean_std(
    #     self,
    #     dataset,
    # ):
    #     if isinstance(dataset, basestring) or isinstance(dataset, list):
    #         dataset = self.get_dataset(dataset)

    #     musigma = dataset.mean(axis=0), dataset.std(axis=0)
    #     return (dataset - musigma[0])/(musigma[1]), musigma

    # def inorm_mean_std(
    #     self,
    #     dataset,
    #     musigma,
    # ):
    #     return dataset*musigma[1] + musigma[0]
        
    def _update_data(
        self,
        sample_file,
        keys_to_add,
    ):
        for key in keys_to_add:
            if key in sample_file.keys():
                self._add_key(key, sample_file, self.data)
             
    @staticmethod
    def _add_key(
        key,
        sfile,
        d
    ):
        if key not in d:
            d[key] = np.asarray(sfile[key])
        else:
            # string data: no dupes!
            if d[key].dtype.kind == 'S':
                assert d[key].shape == sfile[key].shape
                assert all([k1 == k2 for k1,k2 in zip(d[key], sfile[key])])
            else:
                assert d[key].shape[1:] == sfile[key].shape[1:]
                d[key] = np.concatenate([d[key], sfile[key]])
    
def smartpath(path):
    if path.startswith("~/"):
        return path
    return os.path.abspath(path)

def get_cutflow_table(glob_path):
        
    paths = glob.glob(glob_path)
    assert len(paths) > 0, "must have SOME paths"
    
    values = []
    keys = []
    for path in paths:
        with open(path) as f:
            values_comp, keys_comp = map(lambda x: x.strip('\n').split(','), f.readlines())
            values_comp = map(int, values_comp)

def get_training_data(glob_path):
    paths = glob.glob(glob_path)
    d = data_loader("main sample")
    for p in paths:
        d.add_sample(p)
    return d.make_table()

def get_training_data_jets(glob_path):
    return split_to_jets(get_training_data(glob_path))

def get_subheaders(data):
    classes = {}
    i = 0
    n = 0
    h = data.headers
    while i < len(h):
        if str(n) not in h[i]:
            n += 1
            continue
        rep = h[i]
        if "j{}".format(n) in rep:
            rep = rep.replace("j{}".format(n), "jet")
        elif "jet{}".format(n) in rep:
            rep = rep.replace("jet{}".format(n), "jet")
        if n not in classes:
            classes[n] = []
        classes[n].append(rep)
        i += 1
    return classes

def split_to_jets(data):
    """
    given a data table with values for the n leading jets, split into one data 
    table for all jets.
    """
    headers = get_subheaders(data)
    assert len(set().union(*headers.values())) == len(headers.values()[0])
    jets = []
    next = data
    for h in headers:
        to_add, next = next.split_by_column_names("jet{}*".format(h))
        if to_add.shape[1] == 0:
            to_add, next = next.split_by_column_names("j{}*".format(h))
        jets.append(
            data_table(
                data=np.asarray(to_add),
                headers=headers[h],
                name="jet {}".format(h)
            )
        )

    full = data_table(
        data=np.vstack([jt.df for jt in jets]),
        headers=jets[0].headers,
        name="all jet data"
    )
    return full, jets

def log_uniform(low, high, size=None, base=10.):    
    return float(base)**(np.random.uniform(np.log(low)/np.log(base), np.log(high)/np.log(base), size))