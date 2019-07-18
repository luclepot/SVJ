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
import subprocess

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
            data = np.asarray(data)
            if len(data.shape) < 2:
                data = np.expand_dims(data, 1)
            self.data = data
        elif isinstance(data, pd.DataFrame):
            self.headers = data.columns 
            self.data = data.values
        elif isinstance(data, data_table):
            self.headers = data.headers
            self.data = data.df.values
            self.name = data.name
        else:
            data = np.asarray(data)
            if len(data.shape) < 2:
                data = np.expand_dims(data, 1)

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
        out_name=None,
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

        if out_name is None:
            out_name = "'{}' normed to '{}'".format(data.name,self.name)

        ret = data_table(self.scaler.transform(data.df), headers = self.headers, name=out_name)
        return ret

    def inorm(
        self,
        data=None,
        norm_type=0,
        out_name=None,
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

        if out_name is None:
            out_name = "'{}' inv_normed to '{}'".format(data.name,self.name)

        ret = data_table(self.scaler.inverse_transform(data.df), headers=self.headers, name=out_name)
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
        fontsize=10,
        normed=0,
        figloc="lower right",
        figsize=16,
        alpha=0.7,
        xscale="linear",
        yscale="linear",
        histtype='step',
        figname="Untitled",
        savename=None,
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

        if not isinstance(figsize, tuple):
            figsize = (figsize, rows*float(figsize)/cols)

        self.log("plotting distrubution(s) for table(s) {}".format([self.name,] + [o.name for o in others]))
        plt.rcParams['figure.figsize'] = figsize

        for i in range(n):
            ax = plt.subplot(rows, cols, i + 1)
            # weights = np.ones_like(plot_data[i])/float(len(plot_data[i]))
            ax.hist(plot_data[i], bins=bins, range=rng[i], histtype=histtype, normed=normed, label=self.name, weights=weights, alpha=alpha)
            for j in range(len(others)):
                # weights = np.ones_like(plot_others[j][i])/float(len(plot_others[j][i]))
                ax.hist(plot_others[j][i], bins=bins, range=rng[i], histtype=histtype, label=others[j].name, normed=normed, weights=weights, alpha=alpha)

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
        plt.suptitle(figname)
        plt.tight_layout(pad=0.01, w_pad=0.01, h_pad=0.01, rect=[0, 0.03, 1, 0.95])
        if savename is None:
            plt.show()
        else:
            plt.savefig(savename)
    
    def cdrop(
        self,
        globstr,
        inplace=False,
    ):
        to_drop = list(parse_globlist(globstr, list(self.df.columns)))
        modify = None
        if inplace:
            modify = self
        else:
            ret = data_table(self)
            modify = ret
        for d in to_drop:
            modify.df.drop(d, axis=1, inplace=True)
        modify.headers = list(modify.df.columns)
        return modify

    def cfilter(
        self, 
        globstr,
        inplace=False,
    ):
        to_keep = parse_globlist(globstr, list(self.df.columns))
        to_drop = set(self.headers).difference(to_keep)

        modify = None
        if inplace:
            modify = self
        else:
            ret = data_table(self)
            modify = ret
        for d in to_drop:
            modify.df.drop(d, axis=1, inplace=True)
        modify.headers = list(modify.df.columns)
        return modify

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

def parse_globlist(glob_list, match_list):
    if not hasattr(glob_list, "__iter__") or isinstance(glob_list, str):
        glob_list = [glob_list]
    
    for i,x in enumerate(glob_list):
        if isinstance(x, int):
            glob_list[i] = match_list[x]
        
    assert all([isinstance(c, str) for c in glob_list])

    match = set()
    for g in glob_list:
        match.update(glob.fnmatch.filter(match_list, g))

    return match

delphes_jet_tags_dict = {
    1: "down",
    2: "up",
    3: "strange",
    4: "charm",
    5: "bottom",
    6: "top",
    21: "gluon",
    9: "gluon"
}

def plot_error_ratios(main_error, compare_errors, metric='mse', bins= 40, log=False, rng=None, alpha=0.6):
    import matplotlib.pyplot as plt
    raw_counts, binned = np.histogram(main_error[metric], bins=bins, normed=False, range=rng)
    raw_counts = raw_counts.astype(float)
    
    zeros = np.where(raw_counts == 0)[0]
    if len(zeros) > 0:
        cutoff_index = zeros[0]
    else:
        cutoff_index = len(raw_counts)
    raw_counts = raw_counts[:cutoff_index]
    binned = binned[:cutoff_index + 1]
        
    ratios = []
    for e in compare_errors:
        counts, _ = np.histogram(e[metric], bins=bins, normed=False, range=rng)
        counts = counts.astype(float)[:cutoff_index]
        ratio = counts/raw_counts
        ratio_plot = ratio*(main_error.shape[0]/e.shape[0])
        ratios.append((ratio, raw_counts, counts))
        toplot = np.asarray(list(ratio_plot) + [0])
        err = np.asarray(list(1/counts) + [0])
        plt.plot(binned, toplot, label=e.name.lstrip("error ") + " ({0})".format(e.shape[0]), marker='o', alpha=alpha)
        if log:
            plt.yscale("log")
    plt.legend()
    plt.show()
    return ratios

def get_errors(true, pred, out_name="errors", functions=["mse", "mae"], names=[None, None]):
    import tensorflow as tf
    import keras
    if names is None:
        names = ['err {}'.format(i) for i in range(len(functions))]
    
    functions_keep = []
    for i,f in enumerate(functions):
        if isinstance(f, str):
            fuse = getattr(keras.losses, f)
            functions_keep.append(fuse)
            names[i] = f
        else:
            functions_keep.append(f)
    
    raw = [func(true, pred) for func in functions_keep]
    raw = np.asarray(map(lambda x: keras.backend.eval(x) if isinstance(x, tf.Tensor) else x, raw)).T
    return data_table(
        raw,
        headers=[str(f) for f in names],
        name=out_name
    )

def split_table_by_column(column_name, df, tag_names=None, keep_split_column=False, df_to_write=None):
    if df_to_write is None:
        df_to_write = df
    tagged = []
    unique = set(df.loc[:,column_name].values)
    if tag_names is None:
        tag_names = dict([(u, str(u)) for u in unique])

    assert df.shape[0] == df_to_write.shape[0], 'writing and splitting dataframes must have the same size!'

    gb = df.groupby(column_name)
    index = gb.groups
    for region, idx in index.items():
        if keep_split_column or column_name not in df_to_write:
            tagged.append(data_table(df_to_write.iloc[idx], name=tag_names[region]))
        else:
            tagged.append(data_table(df_to_write.iloc[idx].drop(column_name, axis=1), name=tag_names[region]))
    return tagged, dict([(tag_names[k], v) for k,v in index.items()])

def smartpath(path):
    if path.startswith("~/"):
        return path
    return os.path.abspath(path)

def get_cutflow_table(glob_path):
    paths = glob.glob(glob_path)
    assert len(paths) > 0, "must have SOME paths"

    ret = odict()
    for path in paths:
        with open(path) as f:
            values_comp, keys_comp = map(lambda x: x.strip('\n').split(','), f.readlines())
            values_comp = map(int, values_comp)
            keys_comp = map(str.strip, ['no cut'] + keys_comp)
            for k,v in zip(keys_comp, values_comp):
                if k not in ret:
                    ret[k] = 0
                ret[k] = ret[k] + v
    df = pd.DataFrame(ret.items(), columns=['cut_name', 'n_events'])
    df['abs eff.'] = np.round(100.*(df.n_events / df.n_events[0]), 2)
    df['rel eff.'] = np.round([100.] + [100.*(float(df.n_events[i + 1]) / float(df.n_events[i])) for i in range(len(df.n_events) - 1)], 2)
    
    return df

def get_training_data(glob_path, verbose=1):
    paths = glob.glob(glob_path)
    d = data_loader("main sample", verbose=verbose)
    for p in paths:
        d.add_sample(p)
    return d.make_table()

def get_training_data_jets(glob_path, verbose=1):
    return split_to_jets(get_training_data(glob_path, verbose))

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

def get_selections_dict(list_of_selections):
    ret = {}
    for sel in list_of_selections:
        with open(sel, 'r') as f:
            data = map(lambda x: x.strip('\n'), f.readlines())
        for elt in data:
            key, raw = elt.split(': ')
            ret[key] = map(int, raw.split())
    return ret

def get_repo_info():
    info = {}
    info['head'] = subprocess.Popen("git rev-parse --show-toplevel".split(), stdout=subprocess.PIPE).communicate()[0].strip('\n')
    info['name'] = subprocess.Popen("git config --get remote.origin.url".split(), stdout=subprocess.PIPE).communicate()[0].strip('\n')
    return info

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

def split_by_tag(data, tag_column="jetFlavor", printout=True):
    tagged, tag_index = split_table_by_column(
        "jetFlavor",
        data,
        delphes_jet_tags_dict,
        False
    )
    if printout:
        sizes = map(lambda x: x.shape[0], tagged)
        for t,s in zip(tagged, sizes):
            print  "{} jet: {}, {}%".format(t.name, s, round(100.*s/sum(sizes), 1))
        
    return tagged, tag_index
    
def compare_tags(datasets):
    
    tags = map(lambda x: dict([(t.name, t) for t in split_by_tag(x, printout=False)[0]]), datasets)
    tag_ids = set().union(*[set([tn for tn in tlist]) for tlist in tags])
    
    for tag_id in tag_ids:
        print "{}:".format(tag_id)
        for t,d in zip(tags, datasets):
            
            if tag_id in t:
                tag = t[tag_id]
                print "\t{:.1f}% ({}) {}".format(100.*tag.shape[0]/d.shape[0], tag.shape[0], d.name)
            
def get_recon_errors(data_list, autoencoder, **kwargs):

    if not isinstance(data_list, list):
        data_list = [data_list]
    
    recon = []
    errors = []
    
    for i,d in enumerate(data_list):
        recon.append(
            data_table(
                autoencoder.predict(d.data),        
                headers=d.headers,
                name="{0} pred".format(d.name)
            )
        )
        errors.append(
            get_errors(recon[i].data, d.data, out_name="{0} error".format(d.name), **kwargs)
        )
        
    return errors, recon

def roc_auc(data_err, signal_errs):
    ret = {}
    for signal_err in signal_errs:
        pred = np.hstack([signal_err[metric].values, data_err[metric].values])
        true = np.hstack([np.ones(signal_err.shape[0]), np.zeros(data_err.shape[0])])

        roc = roc_curve(true, pred)
        auc = roc_auc_score(true, pred)

        ret[signal_err.name] = {'roc': roc, 'auc': auc}
    return ret

def roc_auc_plot(data_errs, signal_errs, metrics='loss', *args, **kwargs):
    from sklearn.metrics import roc_curve, roc_auc_score
    if not isinstance(metrics, list):
        metrics = [metrics]

    if not isinstance(signal_errs, list):
        signal_errs = [signal_errs]

    if not isinstance(data_errs, list):
        data_errs = [data_errs]
    
    if len(data_errs) == 1:
        data_errs = [data_errs[0] for i in range(len(signal_errs))]
        
    fig, ax_begin, ax_end, plt_end, colors = get_plot_params(1, *args, **kwargs)
    ax = ax_begin(0)
    styles = [ '-','--','-.',':']
    for i,(data_err,signal_err) in enumerate(zip(data_errs, signal_errs)):
	
        for j,metric in enumerate(metrics):
            pred = np.hstack([signal_err[metric].values, data_err[metric].values])
            true = np.hstack([np.ones(signal_err.shape[0]), np.zeros(data_err.shape[0])])

            roc = roc_curve(true, pred)
            auc = roc_auc_score(true, pred)
        
            ax.plot(roc[0], roc[1], styles[j%len(styles)], c=colors[i%len(colors)], label='{} {}, AUC {:.4f}'.format(signal_err.name, metric, auc))

    ax.plot(roc[0], roc[0], '--', c='black')
    ax_end("false positive rate", "true positive rate")
    plt_end()
    plt.show()
    

def load_all_data(data_path, name, cols_to_drop = ["jetM", "*MET*", "*Delta*"]):
    """Returns: a tuple with... 
        a data_table with all data, (columns dropped),
        a list of data_tables, by jetFlavor tag (columns dropped),
        a list of data_tables, split by jet # (columns dropped)
    """
    repo_head = get_repo_info()['head']

    if not os.path.exists(data_path):
        assert len(repo_head) > 0, "not running jupyter notebook from within a repo!! Prolly won't work :-)"
        data_path = os.path.join(repo_head, data_path)
    
    # if not os.path.exists(signal_path):
    #     assert len(repo_head) > 0, "not running jupyter notebook from within a repo!! Prolly won't work :-)"
    #     signal_path = os.path.join(repo_head, signal_path)
        
    # if not os.path.exists(model_path):
    #     assert len(repo_head) > 0, "not running jupyter notebook from within a repo!! Prolly won't work :-)"
    #     model_path = os.path.join(repo_head, model_path)

    data, data_jets = get_training_data_jets(data_path, 0)

    for i,d in enumerate(data_jets):
        data_jets[i].name = d.name + " " + name

    # signal, signal_jets = get_training_data_jets(signal_path, 0)

    data.name = name

    data.cdrop(cols_to_drop, inplace=True)
    # signal.cdrop(cols_to_drop, inplace=True)
    
    print "{} tags:".format(name)
    print "" 
    tagged_data, tag_index_data = split_by_tag(data)
    print ""
    # print "signal tags:"
    # print ""
    # tagged_signal, tag_index_signal = split_by_tag(signal)
    # print ""

    return data, tagged_data, data_jets

def evaluate_model(data_path, signal_path, model_path):
    # get h5 datasets
    repo_head = get_repo_info()['head']

    if not os.path.exists(data_path):
        assert len(repo_head) > 0, "not running jupyter notebook from within a repo!! Prolly won't work :-)"
        data_path = os.path.join(repo_head, data_path)
    
    if not os.path.exists(signal_path):
        assert len(repo_head) > 0, "not running jupyter notebook from within a repo!! Prolly won't work :-)"
        signal_path = os.path.join(repo_head, signal_path)
        
    if not os.path.exists(model_path):
        assert len(repo_head) > 0, "not running jupyter notebook from within a repo!! Prolly won't work :-)"
        model_path = os.path.join(repo_head, model_path)

        
    data, data_jets = get_training_data_jets(data_path, 0)
    signal, signal_jets = get_training_data_jets(signal_path, 0)

    signal.name = "signal"
    data.name = "background"

    # drop some unused columns
    cols_to_drop = ["jetM", "*MET*", "*Delta*"]

    data.cdrop(cols_to_drop, inplace=True)
    signal.cdrop(cols_to_drop, inplace=True)
    
    print "data tags:"
    print "" 
    tagged_data, tag_index_data = split_by_tag(data)
    print ""
    print "signal tags:"
    print ""
    tagged_signal, tag_index_signal = split_by_tag(signal)
    print ""
    
    data_raw = data.cdrop("*Flavor")
    signal_raw = signal.cdrop("*Flavor")

    norm_args = {
        "norm_type": "StandardScaler",
    #     "feature_range": (0.01, 0.99)
    }

    data_norm = data_raw.norm(**norm_args)
    signal_norm = data_raw.norm(signal_raw, **norm_args)
    

    instance = trainer.trainer(model_path, verbose=False)
    autoencoder = instance.load_model()
    encoder, decoder = autoencoder.layers[1:]
    
    tagged_norm = [data_raw.norm(t, **norm_args) for t in tagged_data]

    errors, recon = get_recon_errors([data_norm, signal_norm] + tagged_norm, autoencoder)
    
    data_recon, signal_recon, tagged_recon = recon[0], recon[1], recon[2:]
    data_error, signal_error, tagged_error = errors[0], errors[1], errors[2:]
    
    roc_auc_plot(data_error, signal_error)
    
    data_recon.name = "background pred"
    signal_recon.name = "signal pred"
    data_norm.name = "background (norm)"
    signal_norm.name = "signal (norm)"
    
    data_norm.plot(
        [data_recon, signal_norm, signal_recon],
        normed=1, bins=70, cols=2, figsize=(10,20),
#         rng=((0,2), (-3.2, 3.2), (0,2500), (0,1), (0, 0.8),(0,.1), (0,3000)),
        figname="signal/data comparison"
    )
    
    signal_error.plot(data_error, normed=1, bins=100, figname="signal vs data errors")

def get_plot_params(
    n_plots,
    cols=4,
    figsize=20.,
    yscale='linear',
    xscale='linear',
    figloc='lower right',
    figname='Untitled',
    savename=None,
    ticksize=8,
    fontsize=5,
):
    rows =  n_plots/cols + bool(n_plots%cols)
    if n_plots < cols:
        cols = n_plots
        rows = 1
        
    if not isinstance(figsize, tuple):
        figsize = (figsize, rows*float(figsize)/cols)
    
    fig = plt.figure(figsize=figsize)
    
    def on_axis_begin(i):
        return plt.subplot(rows, cols, i + 1)    
    
    def on_axis_end(xname, yname=''):
        plt.xlabel(xname + " ({0}-scaled)".format(xscale))
        plt.ylabel(yname + " ({0}-scaled)".format(yscale))
        plt.xticks(size=ticksize)
        plt.yticks(size=ticksize)
        plt.xscale(xscale)
        plt.yscale(yscale)
        plt.gca().spines['left']._adjust_location()
        plt.gca().spines['bottom']._adjust_location()
        
    def on_plot_end():
        handles,labels = plt.gca().get_legend_handles_labels()
        by_label = odict(zip(map(str, labels), handles))
        plt.figlegend(by_label.values(), by_label.keys(), loc=figloc)
        # plt.figlegend(handles, labels, loc=figloc)
        plt.suptitle(figname)
        plt.tight_layout(pad=0.01, w_pad=0.01, h_pad=0.01, rect=[0, 0.03, 1, 0.95])
        if savename is None:
            plt.show()
        else:
            plt.savefig(savename)
            
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    return fig, on_axis_begin, on_axis_end, on_plot_end, colors

def plot_spdfs(inputs, outputs, bins=100, *args, **kwargs):
    if not isinstance(inputs, list):
        inputs = [inputs]
    if not isinstance(outputs, list):
        outputs = [outputs]
        
    # assert all([isinstance(inp, data_table) for inp in inputs]), "inputs mst be utils.data_table format"
    assert len(inputs) > 0, "must have SOME inputs"
    assert len(outputs) > 0, "must have SOME outputs"
    assert len(inputs) == len(outputs), "# of outputs and inputs must be the same"
    
    columns = inputs[0].headers
    assert all([columns == inp.headers for inp in inputs]), "all inputs must have identical column titles"

    fig, ax_begin, ax_end, plt_end, colors = get_plot_params(len(columns), *args, **kwargs)
    
    for i,name in enumerate(columns):
        
        ax_begin(i)

        for j, (IN, OUT) in enumerate(zip(inputs, outputs)):

            dname = IN.name
            centers, (content, content_new), width = get_bin_content(IN.data[:,i], OUT[0][:,i], OUT[1][:,i], bins) 

            sfac = float(IN.shape[0])
            plt.errorbar(centers, content/sfac, xerr=width/2., yerr=np.sqrt(content)/sfac, fmt='.', c=colors[j], label='{} input'.format(dname))
            plt.errorbar(centers, content_new/sfac, xerr=width/2., fmt='--', c=colors[j], label='{} spdf'.format(dname), alpha=0.7)
    #     plt.hist(mu, histtype='step', bins=bins)

        ax_end(name)
        
    plt_end()

def get_bin_content(aux, mu, sigma, bins=50):
    
    hrange = (np.percentile(aux, 0.1), np.percentile(aux, 99.9))
    
    content, edges = np.histogram(aux, bins=bins, range=hrange)
    centers = 0.5*(edges[1:] + edges[:-1])
    
    width = centers[1] - centers[0]
    
    bin_content = np.sum(content)*width*sum_of_gaussians(centers, mu, sigma)
    
    return centers, (content, bin_content), width

def sum_of_gaussians(x, mu_vec, sigma_vec):
    x = np.atleast_2d(x)
    if x.shape[0] <= x.shape[1]:
        x = x.T
    x_norm = (x - mu_vec)/sigma_vec
    single_gaus_val = np.exp(-0.5*np.square(x_norm))/(sigma_vec*np.sqrt(2*np.pi))
    return np.sum(single_gaus_val, axis=1)/mu_vec.shape[0]
