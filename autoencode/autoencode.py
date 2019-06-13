import numpy as np 
from keras.layers import Input, Dense
from keras.models import Model
import keras.optimizers
import keras.initializers
import keras.callbacks
from collections import OrderedDict as odict
import os
from operator import mul
import h5py
import matplotlib.pyplot as plt
import glob
from sklearn.model_selection import train_test_split
import pandas as pd
import random 
import os
import json
import traceback
from datetime import datetime
import dateutils

def _smartpath(
    path,
):
    if path.startswith("~/"):
        return path
    return os.path.abspath(path)

def compare_features(
    true,
    preds,
    bins=30,
    max_features=20,
    cols=4,
    dataset_names=None,
    labels=None,
    **kwargs
):
    n = min(true.shape[1], max_features)
    rows = n/cols + bool(n%cols)

    if labels is None:
        labels = ["var {0}".format(i) for i in range(n)]

    if not hasattr(preds, "__iter__"):
        preds = [preds]

    if dataset_names is None:
        dataset_names = ["pred {}".format(i) for i in range(len(preds))]

    for i in range(n):
        gmax, gmin = max(true[:,i].max(), pred[:,i].max()), min(true[:,i].min(), pred[:,i].min())
        plt.subplot(rows, cols, i + 1)
        plt.hist(true[:,i], bins=bins, range=(gmin,gmax), histtype="step", label=labels[i] + " true", **kwargs)
        for pred in preds:
            plt.hist(pred[:,i], bins=bins, range=(gmin,gmax), histtype="step", label=labels[i] + " " + " pred", **kwargs)
        plt.legend()

    plt.show()

class logger:

    def __init__(
        self,
    ):
        self._LOG_PREFIX = "logger :: "
        self._ERROR_PREFIX = "ERROR: "
        self.VERBOSE = True

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

class data_table:

    def __init__(
        self,
        data,
        headers=None,
        name=None
    ):
        self.name = name or "untitled"
        self.data = data
        self.headers = headers if headers is not None else ["dist " + str(i + 1) for i in range(self.data.shape[1])]
        assert len(self.data.shape) == 2, "data must be matrix!"
        assert len(self.headers) == self.data.shape[1], "n columns must be equal to n column headers"
        assert len(self.data) > 0, "n samples must be greater than zero"
        self.df = pd.DataFrame(self.data, columns=self.headers)
        
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

    def train_test_split(
        self,
        test_fraction=0.25
    ):
        dtrain, dtest = train_test_split(self, test_size=test_fraction)
        return (data_table(np.asarray(dtrain), np.asarray(dtrain.columns), "train"),
            data_table(np.asarray(dtest), np.asarray(dtest.columns), "test"))

    def plot(
        self,
        values,
        others=[],
        bins=15,
        rng=None,
        cols=4,
        ticksize=8,
        fontsize=10,
        normed=0,
    ):
        if isinstance(values, str):
            values = [key for key in self.headers if glob.fnmatch.fnmatch(key, values)]
        if not hasattr(values, "__iter__"):
            values = [values]
        for i in range(len(values)):
            if isinstance(values[i], int):
                values[i] = self.headers[values[i]]
        
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
            rng = [(p.min(), p.max()) for p in plot_data]
        elif len(rng) == 2 and all([not hasattr(r, "__iter__") for r in rng]):
            rng = [rng for i in range(len(plot_data))]

        weights = None

        for i in range(n):
            plt.subplot(rows, cols, i + 1)
            # weights = np.ones_like(plot_data[i])/float(len(plot_data[i]))
            plt.hist(plot_data[i], bins=bins, range=rng[i], histtype='step', normed=normed, label=self.name, weights=weights)
            for j in range(len(others)):
                # weights = np.ones_like(plot_others[j][i])/float(len(plot_others[j][i]))
                plt.hist(plot_others[j][i], bins=bins, range=rng[i], histtype='step', label=others[j].name, normed=normed, weights=weights)
            plt.xlabel(plot_data[i].name, fontsize=fontsize)
            plt.xticks(size=ticksize)
            plt.yticks(size=ticksize)
        
        plt.legend()
        plt.tight_layout(pad=0.01, w_pad=0.01, h_pad=0.01)
        plt.show()
    
    def _rows(
        self,
        cols,
        n
    ):
        return n/cols + bool(n%cols)

class data_loader(logger):

    def __init__(
        self,
        verbose=True
    ):
        logger.__init__(self)
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
        filepath = _smartpath(sample_path)
        assert os.path.exists(filepath)
        if filepath not in self.samples:
            self.log("Adding sample at path '{}'".format(filepath))
            self.samples[filepath] = h5py.File(filepath)
            if self.sample_keys is None:
                self.sample_keys = set(self.samples[filepath].keys())
            else:
                self.sample_keys = self.sample_keys.intersection(set(self.samples[filepath].keys()))

            self._update_data(self.samples[filepath], self.samples[filepath].keys())
            self.table = self.make_table("*data", "*names", "main sample")

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
            raise ArgumentError, "Cannot ask for mixed type datasets! types: {0}".format(types)

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
        filepath = _smartpath(filepath)
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

    def plot(
        self,
        *args,
        **kwargs
    ):
        return self.table.plot(*args, **kwargs)

    def make_table(
        self,
        value_keys,
        header_keys=None,
        name=None,
    ):
        values, vdict = self.get_dataset(value_keys)
        headers, hdict = None if header_keys is None else self.get_dataset(header_keys) 

        assert len(values.shape) == 2, "data must be 2-dimensional and numeric!"
        assert values.shape[1] == headers.shape[0], "data must have the same number of columns as there are headers!!"

        return data_table(values, headers, name)

    def norm_min_max(
        self,
        dataset,
        ab=(0,1)
    ):
        if isinstance(dataset, basestring) or isinstance(dataset, list):
            dataset = self.get_dataset(dataset)
        a,b = ab
        rng = dataset.min(axis=0), dataset.max(axis=0)
        return (b-a)*(dataset - rng[0])/(rng[1] - rng[0]) + a, rng

    def inorm_min_max(
        self,
        dataset,
        rng,
        ab=(0,1)
    ):
        a,b = ab
        return ((dataset - a)/(b - a))*(rng[1] - rng[0]) + rng[0]
        
    def norm_mean_std(
        self,
        dataset,
    ):
        if isinstance(dataset, basestring) or isinstance(dataset, list):
            dataset = self.get_dataset(dataset)

        musigma = dataset.mean(axis=0), dataset.std(axis=0)
        return (dataset - musigma[0])/(musigma[1]), musigma

    def inorm_mean_std(
        self,
        dataset,
        musigma,
    ):
        return dataset*musigma[1] + musigma[0]
        
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
    
class base_autoencoder(logger):

    def __init__(
        self,
        name="autoencoder",
        verbose=True,
    ):
        logger.__init__(self)
        self._LOG_PREFIX = "base_autoencoder :: "
        self.VERBOSE = verbose
        self.name = name
        self.layers = []

        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.history = None

    def __str__(
        self,
    ):
        s = self.log('Current Structure:', True)
        for layer in self.layers:
            s += self.log("{0}: {1} nodes {2}".format(layer[0], layer[1], layer[2:]), True)
        return s

    def __repr__(
        self,
    ):
        return str(self)

    def remove(
        self,
        index=None,
    ):
        if index is None:
            index = -1
        self.layers.pop(index)

    def add(
        self,
        nodes,
        activation='relu',
        reg=None,
        name=None,
        bias_init='zeros',
        kernel_init='glorot_uniform'
    ):
        if name is None:
            name = "layer_{0}".format(len(self.layers) + 1)
        
        self.layers.append([name, nodes, activation, reg, bias_init, kernel_init])

    def build(
        self,
        encoding_index=None,
        optimizer='adam',
        loss='mse'
    ):

        assert len(self.layers) >= 3, "need to have input, bottleneck, output!"

        if encoding_index is None:
            encoding_index = self._find_bottleneck(self.layers[1:-1]) + 1

        # grab individual layers
        input_layer = self.layers[0]
        inner_interms = self.layers[1:encoding_index]
        encoded_layer = self.layers[encoding_index]
        outer_interms = self.layers[encoding_index + 1:-1]
        output_layer = self.layers[-1]

        # get necessary keras layers
        inputs = self._input(input_layer)
        encoded = self._add_layer(encoded_layer, self._add_layers(inner_interms, inputs))
        encoded_input = self._input(encoded_layer)
        outputs = self._add_layer(output_layer, self._add_layers(outer_interms, encoded_input))

        # make keras models for encoder, decoder, and autoencoder
        self.encoder = Model(inputs, encoded, name='encoder')
        self.decoder = Model(encoded_input, outputs, name='decoder')
        self.autoencoder = Model(inputs, self.decoder(self.encoder(inputs)), name='autoencoder')

        self.autoencoder.compile(optimizer, loss)

        return self.encoder, self.decoder, self.autoencoder

    def fit(
        self,
        x,
        y,
        validation_data,
        batch_size=10,
        *args,
        **kwargs
    ):

        if self.autoencoder is None: 
            raise AttributeError, "Model not yet built!"

        try:
            self.history = self.autoencoder.fit(
                x=x,
                y=y,
                steps_per_epoch=int(np.ceil(len(x)/batch_size)),
                validation_steps=int(np.ceil(len(validation_data[0])/batch_size)),
                validation_data=validation_data,
                *args,
                **kwargs
            )

        except KeyboardInterrupt:
            return None

        return self.history

    def save(
        self,
        path,
    ):
        path = _smartpath(path)
        path = _smartpath(path)
        if not os.path.exists(path):
            os.mkdirs(path)
        to_save = ['autoencoder', 'encoder', 'decoder']
        filenames = [os.path.join(path, model + '.h5') for model in to_save]
        for filename,typename in zip(filenames, to_save):
            if os.path.exists(_smartpath(filename)):
                raise AttributeError, "Model already exists at file '{}'!!".format(filename)
            getattr(self, typename).save(filename)

    def load(
        self,
        path,
    ):
        to_load = ['autoencoder', 'encoder', 'decoder']
        filenames = [os.path.join(path, model + '.h5') for model in to_load]
        for filename,typename in zip(filenames, to_load):
            if not os.path.exists(_smartpath(filename)):
                raise AttributeError, "Model does not exist at file '{}'!!".format(filename)
            setattr(self, typename, keras.models.load_model(filename))

    def _find_bottleneck(
        self,
        layers,
    ):
        imin = 0
        lmin = layers[0][1]
        for i,layer in enumerate(layers):
            if layer[1] < lmin:
                imin = i
                lmin = layer[1]
        return imin

    def _add_layers(
        self,
        layers,
        base_layer,
    ):
        lnext = base_layer
        for layer in layers:
            temp = lnext
            lnext = Dense(layer[1], activation=layer[2], activity_regularizer=layer[3], name=layer[0], bias_initializer=layer[4], kernel_initializer=layer[5])(temp)
        return lnext

    def _add_layer(
        self,
        layer,
        base_layer,
    ):
        return Dense(layer[1], activation=layer[2], activity_regularizer=layer[3], name=layer[0], bias_initializer=layer[4], kernel_initializer=layer[5])(base_layer)

    def _input(
        self,
        layer,
    ):
        return Input(shape=(layer[1],), name=layer[0])

    # return locals()

class h5_element_wrapper(logger):
    def __init__(
        self,
        h5file,
        group,
        name,
        istype=None,
        verbose=True,
        overwrite=False
    ):
        self._LOG_PREFIX = "h5_elt '{}' :: ".format(name)
        self.VERBOSE = verbose

        self.h5file = h5file
        self.group = group
        self.name = name
        self.fnamefull = self.h5file.filename
        self.fname = os.path.basename(self.fnamefull)

        if self.group not in self.h5file.keys():
            self.log("creating group '{}' in file '{}'".format(self.group, self.fname))
            self.h5file.create_group(self.group)

        if self.name not in self.h5file[self.group].keys():
            self.log("creating dataset '{}/{}' in file '{}'".format(self.group, self.name, self.fname))
            self.h5file[self.group].create_dataset(self.name,(0,))

        elif overwrite:
            del self.h5file[self.group][self.name]
            self.log("recreating dataset '{}/{}' in file '{}'".format(self.group, self.name, self.fname))
            self.h5file[self.group].create_dataset(self.name,(0,))
        else:
            self.log("loading dataset '{}/{}' from file '{}'".format(self.group, self.name, self.fname))

        self.core = self.h5file[self.group][self.name]

        self.comp = np.asarray if istype is None else self.dict_comp
        self.conv = np.asarray if istype is None else istype
        self.rep = self.conv(self.core)

    def __str__(
        self
    ):
        return str(self.rep)

    def __repr__(
        self,
    ):
        return repr(self.rep)

    def dict_comp(
        self,
        dict_in,
    ):
        return np.asarray([dict_in.keys(), dict_in.values()], dtype="S").T

    def update(
        self,
        new_data,
    ):
        if self.name in self.h5file[self.group]:
            del self.h5file[self.group][self.name]
        self.h5file[self.group].create_dataset(name=self.name, data=self.comp(new_data))
        self.core = self.h5file[self.group][self.name]
        self.rep = self.conv(self.core)

    def h5(
        self,
    ):
        return self.core

    def empty(
        self,
    ):
        return len(self) == 0

    def __getattr__(
        self,
        attr,
    ):
        return getattr(self.rep, attr)

class training_shell(logger):
    """
    Wraps training/testing/evaluation activity for a model in an h5 file saver, which keeps all
    training inputs/outputs, model performance stats, model weights, etc.
    """

    ### LIFE/DEATH

    def __init__(
        self,
        name,
        verbose=True,
        overwrite=False,
    ):
        logger.__init__(self)

        self._LOG_PREFIX = "train_shell :: "
        self.VERBOSE = verbose

        self.config_file = _smartpath(name)
    
        
        if not self.config_file.endswith(".h5"):
            self.config_file += ".h5"
            
        preexisting = os.path.exists(self.config_file) and not overwrite
        self.file = None

        if self.locked(self.config_file):
            self._throw("filename '{}' is already being edited in another instance!!".format(self.config_file))

        self.file = h5py.File(self.config_file)

        self._lock_file(self.config_file)

        try:
            file_attrs = {
                "params": (["training", "config"], dict),
                "data": (["true", "pred", "metrics", "metric_names"], None)
            }
            
            for attr in file_attrs:
                for subattr in file_attrs[attr][0]:
                    setattr(self, subattr, h5_element_wrapper(self.file, attr, subattr, file_attrs[attr][1], overwrite=overwrite))
                setattr(self, attr, self.file[attr])
                        
        except Exception as e:
            self.error(traceback.format_exc())
            self._throw(e, unlock=True)

        cdict = self.config.copy()
        
        if not preexisting:
            cdict = odict()
            cdict['name'] = name
            cdict['trained'] = ''
            cdict['model_file'] = ''
        
        self.config.update(cdict)

    def _throw(
        self,
        msg,
        exc=AttributeError,
        unlock=False,
    ):
        self.close(unlock)
        self.error(msg)
        raise exc, msg

    def close(
        self,
        unlock=False,
    ):
        if self.file is not None:
            self.file.close()
        if unlock:
            self._unlock_file(self.config_file)

    def __del__(
        self,
    ):
        self.close(unlock=True)

    ### LOCKING

    def _lock_file(
        self,
        fname,
    ):
        open(self._get_lock(fname), "w+").close()

    def _unlock_file(
        self,
        fname,
    ):
        os.remove(self._get_lock(fname))

    def locked(
        self,
        fname,
    ):
        return os.path.exists(self._get_lock(fname))

    def _get_lock(
        self,
        fname,
    ):
        return os.path.join(os.path.dirname(fname), ".lock." + os.path.basename(fname))

    ### ACTUAL WRAPPERS

    def train(
        self,
        x,
        y,
        model=None,
        epochs=10,
        batch_size=32,
        force=False
    ):
        if self.config['trained']:
            if not force:
                self._throw("Model already trained, with timestamp {}!!".format(self.config['trained']))
            self.log("forcing already trained model to re-train")
            if os.path.exists(self.config['model_file']):
                new_model = keras.models.load_model(self.config['model_file'])
                if model is not None:
                    assert model.get_params() == new_model.get_params(), 'input and saved models must be the same!'
                model = new_model
    
        start = datetime.now()
        history = []
        try:
            for epoch in range(epochs):
                history.append(
                        model.fit(
                        x=x,
                        y=y,
                        steps_per_epoch=int(np.ceil(len(x)/batch_size)),
                        validation_steps=int(np.ceil(len(validation_data[0])/batch_size)),
                        validation_data=validation_data,
                        initial_epoch=epoch,
                        epochs=epoch + 1,

                    ).history.values()
                )
            
        except:
            self.error(traceback.format_exc())
                    
        end = datetime.now()

        model_file = self.config_file.replace(".h5", "_model.h5")
        model.save(model_file)

        cdict = self.config.copy()
        cdict['model_file'] = model_file
        cdict['trained'] = str(start) + " ::: " + str(end)
        self.config.update(cdict)

        tdict = self.training.copy()
        tdict['time'] = str(end - start)

        self.training.update(tdict)

        # return self.history

    

def rand_search(n):
    def r(l):
        return l[random.randint(0, len(l) - 1)]
    optimizers = ['adam', 'adagrad']
    losses = ['mean_squared_error','mean_absolute_error']
    activations = ['linear', 'tanh']
    bottlenecks = [4,5,6,7,8]
    batches = [10,50,100]
    epochs = [40]
    pdict = odict()
    
    import json

    for i in range(n):

        opt = r(optimizers)
        loss = r(losses)
        act = r(activations)
        bn = r(bottlenecks)
        batch = r(batches)
        epoch = r(epochs)

        strname = "{}-{}-{}-{}-{}-{}".format(bn, opt, loss, act, batch, epoch)

        if strname in pdict:
            continue

        pdict[strname] = {}

        pdict[strname]['params'] = {
            'optimizer': opt,
            'loss': loss,
            'activation': act,
            'bottleneck': bn,
            'epochs': epoch,
            'batchsize': batch,
        }

        print strname

        pdict[strname]['out'] = basic_run(bn, opt, loss, act, epoch, batch)

        if os.path.exists("pdict.json"):
            old = json.load(open("pdict.json"))
            pdict.update(old)
        json.dump(pdict, open("pdict.json", 'w+'))

        print pdict.keys()

# b = rand_search(100)

class LEGACY:

    class basic_autoencoder(logger):

        def __init__(
            self,
            name,
            path=None,
        ):
            self.name = name
            self.path = _smartpath(path or os.path.curdir)
            self.BUILT = False
            self.PROCESSED = False
            self.BUILT_TRAINING_DATASET = False
            self.TRAINED = False
            self.samples = odict()
            self.processed = odict()
            self.data = odict()
            self.labels = odict()
            self.train_params = odict()
            self.build_params = odict()

        def add_sample(
            self,
            file,
        ):
            filepath = _smartpath(file)
            assert os.path.exists(filepath)
            if filepath not in self.samples:
                self.samples[filepath] = h5py.File(filepath)

        @staticmethod
        def add_key(
            key,
            keycheck,
            sfile,
            d
        ):
            if keycheck == key:
                if key not in d:
                    d[key] = np.asarray(sfile[key])
                else:
                    if d[key].dtype.kind == 'S':
                        assert d[key].shape == sfile[key].shape
                        assert all([k1 == k2 for k1,k2 in zip(d[key], sfile[key])])
                    else:
                        assert d[key].shape[1:] == sfile[key].shape[1:]
                        d[key] = np.concatenate([d[key], sfile[key]])

        def process_samples(
            self,
            save_result=False,
        ):
            for sample_path, sample_file in self.samples.items():
                if os.path.abspath(sample_path) in self.processed:
                    continue

                for key in sample_file.keys():
                    self.add_key(key, "event_feature_data", sample_file, self.data)
                    self.add_key(key, "event_feature_names", sample_file, self.labels)
                    self.add_key(key, "jet_constituent_data", sample_file, self.data)
                    self.add_key(key, "jet_constituent_names", sample_file, self.labels)

                self.processed[os.path.abspath(sample_path)] = True

            if save_result:
                f = h5py.File(os.path.join(self.path, "{0}_combined_data.h5".format(self.name)), "w")
                for key,data in self.data.items():
                    f.create_dataset(key, data=data)
                for key,data in self.labels.items():
                    f.create_dataset(key, data=data)
                f.close()

            self.PROCESSED = True

        def build_training_dataset(
            self,
        ):
            assert self.PROCESSED

            # make sure there are equal numbers of samples for all data
            samples = set([x.shape[0] for x in self.data.values()])
            assert len(samples) == 1
            sample_size = samples.pop()

            sizes = [reduce(mul, x.shape[1:], 1) for x in self.data.values()]
            splits = [0,] + [sum(sizes[:i+1]) for i in range(len(sizes))]
        
            self.train_dataset = np.empty((sample_size, sum(sizes)))

            for i, datum in enumerate(self.data.values()):
                self.train_dataset[:,splits[i]:splits[i + 1]] = datum.reshape(datum.shape[0], sizes[i])

            self.dmin, self.dmax = self.train_dataset.min(axis=0), self.train_dataset.max(axis=0)
            self.normalized = (self.train_dataset - self.dmin)/(self.dmax - self.dmin)

            self.BUILT_TRAINING_DATASET = True

        def reconstruct_dataset(
            self,
            dataset,
        ):
            assert self.BUILT_TRAINING_DATASET
            denorm = dataset*(self.dmax - self.dmin)
            denorm += self.dmin

            sizes = [reduce(mul, x.shape[1:], 1) for x in self.data.values()]
            splits = [0,] + [sum(sizes[:i+1]) for i in range(len(sizes))]

            recon = odict()
            for i, (key, value) in enumerate(self.data.items()):
                recon[key] = denorm[:,splits[i]:splits[i + 1]]

            return recon

        def build(
            self,
            bottleneck_dim,
            intermediate_dims=[],
            output_data=None,
            loss_function="mse",
            kinit="random_uniform",
            binit="random_uniform",
            reg=keras.regularizers.l1(10e-8),
            intermediate_activation='relu',
            encoded_activation='relu',
            output_activation='tanh',
        ):

            self.build_training_dataset()

            if output_data is None:
                output_data = self.data.keys()

            assert self.BUILT_TRAINING_DATASET

            self.n_samples, self.input_dim = self.train_dataset.shape
            self.output_dim = self.input_dim
            self.intermediate_dims = intermediate_dims
            self.bottleneck_dim = bottleneck_dim
            self.inputs = Input(shape=(self.input_dim,), name='encoder_input')

            if len(intermediate_dims) > 0:
                interms1 = []
                interms1.append(Dense(self.intermediate_dims[0],
                activation=intermediate_activation,
                kernel_initializer=kinit,
                bias_initializer=binit,
                    activity_regularizer=reg)(self.inputs))
                for i,dim in enumerate(self.intermediate_dims[1:]):
                    interms1.append(Dense(dim, activation=intermediate_activation,
                    kernel_initializer=kinit,
                    bias_initializer=binit,
                    activity_regularizer=reg)(interms1[i - 1]))
            else:
                interms1 = [self.inputs]

            encoded = Dense(self.bottleneck_dim, activation=encoded_activation, kernel_initializer=kinit, bias_initializer=binit, activity_regularizer=reg)(interms1[-1])

            self.encoder = Model(self.inputs, encoded, name='encoder')
            self.encoder.summary()

            decode_inputs = Input(shape=(self.bottleneck_dim,), name='decoder_input')

            if len(self.intermediate_dims) > 0:
                interms2 = []
                interms2.append(Dense(self.intermediate_dims[0], activation=intermediate_activation, kernel_initializer=kinit, bias_initializer=binit, activity_regularizer=reg)(decode_inputs))
                for i,dim in enumerate(intermediate_dims[1:]):
                    interms2.append(Dense(dim, activation=intermediate_activation, kernel_initializer=kinit, bias_initializer=binit, activity_regularizer=reg)(interms2[i - 1]))
            else:
                interms2 = [decode_inputs]

            self.outputs = Dense(self.output_dim, activation=output_activation)(interms2[-1])

            self.decoder = Model(decode_inputs, self.outputs, name='decoder')
            self.decoder.summary()

            self.outputs = self.decoder(self.encoder(self.inputs))
            self.autoencoder = Model(self.inputs, self.outputs, name='vae')
            # self.autoencoder.compile('adam', 'mse', metrics=['accuracy'])

            # self.loss = get_keras_loss(loss_function)(self.inputs, self.outputs)
            # self.autoencoder.add_loss(self.loss)
            self.loss = loss_function
            self.autoencoder.summary()

            
            self.build_params['kinit'] = kinit
            self.build_params['binit'] = binit
            self.build_params['n_samples'] = self.n_samples
            self.build_params['intermediate_activation'] = intermediate_activation
            self.build_params['encoded_activation'] = encoded_activation
            self.build_params['output_activation'] = output_activation
            self.build_params['loss'] = self.loss
            self.build_params['shape'] = tuple([self.input_dim,] + self.intermediate_dims + [self.bottleneck_dim,] + list(np.flip(self.intermediate_dims,0)) + [self.output_dim])

            self.BUILT = True

            return self.autoencoder

        def train(
            self,
            input_data=None,
            output_data=None,
            validation_split=0.25,
            epochs=1,
            verbose=True,
            batch_size=1,
            optimizer='adam',
            shuffle=True,
            learning_rate=0.0003
        ):
            assert self.BUILT

            if input_data is None:
                input_data = self.normalized
            
            if output_data is None:
                output_data = self.normalized

            n_samples = len(input_data)

            if optimizer in ['adagrad', 'adadelta']:
                learning_rate = 1.0
            self.autoencoder.compile(getattr(keras.optimizers, optimizer)(lr=learning_rate), self.loss)
            self.history = self.autoencoder.fit(
                input_data,
                output_data,
                epochs=epochs,
                # batch_size=batch_size, 
                verbose=verbose,
                validation_split=validation_split,
                steps_per_epoch=int(np.ceil(n_samples*(1-validation_split)/batch_size)),
                validation_steps=int(np.ceil(n_samples*validation_split/batch_size)),
                shuffle=shuffle
                )

            self.reps = self.encoder.predict(input_data)
            self.train_params['validation_split'] = validation_split
            self.train_params['epochs'] = epochs
            self.train_params['batch_size'] = batch_size
            self.train_params['optimizer'] = optimizer
            self.train_params['learning_rate'] = learning_rate
            self.train_params['n_samples'] = n_samples
            print self.train_params
            self.TRAINED = True

        def plot_training_history(
            self,
        ):
            for key,value in b.history.history.items():
                plt.plot(value, label=key)
            plt.legend(); plt.show()

        def plot_rep(
            self,
            i,
            *args,
            **kwargs
        ):
            assert i < self.reps.shape[0]
            n = np.sqrt(self.reps.shape[1])
            assert np.ceil(n) == np.floor(n) ## assert IS SQUARE (lol)
            n = int(n)
            plt.pcolormesh(self.reps[i].reshape(n,n), *args, **kwargs)
            plt.colorbar()
            plt.show()        

        def plot_rep_distributions(
            self,
            hide_zeros=False
        ):
            for i,dist in enumerate(self.reps.T):
                if hide_zeros and all(np.isclose(dist,0)):
                    continue
                plt.hist(dist, range=(self.reps.min(), self.reps.max()), bins=40, histtype='step', label="param " + str(i))
            plt.legend()
            plt.show()
            
        def load(
            self,
            filename=None,
        ):
            assert self.BUILT
            base_filename = self.base_fname()
            if os.path.exists(base_filename):
                self.autoencoder.load_weights(base_filename)
                self.reps = self.encoder.predict(self.normalized)
                return True
            print "no weights found with filename " + base_filename
            return False

        def save(
            self, 
            filename=None
        ):
            assert self.BUILT
            assert self.TRAINED
            base_filename = filename or self.base_fname()
            if not os.path.exists(base_filename):
                self.autoencoder.save_weights(base_filename)
                return True
            print "error: identically named file already exists at " + base_filename
            return False
            
        def base_fname(
            self,
        ):
            return os.path.join(self.path, self.name + "_weights.h5")

        def compare_features(
            self,
            bins=30,
            feature_key=None,
            max_features=20,
            cols=4,
        ):
            # assert self.TRAINED
            feature_key = feature_key or "event_feature"
            labels,true = self.labels[feature_key + "_names"], self.data[feature_key + "_data"].T
            pred = self.reconstruct_dataset(self.autoencoder.predict(self.normalized))[feature_key + "_data"].T

            # print labels
            # print true.shape
            # print pred.shape

            n = min(len(true), max_features)
            rows = self.rows(cols, n)

            for i in range(n):
                gmax, gmin = max(true[i].max(), pred[i].max()), min(true[i].min(), pred[i].min())
                plt.subplot(rows, cols, i + 1)
                plt.hist(true[i], bins=bins, range=(gmin,gmax), histtype="step", label=labels[i] + " " + labels[i] + " true")
                plt.hist(pred[i], bins=bins, range=(gmin,gmax), histtype="step", label=labels[i] + " " + labels[i] + " pred")
                plt.legend()
            plt.show()

        def rows(self, cols,n):
            return n/cols + bool(n%cols)

    def sc2(data,alpha=0.5, size=0.1, caxis=2):
        d = data.T
        assert(4 > data.shape[1] > 1)
        ncaxes = [i for i in range(d.shape[0]) if i != caxis or d.shape[0] < 3]
        plt.scatter(*[d[i] for i in ncaxes], s=size, c=None if d.shape[0] < 3 else d[caxis], alpha=alpha)
        plt.show()

    def sc3(data):
        d = data.T
        assert(5 > data.shape[1] > 2)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(d[0], d[1], d[2], c=None if d.shape[0] < 4 else d[3], alpha=0.2)
        plt.show()

    # if __name__=="__main__":
        # samplepath = "../data/SVJfull/*_data.h5"
        # filelist = glob.glob(samplepath)

        # # b = base_autoencoder()

        # b = basic_autoencoder("test1", path="")
        # for f in filelist:
        #     b.add_sample(f)
        # # b.add_sample("../data/output/smallsample_data.h5")
        # b.process_samples()
        # b.build(9, loss_function="mse")
        # if not b.load():
        #     b.train(epochs=10, batch_size=500, validation_split=0.3, learning_rate=0.01, optimizer="adam")
        #     b.save()
        # else:
        #     b.load()
        # b.compare_features()
        # b.plot_training_history()
        # b.plot_rep_distributions(hide_zeros=False)
        # print ret.values()[1].shape

    # class mldata(np.ndarray):
    #     def __new__(cls, input_array, input_dict): #, info=None
    #         # Input array is an already formed ndarray instance
    #         # We first cast to be our class type
    #         obj = np.asarray(input_array).view(cls)

    #         obj.input_dict = input_dict
    #         obj.nmax = obj.max(axis=0)
    #         obj.nmin = obj.min(axis=0)
    #         # add the new attribute to the created instance
    #         # obj.info = info
    #         # Finally, we must return the newly created object:
    #         return obj

    #     def __array_finalize__(self,obj):
    #         # reset the attribute from passed original object
    #         # self.info = getattr(obj, 'info', None)
    #         self.nmax = getattr(obj, 'nmax')
    #         self.nmin = getattr(obj, 'nmin')
    #         self.input_dict = getattr(obj, 'input_dict')
    #         # We do not need to return anything

    #     def normalize(self):
    #         return (self - self.nmin)/(self.nmax - self.nmin)

    #     def denormalize(self):
    #         return (self)

    # def autoencode(
    #     h5_pathname,
    #     epochs=10,
    #     batch_size=1,
    #     neck_dim=2,
    #     intermediate_dims=[], 
    #     validation_split=0.3,
    #     load_if_possible=True,
    # ):
    #     sample_size,input_size = data.shape
    #     input_shape = (input_size,)

    #     inputs = Input(shape=input_shape, name='encoder_input')

    #     if len(intermediate_dims) > 0:
    #         interms1 = []
    #         interms1.append(Dense(intermediate_dims[0], activation='relu')(inputs))
    #         for i,dim in enumerate(intermediate_dims[1:]):
    #             interms1.append(Dense(dim, activation='relu')(interms1[i - 1]))
    #     else:
    #         interms1 = [inputs]

    #     encoded = Dense(neck_dim, activation='relu')(interms1[-1])

    #     encoder = Model(inputs, encoded, name='encoder')
    #     encoder.summary()

    #     decode_inputs = Input(shape=(neck_dim,), name='decoder_input')

    #     if len(intermediate_dims) > 0:
    #         interms2 = []
    #         interms2.append(Dense(intermediate_dims[0], activation='relu')(decode_inputs))
    #         for i,dim in enumerate(intermediate_dims[1:]):
    #             interms2.append(Dense(dim, activation='relu')(interms2[i - 1]))
    #     else:
    #         interms2 = [decode_inputs]

    #     outputs = Dense(input_size, activation='tanh')(interms2[-1])

    #     decoder = Model(decode_inputs, outputs, name='decoder')
    #     decoder.summary()

    #     outputs = decoder(encoder(inputs))
    #     autoencoder = Model(inputs, outputs, name='vae')
    #     autoencoder.summary()

    #     loss = keras.losses.mean_squared_error(
    #         inputs,
    #         outputs,
    #     )

    #     autoencoder.add_loss(loss)
    #     autoencoder.compile(optimizer='adam')

    #     npath = '/{}/model.h5'.format(name)
    #     npath = os.path.abspath(os.curdir) + npath
    #     if os.path.exists(npath) and load_if_possible:
    #         print("loading weights from npath: " + npath)
    #         autoencoder.load_weights(npath)
    #     else:
    #         autoencoder.fit(
    #             data,
    #             epochs=epochs,
    #             # batch_size=batch_size, 
    #             verbose=True,
    #             validation_split=validation_split,
    #             steps_per_epoch=int(np.ceil(sample_size*(1-validation_split)/batch_size)),
    #             validation_steps=int(np.ceil(sample_size*validation_split/batch_size))
    #             )
    #         autoencoder.save_weights(npath)

    #     return locals()
