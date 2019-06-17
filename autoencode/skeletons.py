import numpy as np 
from keras.layers import Input, Dense
from keras.models import Model, model_from_json
from collections import OrderedDict as odict
import os
import traceback
from datetime import datetime
from utils import logger, smartpath
import h5py

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

class training_skeleton(logger):
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

        self.config_file = smartpath(name)
        self.path = os.path.dirname(self.config_file)
        
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
                "data": ([
                    "metric_names"
                    ], None)
            }
            
            for attr in file_attrs:
                for subattr in file_attrs[attr][0]:
                    setattr(self, subattr, h5_element_wrapper(self.file, attr, subattr, file_attrs[attr][1], overwrite=overwrite))
                setattr(self, attr, self.file[attr])

            for subattr in self.metric_names:
                setattr(self, subattr, h5_element_wrapper(self.file, "metric_names", subattr, None, overwrite=overwrite))
                        
        except Exception as e:
            self.error(traceback.format_exc())
            self._throw(e, unlock=True)

        cdict = self.config.copy()
        tdict = self.training.copy()
        
        if not preexisting:
            cdict = odict()
            cdict['name'] = name
            cdict['trained'] = ''
            cdict['model_json'] = ''
            # cdict['model_weights'] = ''
            tdict['epoch_splits'] = '[]'

        self.metrics = odict()
        self.training.update(tdict)
        self.config.update(cdict)
        self._update_metrics_dict(self.metrics)

    def _update_metrics_dict(
        self, 
        mdict_to_fill
    ):  
        for metric in self.metric_names:
            if hasattr(self, metric):
                mdict_to_fill[metric] = getattr(self, metric).rep

    def _throw(
        self,
        msg,
        exc=AttributeError,
        unlock=False,
    ):
        if unlock:
            self.close()
        self.error(msg)
        raise exc, msg

    def close(
        self,
        unlock=False,
    ):
        try:
            if self.file is not None:
                self.file.close()
                self.file = None
        except:
            pass
        try:
            self._unlock_file(self.config_file)
        except:
            pass

    def __del__(
        self,
    ):
        self.close(unlock=True)

    ### LOCKING

    def _lock_file(
        self,
        fname,
    ):
        self.log("locking file '{}'".format(fname))
        open(self._get_lock(fname), "w+").close()

    def _unlock_file(
        self,
        fname,
    ):
        to_remove = self._get_lock(fname)
        if os.path.exists(to_remove):
            self.log("unlocking file '{}'".format(os.path.basename(fname)))
            os.remove(to_remove)

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
        x_train,
        y_train=None,
        x_test=None,
        y_test=None,
        model=None,
        epochs=10,
        batch_size=32,
        force=False,
        metrics=[],
        loss=None,
        optimizer=None,
    ):
        w_path = self.config_file.replace(".h5", "_weights.h5")
        # if already trained
        if self.config['trained']:
            if self.config['model_json']:
                if model is None:
                    model = model_from_json(self.config['model_json'])
                    model.load_weights(w_path)
                    self.log("using saved model")
                else:
                    if not force:
                        model = model_from_json(self.config['model_json'])
                        model.load_weights(w_path)
                        self.error("IGNORING PASSED PARAMETER 'model'")
                        self.log("using saved model")
                    else:
                        if isinstance(model, str):
                            model = load_model(model)
                        self.log("using model passed as function argument")
            else:
                if model is None:
                    self._throw("no model passed, and saved model not found!")
                if isinstance(model, str):
                    model = load_model(model)
                self.log("using model passed as function argument")
        
        if model is None:
            self._throw("no model passed and no saved model found!!")

        if optimizer is None:
            if hasattr(model, "optimizer"):
                optimizer = model.optimizer
            else:
                optimizer = "adam"
        if loss is None:
            if hasattr(model, "loss"):
                loss = model.loss
            else:
                loss = "mse"

        if metrics is None:
            if hasattr(model, "metrics"):
                metrics = model.metrics
            else:
                metrics = []

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        model.summary()

        start = datetime.now()

        if y_train is None:
            y_train = x_train
        
        if x_test is None:
            x_test = np.zeros((0,) + x_train.shape[1:])

        if y_test is None:
            y_test = x_test.copy()

        assert x_test.shape[0] == y_test.shape[0]
        assert x_train.shape[0] == y_train.shape[0]
        assert x_test.shape[1] == x_train.shape[1]
        assert y_test.shape[1] == y_train.shape[1]


        previous_epochs = eval(self.training['epoch_splits'])
        master_epoch_n = sum(previous_epochs)

        finished_epoch_n = master_epoch_n + epochs
        

        # last_full_epoch = len(self.metrics[-1][(self.metrics[-1] != -1)])
        # self.metrics = self.metrics[:,:start_epoch]
        # for metric,name in zip(self.metrics, self.metric_names):
        #     history[name] = self.metric

        history = odict()

        try:
            for epoch in range(epochs):
                self.log("TRAINING EPOCH {}/{}".format(master_epoch_n, finished_epoch_n))
                nhistory = model.fit(
                    x=x_train,
                    y=y_train,
                    steps_per_epoch=int(np.ceil(len(x_train)/batch_size)),
                    validation_steps=int(np.ceil(len(x_test)/batch_size)),
                    validation_data=[x_test, y_test],
                    initial_epoch=master_epoch_n,
                    epochs=master_epoch_n + 1,
                    verbose=1,
                ).history

                if epoch == 0:
                    for metric in nhistory:
                        history[metric] = []

                for metric in nhistory:
                    history[metric].append([master_epoch_n, nhistory[metric][0]])

                master_epoch_n += 1

        except:
            self.error(traceback.format_exc())
            print history
            if all([len(v) == 0 for v in history]):
                self._throw("quitting")

        n_epochs_finished = min(map(len, history.values()))
        self.log("")
        self.log("trained {} epochs!".format(n_epochs_finished))
        self.log("")

        hvalues = [hv[:n_epochs_finished] for hv in history.values()]
        hkeys = history.keys()

        for key,value in zip(hkeys, hvalues):
            if hasattr(self, key):
                print key, value
                getattr(self, key).update(np.concatenate([getattr(self, key), value]))
            else:
                setattr(self, key, h5_element_wrapper(self.file, "metric_names", key, None, overwrite=True))
                # setattr(self, key, h5_element_wrapper(self.file, "metric_names", key, None, overwrite=True))
                getattr(self, key).update(np.asarray(value))

        self.metric_names.update(np.asarray(list(set(self.metric_names).union(hkeys))))
        self._update_metrics_dict(self.metrics)

        previous_epochs.append(n_epochs_finished)
        finished_epoch_n = sum(previous_epochs)

        end = datetime.now()

        model.save_weights(w_path)

        cdict = self.config.copy()
        cdict['model_json'] = model.to_json()
        # cdict['model_weights'] = w_path
        cdict['trained'] = str(start) + " ::: " + str(end)
        self.config.update(cdict)
        self.log("model saved")

        tdict = self.training.copy()
        tdict['time'] = str(end - start)
        tdict['epochs'] = epochs
        tdict['batch_size'] = batch_size
        tdict['epoch_splits'] = str(previous_epochs)
        self.training.update(tdict)

        # return odel 
        # hkeys = history.keys()
        # self.metric_names.update(hkeys)

        # self.x_train.update(x_train)
        # self.y_train.update(y_train)
        # self.x_test.update(x_test)
        # self.y_test.update(y_test)
        

        # maxlen = min(map(len, history.values()))

        # hvalues = np.asarray([a[:maxlen] for a in history.values()]))
        # hkeys = history.keys()

        # new_metrics = self.metrics.copy()
        # new_keys = self.metric_names.copy()

        # existing = [(i,new_keys.index(hkeys[i])) for i in range(len(hkeys)) if hkeys[i] in new_keys]
        # to_add = [(i,-1) for i in range(len(hkeys)) if hkeys[i] not in new_keys]
        # unused = [(-1,i) for i in range(len(new_keys)) if new_keys[i] not in hkeys]
        


        # for i,key in enumerate(to_add):
            
        #     new_metrics = np.vstack([new_metrics, ])
        # self.metrics.update(hvalues)

        return model

class autoencoder_skeleton(logger):

    def __init__(
        self,
        name="autoencoder",
        verbose=True,
    ):
        logger.__init__(self)
        self._LOG_PREFIX = "autoencoder_skeleton :: "
        self.VERBOSE = verbose
        self.name = name
        self.layers = []

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
        loss='mse',
        metrics=['accuracy']
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
        encoder = Model(inputs, encoded, name='encoder')
        decoder = Model(encoded_input, outputs, name='decoder')
        autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')

        autoencoder.compile(optimizer, loss, metrics=metrics)

        return encoder, decoder, autoencoder

    # def fit(
    #     self,
    #     x,
    #     y,
    #     validation_data,
    #     batch_size=10,
    #     *args,
    #     **kwargs
    # ):

    #     if self.autoencoder is None: 
    #         raise AttributeError, "Model not yet built!"

    #     try:
    #         self.history = self.autoencoder.fit(
    #             x=x,
    #             y=y,
    #             steps_per_epoch=int(np.ceil(len(x)/batch_size)),
    #             validation_steps=int(np.ceil(len(validation_data[0])/batch_size)),
    #             validation_data=validation_data,
    #             *args,
    #             **kwargs
    #         )

    #     except KeyboardInterrupt:
    #         return None

    #     return self.history

    # def save(
    #     self,
    #     path,
    # ):
    #     path = smartpath(path)
    #     path = smartpath(path)
    #     if not os.path.exists(path):
    #         os.mkdirs(path)
    #     to_save = ['autoencoder', 'encoder', 'decoder']
    #     filenames = [os.path.join(path, model + '.h5') for model in to_save]
    #     for filename,typename in zip(filenames, to_save):
    #         if os.path.exists(smartpath(filename)):
    #             raise AttributeError, "Model already exists at file '{}'!!".format(filename)
    #         getattr(self, typename).save(filename)

    # def load(
    #     self,
    #     path,
    # ):
    #     to_load = ['autoencoder', 'encoder', 'decoder']
    #     filenames = [os.path.join(path, model + '.h5') for model in to_load]
    #     for filename,typename in zip(filenames, to_load):
    #         if not os.path.exists(smartpath(filename)):
    #             raise AttributeError, "Model does not exist at file '{}'!!".format(filename)
    #         setattr(self, typename, keras.models.load_model(filename))

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
