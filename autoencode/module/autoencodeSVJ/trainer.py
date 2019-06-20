import numpy as np
from keras.models import Model, model_from_json
from collections import OrderedDict as odict
import os
import traceback
from datetime import datetime
from utils import logger, smartpath
import h5py
import matplotlib.pyplot as plt
import glob

plt.rcParams.update({'font.size': 18})
plt.rcParams['figure.figsize'] = (10,10)

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

        # self._lock_file(self.config_file)

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
            self._throw(e, unlock=False)

        cdict = self.config.copy()
        tdict = self.training.copy()
        
        if not preexisting:
            cdict = odict()
            cdict['name'] = name
            cdict['trained'] = ''
            cdict['model_json'] = ''
            # cdict['model_weights'] = ''
            tdict['batch_size'] = '[]'
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
        self.close()

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

    def load_model(
        self,
        model=None,
        force=False,
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
            
        return model

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
        verbose=1,
        callbacks=None,
    ):
        w_path = self.config_file.replace(".h5", "_weights.h5")

        model = self.load_model(model, force)

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
                    verbose=verbose,
                    callbacks=callbacks
                ).history

                if epoch == 0:
                    for metric in nhistory:
                        history[metric] = []

                for metric in nhistory:
                    history[metric].append([master_epoch_n, nhistory[metric][0]])

                master_epoch_n += 1

        except:
            self.error(traceback.format_exc())
            if all([len(v) == 0 for v in history]):
                self._throw("quitting")

        if len(history.values()) == 0:
            n_epochs_finished = 0
        else:
            n_epochs_finished = min(map(len, history.values()))

        self.log("")
        self.log("trained {} epochs!".format(n_epochs_finished))
        self.log("")

        hvalues = [hv[:n_epochs_finished] for hv in history.values()]
        hkeys = history.keys()

        for key,value in zip(hkeys, hvalues):
            if hasattr(self, key):
                # print(getattr(self, key))
                # print(np.concatenate([getattr(self, key), value]))
                getattr(self, key).update(np.concatenate([getattr(self, key).rep, value]))
            else:
                # print(getattr(self, key))
                # print(np.concatenate([getattr(self, key), value]))
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
        tdict['batch_size'] = str(eval(tdict['batch_size']) + [batch_size,]*n_epochs_finished)
        tdict['epoch_splits'] = str(eval(tdict['epoch_splits']) + previous_epochs)
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

    def plot_metrics(
        self,
        fnmatch_criteria="*loss*",
        yscale=None,
        figsize=(7,5)
    ):
        names = []
        metrics = []
        for mname in self.metrics:
            if glob.fnmatch.fnmatch(mname, fnmatch_criteria):
                names.append(mname)
                metrics.append(self.metrics[mname])
                
        # break concatenated plot arrays into individual components
        plots = []
        for mn,metric in enumerate(metrics):
            splits = [0,] + list(np.where(np.diff(metric[:,0].astype(int)) > 1)[0]) + [len(metric[:,0]),]
            plots.append([])
            for i in range(len(splits[:-1])):
                plots[mn].append(metric[splits[i] + 1:splits[i+1] + 1,:])

        # plot em'

        plt.rcParams['figure.figsize'] = figsize
        plt.rcParams.update({'font.size': 18})
        if yscale is not None:
            plt.yscale(yscale)
        plt.xlabel("epoch number")
        plt.ylabel("metric value" + (" ({}-scaled)".format(yscale) if yscale else ""))
        
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for color,plot,name in zip(colors, plots, names):
            for subplot in plot:
                plt.plot(subplot[:,0], subplot[:,1], c=color, label=name)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = odict(zip(map(str, labels), handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.tight_layout()
        plt.show()

    def remove(
        self,
        sure=False,
    ):
        if not sure:
            self.error("NOT DELETING: run again with keyword 'sure=True' to remove!")
        else:
            for f in [self.config_file, self.config_file.replace(".h5", "_weights.h5")]:
                if os.path.exists(f):
                    os.remove(f)
            self.log("removed associated data files for self!")
