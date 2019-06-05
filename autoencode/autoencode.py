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

class basic_autoencoder:

    def __init__(
        self,
        name,
        path=None,
    ):
        self.name = name
        self.path = self._smartpath(path or os.path.curdir)
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
        filepath = self._smartpath(file)
        assert os.path.exists(filepath)
        if filepath not in self.samples:
            self.samples[filepath] = h5py.File(filepath)

    @staticmethod
    def add_key(key, keycheck, sfile, d):
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

    def reconstruct_output(
        self,
        output_dataset,
    ):
        

    def build(
        self,
        bottleneck_dim,
        intermediate_dims=[],
        loss_function="mse",
        kinit="random_uniform",
        binit="random_uniform",
        reg=keras.regularizers.l1(10e-8),
        intermediate_activation='relu',
        encoded_activation='relu',
        output_activation='tanh',
    ):

        self.build_training_dataset()

        assert self.BUILT_TRAINING_DATASET

        self.n_samples, self.input_dim = self.train_dataset.shape
        self.output_dim = self.input_dim
        self.intermediate_dims = intermediate_dims
        self.bottleneck_dim = bottleneck_dim
        self.inputs = Input(shape=(self.input_dim,), name='encoder_input')

        if len(intermediate_dims) > 0:
            interms1 = []
            interms1.append(Dense(self.intermediate_dims[0], activation=intermediate_activation, kernel_initializer=kinit, bias_initializer=binit, activity_regularizer=reg)(self.inputs))
            for i,dim in enumerate(self.intermediate_dims[1:]):
                interms1.append(Dense(dim, activation=intermediate_activation, kernel_initializer=kinit, bias_initializer=binit, activity_regularizer=reg)(interms1[i - 1]))
        else:
            interms1 = [self.inputs]

        encoded = Dense(self.bottleneck_dim, activation=encoded_activation, kernel_initializer=kinit, bias_initializer=binit, activity_regularizer=reg)(interms1[-1])

        self.encoder = Model(self.inputs, encoded, name='encoder')
        self.encoder.summary()

        decode_inputs = Input(shape=(self.bottleneck_dim,), name='decoder_input')

        if len(self.intermediate_dims) > 0:
            interms2 = []
            interms2.append(Dense(self.intermediate_dims[0], activation=intermediate_activation, kernel_initializer=kinit, bias_initializer=binit, activity_regularizer=reg)(input_img)(decode_inputs))
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
        validation_split=0.25,
        epochs=1,
        verbose=True,
        batch_size=1,
        optimizer='adam',
        shuffle=True,
        learning_rate=0.005
    ):
        assert self.BUILT

        if optimizer in ['adagrad', 'adadelta']:
            learning_rate = 1.0
        self.autoencoder.compile(getattr(keras.optimizers, optimizer)(lr=learning_rate), self.loss)
        self.history = self.autoencoder.fit(
            self.normalized,
            self.normalized,
            epochs=epochs,
            # batch_size=batch_size, 
            verbose=verbose,
            validation_split=validation_split,
            steps_per_epoch=int(np.ceil(self.n_samples*(1-validation_split)/batch_size)),
            validation_steps=int(np.ceil(self.n_samples*validation_split/batch_size)),
            shuffle=shuffle
            )

        self.reps = self.encoder.predict(self.normalized)
        self.train_params['validation_split'] = validation_split
        self.train_params['epochs'] = epochs
        self.train_params['batch_size'] = batch_size
        self.train_params['optimizer'] = optimizer
        self.train_params['learning_rate'] = learning_rate
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
        assert i < self.normalized.shape[0]
        n = np.sqrt(self.normalized.shape[1])
        assert np.ciel(n) == np.floor(n) ## assert IS SQUARE (lol)
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

    def _smartpath(
        self,
        path,
    ):
        if path.startswith("~/"):
            return path
        return os.path.abspath(path)

if __name__=="__main__":
    samplepath = "../data/first10/first10_data.h5"
    samplepath = "../data/full/0_data.h5"
    b = basic_autoencoder("large", path="")
    b.add_sample("../data/full/0_data.h5")
    b.add_sample("../data/full/2_data.h5")
    b.add_sample("../data/full/5_data.h5")
    b.add_sample("../data/full/6_data.h5")
    b.add_sample("../data/full/7_data.h5")
    b.add_sample("../data/full/8_data.h5")

    # b.add_sample("../data/output/smallsample_data.h5")
    b.process_samples()
    b.build(9, loss_function="binary_crossentropy")
    b.train(epochs=100, batch_size=50, validation_split=0.3, learning_rate=0.01, optimizer="adadelta")
    # b.plot_training_history()
    b.plot_rep_distributions(hide_zeros=False)
    # print ret.values()[1].shape

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
