import numpy as np 
from keras.layers import Input, Dense
from keras.models import Model
import keras.optimizers
from collections import OrderedDict as odict
import os
from operator import mul
import h5py

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

    def build(
        self,
        bottleneck_dim,
        intermediate_dims=[],
        loss_function="mse",
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
            interms1.append(Dense(self.intermediate_dims[0], activation='relu')(self.inputs))
            for i,dim in enumerate(self.intermediate_dims[1:]):
                interms1.append(Dense(dim, activation='relu')(interms1[i - 1]))
        else:
            interms1 = [self.inputs]

        encoded = Dense(self.bottleneck_dim, activation='relu')(interms1[-1])

        self.encoder = Model(self.inputs, encoded, name='encoder')
        # self.encoder.summary()

        decode_inputs = Input(shape=(self.bottleneck_dim,), name='decoder_input')

        if len(self.intermediate_dims) > 0:
            interms2 = []
            interms2.append(Dense(self.intermediate_dims[0], activation='relu')(decode_inputs))
            for i,dim in enumerate(intermediate_dims[1:]):
                interms2.append(Dense(dim, activation='relu')(interms2[i - 1]))
        else:
            interms2 = [decode_inputs]

        self.outputs = Dense(self.output_dim, activation='tanh')(interms2[-1])

        self.decoder = Model(decode_inputs, self.outputs, name='decoder')
        # self.decoder.summary()

        self.outputs = self.decoder(self.encoder(self.inputs))
        self.autoencoder = Model(self.inputs, self.outputs, name='vae')
        # self.autoencoder.compile('adam', 'mse', metrics=['accuracy'])

        # self.loss = get_keras_loss(loss_function)(self.inputs, self.outputs)
        # self.autoencoder.add_loss(self.loss)
        self.loss = loss_function
        self.autoencoder.summary()

        self.BUILT = True

        return self.autoencoder

    def train(
        self,
        validation_split=0.25,
        epochs=1,
        verbose=True,
        batch_size=1,
        optimizer='adam',
        shuffle=True
    ):
        assert self.BUILT

        self.autoencoder.compile(keras.optimizers.SGD(lr=0.01), 'mse')
        self.autoencoder.fit(
            self.normalized,
            self.normalized,
            epochs=epochs,
            # batch_size=batch_size, 
            verbose=verbose,
            validation_split=validation_split,
            steps_per_epoch=int(np.ceil(self.n_samples*(1-validation_split)/batch_size)),
            validation_steps=int(np.ceil(self.n_samples*validation_split/batch_size)),
            shuffle=shuffle,
            )

        self.autoencoder.save_weights(os.path.join(self.path, "{0}_weights.h5".format(self.name)))

    def load(
        self,
        filename=None,
    ):
        assert self.BUILT
        base_filename = filename or os.path.join(self.path, self.name + "_weights.h5")
        raise NotImplementedError

    def save(
        self, 
        filename=None
    ):
        assert self.BUILT
        assert self.TRAINED
        base_filename = filename or os.path.join(self.path, self.name + "_weights.h5")
        raise NotImplementedError

    def _smartpath(
        self,
        path,
    ):
        if path.startswith("~/"):
            return path
        return os.path.abspath(path)

if __name__=="__main__":
    samplepath = "../data/first10/first10_data.h5"
    b = basic_autoencoder("testsample", path="../data/first10/")
    b.add_sample(samplepath)
    b.add_sample("../data/output/smallsample_data.h5")
    b.process_samples()
    b.build(10)
    b.train(epochs=100, batch_size=41, validation_split=0.333333333333333333)
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
