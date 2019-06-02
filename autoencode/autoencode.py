import numpy as np 
from keras.layers import Input, Dense
from keras.models import Model
from keras.losses import get as get_keras_loss
import os
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
        self.TRAINED = False
        self.samples = {}

    def add_sample(
        self,
        file,
    ):
        filepath = self._smartpath(file)
        assert os.path.exists(filepath)
        if filepath not in self.samples:
            self.samples[filepath] = h5py.File(filepath)

    # def process_samples(
    #     self,
    # ):



    def build( 
        self,
        input_dim,
        bottleneck_dim,
        output_dim=None,
        intermediate_dims=[],
        loss_function="mse",
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
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
        self.encoder.summary()

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
        self.decoder.summary()

        self.outputs = self.decoder(self.encoder(self.inputs))
        self.autoencoder = Model(self.inputs, self.outputs, name='vae')
        self.autoencoder.summary()

        self.loss = get_keras_loss(loss_function)(self.inputs, self.outputs)

        self.autoencoder.add_loss(self.loss)

        self.BUILT = True

        return self.autoencoder

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
    b.build(10, 5)
    b.add_sample(samplepath)

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
