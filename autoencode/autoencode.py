import numpy as np 
import keras
import os

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

    def build( 
        self,
        input_dim,
        bottleneck_dim,
        output_dim=None,
        intermediate_dims=[],
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        self.intermediate_dims = intermediate_dims
        self.bottleneck_dim = bottleneck_dim
        self.inputs = keras.layers.Input(shape=(self.input_dim,), name='encoder_input')

        if len(intermediate_dims) > 0:
            interms1 = []
            interms1.append(keras.layers.Dense(self.intermediate_dims[0], activation='relu')(self.inputs))
            for i,dim in enumerate(self.intermediate_dims[1:]):
                interms1.append(keras.layers.Dense(dim, activation='relu')(interms1[i - 1]))
        else:
            interms1 = [self.inputs]

        encoded = keras.layers.Dense(self.bottleneck_dim, activation='relu')(interms1[-1])

        self.encoder = keras.models.Model(self.inputs, encoded, name='encoder')
        self.encoder.summary()

        decode_inputs = keras.layers.Input(shape=(self.bottleneck_dim,), name='decoder_input')

        if len(self.intermediate_dims) > 0:
            interms2 = []
            interms2.append(keras.layers.Dense(self.intermediate_dims[0], activation='relu')(decode_inputs))
            for i,dim in enumerate(intermediate_dims[1:]):
                interms2.append(keras.layers.Dense(dim, activation='relu')(interms2[i - 1]))
        else:
            interms2 = [decode_inputs]

        self.outputs = keras.layers.Dense(self.output_dim, activation='tanh')(interms2[-1])

        self.decoder = keras.models.Model(decode_inputs, self.outputs, name='decoder')
        self.decoder.summary()

        self.outputs = self.decoder(self.encoder(self.inputs))
        self.autoencoder = keras.models.Model(self.inputs, self.outputs, name='vae')
        self.autoencoder.summary()

    def _smartpath(
        self,
        path,
    ):
        if path.startswith("~/"):
            return path
        return os.path.abspath(path)

b = basic_autoencoder("test")
b.build(10, 2)


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

#     inputs = keras.layers.Input(shape=input_shape, name='encoder_input')

#     if len(intermediate_dims) > 0:
#         interms1 = []
#         interms1.append(keras.layers.Dense(intermediate_dims[0], activation='relu')(inputs))
#         for i,dim in enumerate(intermediate_dims[1:]):
#             interms1.append(keras.layers.Dense(dim, activation='relu')(interms1[i - 1]))
#     else:
#         interms1 = [inputs]

#     encoded = keras.layers.Dense(neck_dim, activation='relu')(interms1[-1])

#     encoder = keras.models.Model(inputs, encoded, name='encoder')
#     encoder.summary()

#     decode_inputs = keras.layers.Input(shape=(neck_dim,), name='decoder_input')

#     if len(intermediate_dims) > 0:
#         interms2 = []
#         interms2.append(keras.layers.Dense(intermediate_dims[0], activation='relu')(decode_inputs))
#         for i,dim in enumerate(intermediate_dims[1:]):
#             interms2.append(keras.layers.Dense(dim, activation='relu')(interms2[i - 1]))
#     else:
#         interms2 = [decode_inputs]

#     outputs = keras.layers.Dense(input_size, activation='tanh')(interms2[-1])

#     decoder = keras.models.Model(decode_inputs, outputs, name='decoder')
#     decoder.summary()

#     outputs = decoder(encoder(inputs))
#     autoencoder = keras.models.Model(inputs, outputs, name='vae')
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
