from skeletons import training_skeleton, autoencoder_skeleton
from utils import data_loader, data_table

def grab_data(filename, split):

    loader = data_loader(True)
    loader.add_sample(filename)

    data = loader.make_table()

    train, test = data.train_test_split(split)
    train_norm, test_norm = data.norm(train), data.norm(test) 

    return train_norm, test_norm, data

def shallow(bn, n_features, central_activation='relu'):

    model = autoencoder_skeleton()
    model.add(n_features, 'relu', )
    model.add(bn, central_activation)
    model.add(n_features, 'linear')
    
    return model.build(optimizer='adam', loss='mse')

def medium(bn, n_features, central_activation='relu'):
    model = autoencoder_skeleton()
    model.add(n_features, 'relu')
    model.add((n_features - bn)/2 + bn)
    model.add(bn, central_activation)
    model.add((n_features - bn)/2 + bn)
    model.add(n_features, 'linear')    
    
    return model.build(optimizer='adam', loss='mse')

def deep(bn, n_features, central_activation='relu', depth=100, intermediate_layers=2):
    model = autoencoder_skeleton()
    model.add(n_features, 'relu')
    for i in range(intermediate_layers):
        model.add(depth, 'relu')
    model.add(bn, central_activation)
    for i in range(intermediate_layers):
        model.add(depth, 'relu')
    model.add(n_features, 'linear')

    return model.build(optimizer='adam', loss='mse')

# def rand_search(n):
#     def r(l):
#         return l[random.randint(0, len(l) - 1)]
#     optimizers = ['adam', 'adagrad']
#     losses = ['mean_squared_error','mean_absolute_error']
#     activations = ['linear', 'tanh']
#     bottlenecks = [4,5,6,7,8]
#     batches = [10,50,100]
#     epochs = [40]
#     pdict = odict()
    
#     import json

#     for i in range(n):

#         opt = r(optimizers)
#         loss = r(losses)
#         act = r(activations)
#         bn = r(bottlenecks)
#         batch = r(batches)
#         epoch = r(epochs)

#         strname = "{}-{}-{}-{}-{}-{}".format(bn, opt, loss, act, batch, epoch)

#         if strname in pdict:
#             continue

#         pdict[strname] = {}

#         pdict[strname]['params'] = {
#             'optimizer': opt,
#             'loss': loss,
#             'activation': act,
#             'bottleneck': bn,
#             'epochs': epoch,
#             'batchsize': batch,
#         }

#         print strname

#         pdict[strname]['out'] = basic_run(bn, opt, loss, act, epoch, batch)

#         if os.path.exists("pdict.json"):
#             old = json.load(open("pdict.json"))
#             pdict.update(old)
#         json.dump(pdict, open("pdict.json", 'w+'))

#         print pdict.keys()
