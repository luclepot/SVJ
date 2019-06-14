
def basic_run(bn, optimizer, loss, act, epochs, batch):
    b = data_loader(True)
    b.add_sample("../data/hlfSVJ/full.h5")

    x, x_dict = b.get_dataset('*event_feature_data*')
    x_train, x_test = train_test_split(x, test_size=0.5, random_state=42)
    # x_norm_train, x_minmax_train = (x - x.min())/(x.max() - x.min()), (x.max(), x.min())
    # x_norm_test, x_minmax_test = (x - x.min())/(x.max() - x.min()), (x.max(), x.min())

    x_norm_train, x_minmax_train = b.norm_min_max(x_train, ab=(-1.,1.))
    x_norm_test, x_minmax_test = b.norm_min_max(x_test, ab=(-1.,1.))

    ae = base_autoencoder()
    ae.add(x.shape[1], 'relu')
    ae.add(50, 'relu')
    ae.add(50, 'relu')
    ae.add(bn, 'relu')
    ae.add(50, 'relu')
    ae.add(50, 'relu')
    ae.add(x.shape[1], act)

    encoder,decoder,autoencoder = ae.build(optimizer=optimizer, loss=loss)

    autoencoder.summary()

    history = ae.fit(
        x=x_norm_train,
        y=x_norm_train,
        validation_data=(x_norm_test,x_norm_test),
        batch_size=batch,
        shuffle=False,
        epochs=epochs,
    )

    # comp = data_table(x_norm_train, b.table.headers, "true")
    # comp.plot(
    #     "*",
    #     others=[data_table(autoencoder.predict(x_norm_train), comp.headers, "predicted")],
    #     normed=False,
    #     bins=50
    # )

    # ae.compare_features(x_norm_train, autoencoder.predict(x_norm_train))
    return {
        "history": history.history,
        "true": x_norm_train.tolist(),
        "pred": autoencoder.predict(x_norm_train).tolist(),
    }

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
