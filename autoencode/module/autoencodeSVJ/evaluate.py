import utils
import trainer
import numpy as np
import tensorflow as tf
import os
import models

class ae_evaluation:
    
    def __init__(
        self,
        name,
        qcd_path=None,
        signal_path=None,
        custom_objects={},
    ):
        self.name = utils.summary_by_name(name)
        self.d = utils.load_summary(self.name)

        if qcd_path is None:
            if 'qcd_path' in self.d:
                qcd_path = self.d['qcd_path']
            else:
                raise AttributeError("No QCD path found; please specify!")

        if signal_path is None:
            if 'signal_path' in self.d:
                signal_path = self.d['signal_path']
            else:
                raise AttributeError("No signal path found; please specify!")
        
        self.qcd_path = qcd_path
        self.signal_path = signal_path
                
        self.hlf = self.d['hlf']
        self.eflow = self.d['eflow']
        self.eflow_base = self.d['eflow_base']
        #     signal_path = "data/signal/base_{}/*.h5".format(eflow_base)
        #     qcd_path = "data/background/base_{}/*.h5".format(eflow_base)

        (self.signal,
         self.signal_jets,
         self.signal_event,
         self.signal_flavor) = utils.load_all_data(
            self.signal_path,
            "signal", include_hlf=self.hlf, include_eflow=self.eflow
        )

        (self.qcd,
         self.qcd_jets,
         self.qcd_event,
         self.qcd_flavor) = utils.load_all_data(
            self.qcd_path, 
            "qcd background", include_hlf=self.hlf, include_eflow=self.eflow
        )

        self.seed = self.d['seed']

        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

        self.target_dim = self.d['target_dim']
        self.input_dim = len(self.signal.columns)
        self.test_split = self.d['test_split']
        self.val_split = self.d['val_split']
        self.filename = self.d['filename']
        self.filepath = self.d['filepath']

        if not os.path.exists(self.filepath + ".h5"):
            self.filepath = utils.path_in_repo(self.filepath + ".h5")
            print self.filepath
            if self.filepath is None:
                raise AttributeError("filepath does not exist with spec {}".format(self.d['filepath']))
            else:
                if self.filepath.endswith(".h5"):
                    self.filepath.rstrip(".h5")

        self.norm_args = {
            "norm_type": str(self.d["norm_type"])
        }
        

        self.all_train, self.test = self.qcd.split_by_event(test_fraction=self.test_split, random_state=self.seed, n_skip=len(self.qcd_jets))
        # self.all_train, self.test = self.qcd.train_test_split(self.test_split, self.seed)
        self.train, self.val = self.all_train.train_test_split(self.val_split, self.seed)

        self.train_norm = self.train.norm(out_name="qcd train norm", **self.norm_args)
        self.val_norm = self.train.norm(self.val, out_name="qcd val norm", **self.norm_args)

        self.test_norm = self.test.norm(out_name="qcd test norm", **self.norm_args)
        self.signal_norm = self.signal.norm(out_name="signal norm", **self.norm_args)

        self.train.name = "qcd training data"
        self.test.name = "qcd test data"
        self.val.name = "qcd validation data"
        
        self.custom_objects = custom_objects
        

        self.instance = trainer.trainer(self.filepath)

        self.ae = self.instance.load_model(custom_objects=self.custom_objects)
        
        [self.qcd_err, self.signal_err], [self.qcd_recon, self.signal_recon] = utils.get_recon_errors([self.test_norm, self.signal_norm], self.ae)

        self.qcd_reps = utils.data_table(self.ae.layers[1].predict(self.test_norm.data), name='background reps')
        self.signal_reps = utils.data_table(self.ae.layers[1].predict(self.signal_norm.data), name='signal reps')

        self.qcd_err_jets = [utils.data_table(self.qcd_err.loc[self.qcd_err.index % 2 == i], name=self.qcd_err.name + " jet " + str(i)) for i in range(2)]
        self.signal_err_jets = [utils.data_table(self.signal_err.loc[self.signal_err.index % 2 == i], name=self.signal_err.name + " jet " + str(i)) for i in range(2)]

        self.test_flavor = self.qcd_flavor.iloc[self.test.index]

    def split_my_jets(
        self,
        test_vec,
        signal_vec, 
        split_by_leading_jet,
        split_by_flavor,
        include_names=False,
    ):
        if split_by_flavor and split_by_leading_jet:
            raise AttributeError("Cannot split by both Flavor and leading/subleading jets (too messy of a plot)")

        signal_out = signal_vec
        qcd_out = [test_vec]
        flag = 0

        if split_by_flavor:
            qcd_out = utils.jet_flavor_split(test_vec, self.test_flavor)
            if include_names:
                for i in range(len(qcd_out)):
                    qcd_out[i].name = qcd_out[i].name + ", " + test_vec.name 

        if split_by_leading_jet:
            j1s, j2s = map(utils.data_table, [signal_vec.iloc[0::2], signal_vec.iloc[1::2]])
            j1s.name = 'leading signal jet'
            j2s.name = 'subleading signal jet'
            if include_names:
                j1s.name += ", " + signal_vec.name
                j2s.name += ", " + signal_vec.name
            signal_out = [j1s, j2s]
            qcd_out = test_vec
            flag = 1

        return signal_out, qcd_out, flag

    def retdict(
        self,
        this,
        others,
    ):
        ret = {}
        for elt in [this] + others:
            assert elt.name not in ret
            ret[elt.name] = elt
        return ret

    def recon(
        self,
        show_plot=True,
        signal=True,
        qcd=True,
        pre=True,
        post=True,        
        alpha=1,
        normed=1,
        figname='variable reconstructions',
        figsize=15,
        cols=4,
        split_by_leading_jet=False,
        split_by_flavor=False,
        *args,
        **kwargs
    ):
        assert signal or qcd, "must select one of either 'signal' or 'qcd' distributions to show"
        assert pre or post, "must select one of either 'pre' or 'post' distributions to show"
        
        this_arr, signal_arr = [], []

        signal_pre, qcd_pre, flag_pre = self.split_my_jets(self.test_norm, self.signal_norm, split_by_leading_jet, split_by_flavor, include_names=True)
        signal_post, qcd_post, flag_post = self.split_my_jets(self.qcd_recon, self.signal_recon, split_by_leading_jet, split_by_flavor, include_names=True)

        to_plot = []

        if signal:
            if pre:
                if flag_pre:
                    to_plot += signal_pre
                else:
                    to_plot.append(signal_pre)
            if post:
                if flag_post:
                    to_plot += signal_post
                else:
                    to_plot.append(signal_post)

        if qcd:
            if pre:
                if flag_pre:
                    to_plot.append(qcd_pre)
                else:
                    to_plot += qcd_pre
            if post:
                if flag_post:
                    to_plot.append(qcd_post)            
                else:
                    to_plot += qcd_post

        assert len(to_plot) > 0

        if show_plot:
            to_plot[0].plot(
                to_plot[1:],
                alpha=alpha, normed=normed,
                figname=figname, figsize=figsize,
                cols=cols, *args, **kwargs
            )
            return

        return self.retdict(to_plot[0], to_plot[1:])

    def node_reps(
        self,
        show_plot=True,
        alpha=1,
        normed=1,
        figname='node reps',
        figsize=10,
        figloc='upper right',
        cols=4,
        split_by_leading_jet=False,
        split_by_flavor=False,
        *args,
        **kwargs
    ):
         
        sig, qcd, flag = self.split_my_jets(self.qcd_reps, self.signal_reps, split_by_leading_jet, split_by_flavor)
        
        if flag:
            this, others = qcd, sig
        else:
            this, others = sig, qcd

        if show_plot:             
            this.plot(
                others, alpha=alpha,
                normed=normed, figname=figname, figsize=figsize,
                figloc=figloc, cols=cols, *args, **kwargs
            )
            return 

        return self.retdict(this, others)
        
    def metrics(
        self,
        show_plot=True,
        *args,
        **kwargs
    ):
        if show_plot:
            self.instance.plot_metrics(*args, **kwargs)
            return
        return self.instance.metrics
    
    def error(
        self,
        show_plot=True,
        figsize=15, normed='n', 
        figname='error for eflow variables', 
        yscale='linear', rng=((0, 0.08), (0, 0.3)), 
        split_by_leading_jet=False, split_by_flavor=False,
        figloc="upper right", *args, **kwargs
    ):
        sig, qcd, flag = self.split_my_jets(self.qcd_err, self.signal_err, split_by_leading_jet, split_by_flavor)

        if flag:
            this, others = qcd, sig
        else:
            this, others = sig, qcd

        if show_plot:
            this.plot(
                others, figsize=figsize, normed=normed, 
                figname=figname, 
                yscale=yscale, rng=rng, 
                figloc=figloc, *args, **kwargs
            )
            return
        return self.retdict(this, others)
    
    def roc(
        self,
        show_plot=True,
        metrics=['mae', 'mse'],
        figsize=8,
        figloc=(0.3, 0.2),
        split_by_leading_jet=False,
        *args,
        **kwargs
    ):

        qcd, signal = self.qcd_err, self.signal_err
        if split_by_leading_jet:
            signal, qcd, _ = self.split_my_jets(self.qcd_err, self.signal_err, True, False)
            signal += [self.signal_err]
            signal[-1].name = "combined signal error"
        
        if show_plot:
            utils.roc_auc_plot(
                qcd, signal,
                metrics=metrics, figsize=figsize,
                figloc=figloc
            )
            
            return

        return utils.roc_auc_dict(
            qcd, signal,
            metrics=metrics
        )

    def cut_at_threshold(
        self,
        threshold,
        metric="mae"
    ):
        sig = utils.event_error_tags(self.signal_err_jets, threshold, "signal", metric)
        qcd = utils.event_error_tags(self.qcd_err_jets, threshold, "qcd", metric)

        return {"signal": sig, "qcd": qcd}

    def check_cuts(
        self,
        cuts,
    ):
        for k in cuts:
            s = 0
            print k +":"
            for subk in cuts[k]:
                print " -", str(subk) + ":", cuts[k][subk].shape
                s += len(cuts[k][subk])
            print " - size:", s

        print " - og signal size:", len(e.signal)/2 
        print " - og test size:", len(e.test)/2

    def fill_cuts(
        self,
        cuts,
        output_dir=None,
        rng=(0., 3000.),
        bins=50,
        var="MT"
    ):
        import ROOT as rt
        import root_numpy as rtnp

        if output_dir is None:
            output_dir = os.path.abspath(".")

        out_prefix = os.path.join(output_dir, self.filename)

        all_data = {}

        for name,cut in cuts.items():
            out_name = out_prefix + "_" + name + ".root"
            if os.path.exists(out_name):
                raise AttributeError("File at path " + out_name + " already exists!! Choose another.")
            print "saving root file at " + out_name
            f = rt.TFile(out_name, "RECREATE")
            histos = []
            all_data[out_name] = []
            for jet_n, idx  in cut.items():
                hname = name + "_{}_jet".format(jet_n)
                hist = rt.TH1F(hname, hname, bins, *rng)
                
                data = getattr(self, name + "_event").loc[idx][var]
                rtnp.fill_hist(hist, data)
                all_data[out_name].append(np.histogram(data, bins=bins, range=rng))
                histos.append(hist)

            f.Write()
            
        return all_data


eflow_base_lookup = {
    12: 3,
    13: 3,
    35: 4, 
    36: 4, 
}

def ae_train(
    signal_path,
    qcd_path,
    target_dim,
    hlf=True,
    eflow=True,
    version=None,
    seed=None,
    test_split=0.15, 
    val_split=0.15,
    norm_args={
        "norm_type": "StandardScaler"
    },
    train_me=True,
    batch_size=64,
    loss='mse',
    optimizer='adam',
    epochs=100,
    learning_rate=0.0005,
    custom_objects={},
    interm_architecture=(30,30),
    output_data_path=None,
):

    """Training function for basic autoencoder (inputs == outputs). 
    Will create and save a summary file for this training run, with relevant
    training details etc.

    Not super flexible, but gives a good idea of how good your standard AE is.
    """

    if seed is None:
        seed = np.random.randint(0, 99999999)

    # set random seed
    np.random.seed(seed)
    tf.set_random_seed(seed)

    if output_data_path is None:
        output_data_path = os.path.join(utils.get_repo_info()['head'], "autoencode/data/training_runs")

    # get all our data
    (signal,
     signal_jets,
     signal_event,
     signal_flavor) = utils.load_all_data(
        signal_path,
        "signal", include_hlf=hlf, include_eflow=eflow
    )

    (qcd,
     qcd_jets,
     qcd_event,
     qcd_flavor) = utils.load_all_data(
        qcd_path, 
        "qcd background", include_hlf=hlf, include_eflow=eflow
    )

    if eflow:
        qcd_eflow = len(filter(lambda x: "eflow" in x, qcd.columns))
        signal_eflow = len(filter(lambda x: "eflow" in x, signal.columns))

        assert qcd_eflow == signal_eflow, 'signal and qcd eflow basis must be the same!!'
        eflow_base = eflow_base_lookup[qcd_eflow]
    else:
        eflow_base = 0

    filename = "{}{}{}_".format('hlf_' if hlf else '', 'eflow{}_'.format(eflow_base) if eflow else '', target_dim)
    
    if version is None:
        existing_ids = map(lambda x: int(os.path.basename(x).rstrip('.summary').split('_')[-1].lstrip('v')), utils.summary_match(filename + "v*"))
        assert len(existing_ids) == len(set(existing_ids)), "no duplicate ids"
        id_set = set(existing_ids)
        this_num = 0
        while this_num in id_set:
            this_num += 1
        
        version = this_num

    filename += "v{}".format(version)

    assert len(utils.summary_match(filename)) == 0, "filename '{}' exists already! Change version id, or leave blank.".format(filename)

    filepath = os.path.join(output_data_path, filename)
    input_dim = len(signal.columns)

    data_args = {
        'target_dim': target_dim,
        'input_dim': input_dim,
        'test_split': test_split,
        'val_split': val_split,
        'hlf': hlf, 
        'eflow': eflow,
        'eflow_base': eflow_base,
        'seed': seed,
        'filename': filename,
        'filepath': filepath,
        'qcd_path': qcd_path,
        'signal_path': signal_path,
        'arch': (input_dim,) + interm_architecture + (target_dim,) + tuple(reversed(interm_architecture)) + (input_dim,)
    }

    all_train, test = qcd.split_by_event(test_fraction=test_split, random_state=seed, n_skip=len(qcd_jets))
    train, val = all_train.train_test_split(val_split, seed)

    train_norm = train.norm(out_name="qcd train norm", **norm_args)
    val_norm = train.norm(val, out_name="qcd val norm", **norm_args)
    
    test_norm = test.norm(out_name="qcd test norm", **norm_args)
    signal_norm = signal.norm(out_name="signal norm", **norm_args)

    train.name = "qcd training data"
    test.name = "qcd test data"
    val.name = "qcd validation data"

    instance = trainer.trainer(filepath)

    aes = models.base_autoencoder()
    aes.add(input_dim)
    for elt in interm_architecture:
        aes.add(elt, activation='relu')
    aes.add(target_dim, activation='relu')
    for elt in reversed(interm_architecture):
        aes.add(elt, activation='relu')
    aes.add(input_dim, activation='linear')

    ae = aes.build()
    ae.summary()
    train_args = {
        'batch_size': batch_size, 
        'loss': loss, 
        'optimizer': optimizer,
        'epochs': epochs,
        'learning_rate': learning_rate,
    }

    print "TRAINING WITH PARAMS >>>"
    for arg in train_args:
        print arg, ":", train_args[arg]

    if train_me:
        ae = instance.train(
            x_train=train_norm.data,
            x_test=val_norm.data,
            y_train=train_norm.data,
            y_test=val_norm.data,
            model=ae,
            force=True,
            use_callbacks=True,
            custom_objects=custom_objects, 
            **train_args
        )
    else:
        ae = instance.load_model(custom_objects=custom_objects)

    [data_err, signal_err], [data_recon, signal_recon] = utils.get_recon_errors([test_norm, signal_norm], ae)
    roc_dict = utils.roc_auc_dict(data_err, signal_err, metrics=['mae', 'mse']).values()[0]
    result_args = dict([(r + '_auc', roc_dict[r]['auc']) for r in roc_dict])

    utils.dump_summary_json(result_args, train_args, data_args, norm_args)

    return locals()


