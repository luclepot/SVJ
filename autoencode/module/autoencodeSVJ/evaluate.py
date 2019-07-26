import utils
import trainer
import numpy as np
import tensorflow as tf

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

        self.norm_args = {
            "norm_type": str(self.d["norm_type"])
        }
        
        
        self.all_train, self.test = self.qcd.train_test_split(self.test_split, self.seed)
        self.train, self.val = self.all_train.train_test_split(self.val_split, self.seed)

        self.train_norm = self.train.norm(out_name="qcd train norm", **self.norm_args)
        self.val_norm = self.train.norm(self.val, out_name="qcd val norm", **self.norm_args)

        self.test_norm = self.test.norm(out_name="qcd test norm", **self.norm_args)
        self.signal_norm = self.signal.norm(out_name="signal norm", **self.norm_args)

        self.train.name = "qcd training data"
        self.test.name = "qcd test data"
        self.val.name = "qcd validation data"
        
        self.custom_objects = custom_objects
        
        self.instance = trainer.trainer(self.filename)
        self.ae = self.instance.load_model(custom_objects=self.custom_objects)
        
        [self.qcd_err, self.signal_err], [self.qcd_recon, self.signal_recon] = utils.get_recon_errors([self.test_norm, self.signal_norm], self.ae)

        self.qcd_reps = utils.data_table(self.ae.layers[1].predict(self.test_norm.data), name='background reps')
        self.signal_reps = utils.data_table(self.ae.layers[1].predict(self.signal_norm.data), name='signal reps')
        
    def node_reps(
        self,
        show_plot=True,
        alpha=1,
        normed=1,
        figname='node reps',
        figsize=10,
        figloc='upper right',
        cols=4,
        *args,
        **kwargs
    ):
        if show_plot:
            self.qcd_reps.plot(
                self.signal_reps, alpha=alpha,
                normed=normed, figname=figname, figsize=figsize,
                figloc=figloc, cols=cols, *args, **kwargs
            )
            return 
        return {'qcd': self.qcd_reps, 'signal': self.signal_reps}
        
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
        figloc="upper right", *args, **kwargs
    ):
        if show_plot:
            self.qcd_err.plot(
                self.signal_err, figsize=figsize, normed=normed, 
                figname=figname, 
                yscale=yscale, rng=rng, 
                figloc=figloc, *args, **kwargs
            )
            return
        return {'qcd': self.qcd_err, 'signal': self.signal_err}
    
    def roc(
        self,
        show_plot=True,
        metrics=['mae', 'mse'],
        figsize=8,
        figname=None,
        figloc=(0.3, 0.2),
        *args,
        **kwargs
    ):
        
        if show_plot:
            if figname is None:
                figname = 'ROC (hlf & eflow with $d\leq' + str(self.eflow_base) + '$)',

            utils.roc_auc_plot(
                self.qcd_err, self.signal_err,
                metrics=metrics, figsize=figsize,
                figname=figname, figloc=figloc
            )
            
            return

        roc_dict = utils.roc_auc_dict(
            self.qcd_err, self.signal_err,
            metrics=metrics
        ).values()[0]

        result_args = dict([(r + '_auc', roc_dict[r]['auc']) for r in roc_dict])
        
        return result_args
        