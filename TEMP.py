
import glob
import tensorflow as tf
import pandas as pd
import autoencodeSVJ.utils as utils
import os
import autoencodeSVJ.evaluate as ev

def update_all_signal_evals():
    top = utils.summary().cfilter(['*auc*', 'target_dim', 'filename', 'signal_path', 'batch*', 'learning_rate']).sort_values('mae_auc')[::-1]
    eflow_base = 3
    
    to_add = ['autoencode/data/aucs/{}'.format(f) for f in top.filename.values if not os.path.exists('autoencode/data/aucs/{}'.format(f))]
    to_update = [f for f in glob.glob('autoencode/data/aucs/*') if f.split('/')[-1] not in top.filename.values]
    
    if len(to_add) + len(to_update) > 0:
        
        d = ev.data_holder(
                qcd='data/background/base_3/*.h5',
                **{os.path.basename(p): '{}/base_{}/*.h5'.format(p, eflow_base) for p in glob.glob('data/all_signals/*')}
            )
        d.load()
        
        if len(to_add) > 0:
            print('filelist to add: {}'.format(to_add))

        for path in to_add:
            name = path.split('/')[-1]
            tf.reset_default_graph()            
            a = ev.auc_getter(name, times=True)
            norm, err, recon = a.get_errs_recon(d)
            aucs = a.get_aucs(err)
            fmt = a.auc_metric(aucs)
            fmt.to_csv(path)
            
        if len(to_update) > 0:
            print('filelist to update: {}'.format(to_update))
            
        for path in to_update:

            name = path.split('/')[-1]
            tf.reset_default_graph()            
            a = ev.auc_getter(name, times=True)
            a.update_event_range(d, percentile_n=1)
            norm, err, recon = a.get_errs_recon(d)
            aucs = a.get_aucs(err)
            fmt = a.auc_metric(aucs)
            fmt.to_csv(path)
            
update_all_signal_evals()

