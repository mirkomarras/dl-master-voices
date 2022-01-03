import os
import re
import glob
import json
import numpy as np
import pandas as pd

from argparse import Namespace

def results_df(dirname, pattern=None):
    """Return DataFrame that summarizes a series of optimization runs
    """
    df = pd.DataFrame(columns=('run_id', 'epsilon', 'steps', 'step_size_override', 'clip_av', 'far1-sv', 'far1-mv', 
                               'nes_n', 'nes_sigma', 'mse', 'max_dist', 'pesq'))
    
    if not os.path.isdir(dirname):
        raise IOError('Directory does not exist...')
    
    for subdir in sorted(os.listdir(dirname)):
        
        if pattern is not None and not re.match(pattern, subdir):
            continue
    
        with open(os.path.join(dirname, subdir, 'params.txt')) as f:
            params = f.read()
            args = eval(eval(params))
        
        with open(os.path.join(dirname, subdir, 'stats.json')) as f:
            train_stats = json.load(f)
        
        df = df.append({
            'run_id' : subdir,
            'epsilon' : args.epsilon,
            'steps': args.n_steps,
            'step_size' : args.step_size_override,
            'mse': np.mean(train_stats['mse']),
            'pesq': np.mean(train_stats['pesq']),
            'max_dist': np.mean(train_stats['max_dist']),
            'nes_n' : args.nes_n,
            'nes_sigma' : args.nes_sigma,
            'clip_av' : args.clip_av,
            'far1-sv': np.mean(train_stats['sv_far1_results']),
            'far1-mv': np.mean(train_stats['mv_far1_results'])
            }, ignore_index=True)
    
    # Drop columns with all NaNs
    df = df.dropna(how='all', axis=1)
    
    # Drop columns with all identical entries
    for col in df.columns:
        if len(df[col].unique()) == 1:
            df = df.drop(columns=col)
    
    return df


def cat_df(column_name, frames):
    """Concatenate dataframes with an extra column that describes each frame (taken from dict key)
    """
    for k, df in frames.items():
        df[column_name] = k

    return pd.concat(frames.values(), ignore_index=True)


def progress(dirname, pattern=None, pad=True):
    """Return optimization progress for a series of optimization runs
    """
    results = {}
    labels = {}
    
    for subdir in sorted(os.listdir(dirname)):
        
        if pattern is not None and not re.match(pattern, subdir):
            continue

        with open(os.path.join(dirname, subdir, 'params.txt')) as f:
            args = eval(eval(f.read()))

        labels[subdir] = {
            'step': args.step_size_override,
            'label': f'{args.step_size_override}'
        }

        results[subdir] = []

        for pfile in glob.glob(os.path.join(dirname, subdir, 'opt_progress*.npz')):
            pdata = {x: y for x, y in np.load(pfile, allow_pickle=True).items()}
            results[subdir].append([x['f'] for x in pdata['mv_far1_results']])
            
    # Pad results 
    if pad:
        for k, v in results.items():            
            max_len = max(len(x) for x in v)
            for items in v:
                if len(items) < max_len:
                    items.extend([items[-1]] * (max_len - len(items)))
    
    return results, labels