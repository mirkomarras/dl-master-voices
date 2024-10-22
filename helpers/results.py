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
    if len(df) > 1:
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


def progress(dirname, pattern=None, pad=True, gender='f'):
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
            results[subdir].append([x[gender] for x in pdata['mv_far1_results']])
            
    # Pad results 
    if pad:
        for k, v in results.items():            
            max_len = max(len(x) for x in v)
            for items in v:
                if len(items) < max_len:
                    items.extend([items[-1]] * (max_len - len(items)))
    
    return results, labels


def scatter(dirname, measurement='training', play=False, gender_index=1):
    """Return DataFrame that summarizes a series of optimization runs
    """
    df = pd.DataFrame(columns=('run_id', 'epsilon', 'steps', 'step_size_override', 'clip_av', 'far1-sv', 'far1-mv', 
                               'nes_n', 'nes_sigma', 'mse', 'max_dist', 'pesq'))
    
    if not os.path.isdir(dirname):
        raise IOError(f'Directory does not exist ({dirname})...')
        
    with open(os.path.join(dirname, 'params.txt')) as f:
        params = f.read()
        args = eval(eval(params))
    
    with open(os.path.join(dirname, 'stats.json')) as f:
        train_stats = json.load(f)
    
    # df = df.append({
    #     'run_id' : subdir,
    #     'epsilon' : args.epsilon,
    #     'steps': args.n_steps,
    #     'step_size' : args.step_size_override,
    #     'mse': np.mean(train_stats['mse']),
    #     'pesq': np.mean(train_stats['pesq']),
    #     'max_dist': np.mean(train_stats['max_dist']),
    #     'nes_n' : args.nes_n,
    #     'nes_sigma' : args.nes_sigma,
    #     'clip_av' : args.clip_av,
    #     'far1-sv': np.mean(),
    #     'far1-mv': np.mean()
    #     }, ignore_index=True)

    if measurement == 'training':
        return train_stats['sv_far1_results'], train_stats['mv_far1_results'], np.mean(train_stats['pesq'])

    else:
        pattern = os.path.join(dirname, 'mv', f'*{measurement}-{play:1d}.npz')
        filename = glob.glob(pattern)
        
        if len(filename) == 0:
            raise IOError(f'File not found: {pattern}')
        elif len(filename) == 1:
            filename = filename[0]
        else:            
            raise IOError(f'Found multiple result files matching the current pattern (*{measurement}-{play:1d}.npz)')

        pdata = {x: y for x, y in np.load(filename, allow_pickle=True).items()}
        ir_mv = pdata['results'].item()['gnds'][:, gender_index]
        # pdata['results'].item()['imps'].mean(1)

        # print(np.mean(ir_mv), 

        pattern = os.path.join(dirname, 'sv', f'*{measurement}-{play:1d}.npz')
        filename = glob.glob(pattern)[0]
        pdata = {x: y for x, y in np.load(filename, allow_pickle=True).items()}
        # ir_sv = pdata['results'].item()['imps'].mean(1)
        ir_sv = pdata['results'].item()['gnds'][:, gender_index]

        return ir_sv, ir_mv, np.mean(train_stats['pesq'])


def transferability(dirname, versions=(0,3), gender_index=1, play=False, population='interspeech', policy='avg-far1'):
    # data/results/transfer/vggvox_v000_pgd_wave_f/v000/mv/
    # mv_test_population_interspeech_1000u_10s-resnet34_v000-avg-far1-0.npz
    if population == 'interspeech':
        FN_PREFIX = 'mv_test_population_interspeech_1000u_10s'
    elif population == 'libri':
        FN_PREFIX = 'mv_test_population_libri_100u_10s'
    else:
        raise ValueError('Unknown population!')
    arch = ('resnet50', 'thin_resnet', 'vggvox', 'xvector')
    arch_short = ('R50', 'TR', 'V', 'X')
    encoders = [f'{se}_v000' for se in arch]

    df = pd.DataFrame()

    for target_dir in sorted(os.listdir(dirname)):

        new_row = {
            # 'dirname': target_dir, 
            'target': target_dir.split('_')[0]
            }

        for version in versions:

            prefix = os.path.join(dirname, target_dir, f'v{version:03d}')

            if not os.path.exists(os.path.join(prefix, 'params.txt')):
                continue

            with open(os.path.join(prefix, 'params.txt')) as f:
                params = f.read()
                args = eval(eval(params))

            for i, enc in enumerate(encoders):

                # seed voices
                try:
                    filename = os.path.join(prefix, 'sv', f'{FN_PREFIX}-{enc}-{policy}-{play:1d}.npz')
                    pdata = {x: y for x, y in np.load(filename, allow_pickle=True).items()}
                    ir_per_gender = pdata['results'].item()['gnds'].mean(0)

                    new_row[f'sv/{arch_short[i]}'] = 100 * ir_per_gender[gender_index] 
                except:
                    new_row[f'sv/{arch_short[i]}'] = np.nan

                # master voices
                try:
                    filename = os.path.join(prefix, 'mv', f'{FN_PREFIX}-{enc}-{policy}-{play:1d}.npz')
                    pdata = {x: y for x, y in np.load(filename, allow_pickle=True).items()}
                    ir_per_gender = pdata['results'].item()['gnds'].mean(0)

                    new_row[f'mv({args.step_size_override})/{arch_short[i]}'] = 100 * ir_per_gender[gender_index] 
                except:
                    new_row[f'mv({args.step_size_override})/{arch_short[i]}'] = np.nan
                
        df = df.append(new_row, ignore_index=True)
        # df.loc[target_dir.split('_')[0]] = new_row

    df = df.set_index('target')
    df.columns = pd.MultiIndex.from_tuples([c.split('/') for c in df.columns])

    return df
