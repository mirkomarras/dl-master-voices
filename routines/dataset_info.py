import json
import pandas as pd

with open('config/datasets.json') as f:
    data = json.load(f)

for k, config in data.items():
    print(f'[{k}]')
    seen_users = set()
    
    for split in ('train', 'test'):

        df_train = pd.read_csv(config[split])
        users = set(df_train['user_id'].unique())
        
        # test for overlap
        if seen_users.intersection(users):
            print(f'  WARNING: Overlap detected in {split}! {len(seen_users.intersection(users))} users already seen in previous splits')
        seen_users = seen_users.union(users)

        tr_unique = len(users)
        n_samples = len(df_train) / tr_unique
        tr_m = (df_train['gender'] == 'm').sum() / n_samples
        tr_f = (df_train['gender'] == 'f').sum() / n_samples
        print(f'  {split:<10s}: {tr_unique} users [m={tr_m:.0f}, f={tr_f:.0f}]')
    print('')
