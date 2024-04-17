import pandas as pd
import os

def rem_stat_obj(group):
    x_range = group['X'].max() - group['X'].min()
    y_range = group['Y'].max() - group['Y'].min()
    return x_range > 1 or y_range > 1

df = pd.read_csv('data.csv')
df = df.dropna()
df = df[~df['OBJECT_TYPE'].isin(['AV', 'AGENT'])]
df = df.groupby('TRACK_ID').filter(lambda x: len(x) > 10)
df = df.groupby('TRACK_ID').filter(rem_stat_obj)
df.sort_values(by=['TRACK_ID', 'TIMESTAMP'], inplace=True)
df.to_csv('result.csv', index=False)