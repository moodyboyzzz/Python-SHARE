import pandas as pd
df = pd.read_csv("118.csv")
count = df['TRACK_ID'].nunique()
df = df.sort_values(by=['TRACK_ID', 'TIMESTAMP'])
df['delta_x'] = df.groupby('TRACK_ID')['X'].diff()
df['delta_y'] = df.groupby('TRACK_ID')['Y'].diff()
df['distance'] = (df['delta_x'] ** 2 + df['delta_y'] ** 2) ** 0.5
df['trajectory_length'] = df.groupby('TRACK_ID')['distance'].cumsum()
max_length = round(df.groupby('TRACK_ID')['trajectory_length'].max().max(),2)
new_df = pd.DataFrame({'task1': [count], 'task2': [max_length]})
new_df.to_csv('seminar02.csv', index=False)