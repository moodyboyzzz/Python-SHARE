import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')
plt.figure(figsize=(10, 10))
data['TRACK_ID'] = data['TRACK_ID'].apply(lambda x: x[-12:])

for track_id, group in data.groupby('TRACK_ID'):
    plt.plot(group['X'], group['Y'], label=f'{track_id} ({group["OBJECT_TYPE"].iloc[0]})')

plt.title('Визуализация дорожной сцены')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig('scene.png', dpi=200, bbox_inches='tight')