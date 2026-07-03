from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

with open("data.bin", "rb") as f:
    rows = np.fromfile(f, dtype=np.uint64, count=1)[0]
    cols = np.fromfile(f, dtype=np.uint64, count=1)[0]
    data = np.fromfile(f, dtype=np.float64).reshape(rows, cols)

# gec shape: (N, 3) — columns are G, E, C after warmup
gec = np.array(data)

for k in [5, 6]:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(gec)
    centers = kmeans.cluster_centers_
    
    print(f"\n=== k={k} cluster centers (G, E, C) ===")
    for i, c in enumerate(centers):
        pct = np.mean(labels == i) * 100
        print(f"  Cluster {i}: G={c[0]:.3f}  E={c[1]:.3f}  C={c[2]:.3f}  ({pct:.1f}%)")

"""
# 3D scatter for k=5
kmeans6 = KMeans(n_clusters=6, random_state=42, n_init=20)
labels6 = kmeans6.fit_predict(gec)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(gec[:,0], gec[:,1], gec[:,2], 
                     c=labels6, cmap='tab10', alpha=0.4, s=10)
ax.scatter(*kmeans6.cluster_centers_.T, 
           c='black', marker='X', s=200, label='Centers')
ax.set_xlabel('G'); ax.set_ylabel('E'); ax.set_zlabel('C')
plt.title('GEC Space k=6')
plt.legend()
plt.show()
"""

kmeans6 = KMeans(n_clusters=6, random_state=42, n_init=20)
labels6 = kmeans6.fit_predict(gec)

# Load your CSV
df_raw = pd.read_csv('C:/Users/H/Documents/projects/Avarent/data/etf_daily_log_returns.csv', index_col=0, parse_dates=True)

# Dates after warmup (280 days, no 250, technically 280 but htts)
# im assuming len(gec) == len(df_raw) - 280, but those 30 are included.where e is zero
warmup = len(df_raw) - len(gec)
date_index = df_raw.index[warmup:] #hardcoded 280 and 250 first

# Then the rest
df = pd.DataFrame({
    'date': date_index,
    'cluster': labels6,  # use k=6
    'G': gec[:,0],
    'E': gec[:,1],
    'C': gec[:,2]
})

# Crisis cluster (highest E)
crisis_cluster = df.loc[df['E'].idxmax(), 'cluster']
print("=== CRISIS CLUSTER DATES ===")
print(df[df['cluster'] == crisis_cluster][['date','G','E','C']])

# All clusters with date ranges
print("\n=== CLUSTER DATE RANGES ===")
for c in range(6):
    subset = df[df['cluster'] == c]['date']
    if len(subset) > 0:
        print(f"Cluster {c}: {subset.min().date()} → {subset.max().date()}  ({len(subset)} days)")

# High-C cluster
highC_cluster = df.loc[df['C'].idxmax(), 'cluster']
print(f"\n=== HIGH-C CLUSTER DATES (cluster {highC_cluster}) ===")
print(df[df['cluster'] == highC_cluster][['date','G','E','C']].head(30))
