import pandas as pd
from collections import defaultdict

# 1) Load your label pairs (each row says orig_indexA, orig_indexB are duplicates)
pairs = pd.read_csv('data/inference_labels.csv')  
#   columns: orig_indexA, orig_indexB

# 2) Build connected components over indices
#    We want each group of inter‑connected duplicates to share the same cluster ID.
from collections import deque

# Build adjacency list
adj = defaultdict(list)
for a, b in zip(pairs.orig_indexA, pairs.orig_indexB):
    adj[a].append(b)
    adj[b].append(a)

# Find components via BFS/DFS
cluster_id = {}
current_id = 0
for node in adj:
    if node not in cluster_id:
        # start a new component
        queue = deque([node])
        while queue:
            x = queue.popleft()
            if x in cluster_id:
                continue
            cluster_id[x] = current_id
            for nbr in adj[x]:
                if nbr not in cluster_id:
                    queue.append(nbr)
        current_id += 1

# 3) Assign singleton records (those never in any duplicate pair) their own cluster
#    If you want every record in your hotel list to have a cluster:
all_indices = pd.read_csv('data/addresses_corrected_filled.csv').orig_index.unique()
for idx in all_indices:
    if idx not in cluster_id:
        cluster_id[idx] = current_id
        current_id += 1

# Now `cluster_id[idx]` is a 0..(num_clusters-1) integer for each record.

# 4) When you build your `hotel_pairs.csv` for training,
#    join in these cluster labels:
df = pd.read_csv('data/hotel_pairs.csv')  # must have orig_indexA, orig_indexB, text_a, text_b, label
df['clusterA'] = df.orig_indexA.map(cluster_id)
df['clusterB'] = df.orig_indexB.map(cluster_id)
df.to_csv('data/hotel_pairs_with_clusters.csv', index=False)
