import pandas as pd
from collections import defaultdict, deque

# 1. Load your pairs
df = pd.read_csv('hotel_pairs.csv')  # columns: text_a, text_b, label

# 2. Build adjacency list over only the duplicate edges
adj = defaultdict(set)
for a, b, lbl in zip(df.text_a, df.text_b, df.label):
    if lbl == 1:
        adj[a].add(b)
        adj[b].add(a)

# 3. Find connected components (clusters) via BFS
cluster_id = {}
next_id = 0

for addr in set(df.text_a).union(df.text_b):
    if addr not in cluster_id:
        # BFS this component
        queue = deque([addr])
        while queue:
            node = queue.popleft()
            if node in cluster_id:
                continue
            cluster_id[node] = next_id
            for nbr in adj[node]:
                if nbr not in cluster_id:
                    queue.append(nbr)
        next_id += 1

print(f"Found {next_id} clusters across all addresses")

# 4. Map each row’s text_a/text_b into clusterA/clusterB
df['clusterA'] = df['text_a'].map(cluster_id)
df['clusterB'] = df['text_b'].map(cluster_id)

# 5. Save new CSV for JointBERT training
df.to_csv('hotel_pairs_with_clusters.csv', index=False)
print("Wrote hotel_pairs_with_clusters.csv with clusterA,clusterB columns")
