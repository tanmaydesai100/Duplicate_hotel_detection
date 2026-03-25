
# # import os
# # import yaml
# # import pandas as pd
# # import torch
# # from transformers import BertForSequenceClassification, BertTokenizerFast
# # from math import radians, sin, cos, sqrt, atan2
# # from tqdm import tqdm

# # # Haversine distance for filtering
# # def haversine(lat1, lon1, lat2, lon2):
# #     R = 6371000  # Earth radius in metres
# #     phi1, phi2 = radians(lat1), radians(lat2)
# #     dphi = radians(lat2 - lat1)
# #     dlambda = radians(lon2 - lon1)
# #     a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
# #     return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# # if __name__ == '__main__':
# #     # 1. Load config
# #     base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# #     cfg = yaml.safe_load(open(os.path.join(base_dir, 'configs', 'config.yaml')))

# #     # 2. Locate and load the trained model checkpoint
# #     model_root = cfg['training']['output_dir']
# #     ckpt_dirs = sorted([d for d in os.listdir(model_root) if d.startswith('checkpoint')])
# #     print(f"Available checkpoints: {ckpt_dirs}")
# #     model_dir = os.path.join(model_root, ckpt_dirs[-1]) if ckpt_dirs else model_root
# #     print(f"Loading model from {model_dir}")

# #     # 3. Initialize device and load model
# #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #     print(f"Using device: {device}")
# #     model = BertForSequenceClassification.from_pretrained(model_dir).to(device)
# #     model.eval()

# #     # 4. Initialize tokenizer
# #     tokenizer = BertTokenizerFast.from_pretrained(cfg['model']['name_or_path'])

# #     # 5. Read inference inputs (with coords)
# #     inf = pd.read_csv(os.path.join(base_dir, 'data', 'inference_input.csv'))

# #     # 6. Prepare output container and constants
# #     results = []
# #     threshold = float(cfg['data']['distance_threshold_m'])
# #     max_len = int(cfg['model']['max_length'])
# #     n = len(inf)

# #     # 7. Loop through pairs with progress bar
# #     for i in tqdm(range(n), desc="Outer loop"):  # hotel index A
# #         a = inf.iloc[i]
# #         for j in range(i+1, n):  # hotel index B
# #             b = inf.iloc[j]
# #             dist = haversine(a.lat, a.long, b.lat, b.long)
# #             if dist > threshold:
# #                 continue

# #             # Tokenize pair and move inputs to device
# #             inputs = tokenizer(
# #                 a.text, b.text,
# #                 return_tensors='pt',
# #                 truncation=True,
# #                 padding='max_length',
# #                 max_length=max_len
# #             )
# #             inputs = {k: v.to(device) for k, v in inputs.items()}

# #             # Inference
# #             with torch.no_grad():
# #                 logits = model(**inputs).logits
# #                 prob = torch.softmax(logits, dim=1)[0, 1].item()

# #             results.append({
# #                 'orig_indexA': a.orig_index,
# #                 'orig_indexB': b.orig_index,
# #                 'duplicate_prob': prob,
# #                 'distance_m': dist
# #             })

# #     # 8. Save results
# #     out_path = os.path.join(base_dir, 'data', 'inference_results.csv')
# #     pd.DataFrame(results).to_csv(out_path, index=False)
# #     print(f"Saved inference results to {out_path}")

# import os
# import yaml
# import pandas as pd
# import torch
# from transformers import BertForSequenceClassification, BertTokenizerFast
# from math import radians, sin, cos, sqrt, atan2
# from tqdm import tqdm

# # Haversine distance for filtering
# def haversine(lat1, lon1, lat2, lon2):
#     R = 6371000  # Earth radius in metres
#     phi1, phi2 = radians(lat1), radians(lat2)
#     dphi = radians(lat2 - lat1)
#     dlambda = radians(lon2 - lon1)
#     a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
#     return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# if __name__ == '__main__':
#     # 1. Load config
#     base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#     cfg = yaml.safe_load(open(os.path.join(base_dir, 'configs', 'config.yaml')))

#     # 2. Locate and load the trained model checkpoint
#     model_root = cfg['training']['output_dir']
#     ckpt_dirs = sorted([d for d in os.listdir(model_root) if d.startswith('checkpoint')])
#     print(f"Available checkpoints: {ckpt_dirs}")
#     model_dir = os.path.join(model_root, ckpt_dirs[-1]) if ckpt_dirs else model_root
#     print(f"Loading model from {model_dir}")

#     # 3. Initialize device and load model
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
#     model = BertForSequenceClassification.from_pretrained(model_dir).to(device)
#     model.eval()

#     # 4. Initialize tokenizer
#     tokenizer = BertTokenizerFast.from_pretrained(cfg['model']['name_or_path'])

#     # 5. Read inference inputs (with coords)
#     inf = pd.read_csv(os.path.join(base_dir, 'data', 'inference_input.csv'))

#     # 6. Prepare output container and batching parameters
#     results = []
#     threshold = float(cfg['data']['distance_threshold_m'])
#     max_len = int(cfg['model']['max_length'])
#     batch_size = int(cfg.get('inference', {}).get('batch_size', 32))
#     buffer = []

#     def process_batch(batch):
#         texts_a = [x['a_text'] for x in batch]
#         texts_b = [x['b_text'] for x in batch]
#         inputs = tokenizer(
#             texts_a,
#             texts_b,
#             return_tensors='pt',
#             truncation=True,
#             padding='max_length',
#             max_length=max_len
#         )
#         inputs = {k: v.to(device) for k, v in inputs.items()}
#         with torch.no_grad():
#             logits = model(**inputs).logits
#             probs = torch.softmax(logits, dim=1)[:, 1].tolist()
#         for item, prob in zip(batch, probs):
#             results.append({
#                 'orig_indexA': item['idx_a'],
#                 'orig_indexB': item['idx_b'],
#                 'duplicate_prob': prob,
#                 'distance_m': item['distance']
#             })

#     # 7. Loop through pairs with progress bar and batch processing
#     n = len(inf)
#     for i in tqdm(range(n), desc="Outer loop"):  # hotel index A
#         a = inf.iloc[i]
#         for j in range(i+1, n):  # hotel index B
#             b = inf.iloc[j]
#             dist = haversine(a.lat, a.long, b.lat, b.long)
#             if dist > threshold:
#                 continue
#             buffer.append({
#                 'a_text': a.text,
#                 'b_text': b.text,
#                 'idx_a': a.orig_index,
#                 'idx_b': b.orig_index,
#                 'distance': dist
#             })
#             if len(buffer) >= batch_size:
#                 process_batch(buffer)
#                 buffer = []

#     # process remaining
#     if buffer:
#         process_batch(buffer)

#     # 8. Save results
#     out_path = os.path.join(base_dir, 'data', 'inference_results.csv')
#     pd.DataFrame(results).to_csv(out_path, index=False)
#     print(f"Saved inference results to {out_path}")
# import os
# import yaml
# import pandas as pd
# import torch
# from transformers import BertForSequenceClassification, BertTokenizerFast
# from math import radians, sin, cos, sqrt, atan2
# from tqdm import tqdm

# # Haversine distance for filtering
# def haversine(lat1, lon1, lat2, lon2):
#     R = 6371000  # Earth radius in metres
#     phi1, phi2 = radians(lat1), radians(lat2)
#     dphi = radians(lat2 - lat1)
#     dlambda = radians(lon2 - lon1)
#     a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
#     return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# if __name__ == '__main__':
#     # 1. Load config
#     base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#     cfg = yaml.safe_load(open(os.path.join(base_dir, 'configs', 'config.yaml')))
#     inf_cfg = cfg.get('inference', {})

#     # 2. Locate and load the trained model checkpoint
#     model_root = cfg['training']['output_dir']
#     ckpt_dirs = sorted([d for d in os.listdir(model_root) if d.startswith('checkpoint')])
#     print(f"Available checkpoints: {ckpt_dirs}")
#     model_dir = os.path.join(model_root, ckpt_dirs[-1]) if ckpt_dirs else model_root
#     print(f"Loading model from {model_dir}")

#     # 3. Initialize device and load model
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
#     model = BertForSequenceClassification.from_pretrained(model_dir).to(device)
#     model.eval()

#     # 4. Initialize tokenizer
#     tokenizer = BertTokenizerFast.from_pretrained(cfg['model']['name_or_path'])

#     # 5. Read inference inputs (with coords)
#     inf_path = os.path.join(base_dir, 'data', 'inference_input.csv')
#     inf = pd.read_csv(inf_path)

#     # 6. Read ground-truth duplicate labels
#     labels_path = os.path.join(base_dir, 'data', 'inference_labels.csv')
#     df_labels = pd.read_csv(labels_path)
#     # build set of duplicate pairs (as frozenset for unordered)
#     dup_set = set(
#         frozenset((row.orig_indexA, row.orig_indexB))
#         for row in df_labels.itertuples()
#     )

#     # 7. Prepare output container and batching parameters
#     results = []
#     prob_threshold = float(inf_cfg.get('duplicate_prob_threshold', 0.5))
#     dist_thresh = float(cfg['data']['distance_threshold_m'])
#     max_len = int(cfg['model']['max_length'])
#     batch_size = int(inf_cfg.get('batch_size', 32))
#     buffer = []

#     def process_batch(batch):
#         texts_a = [x['a_text'] for x in batch]
#         texts_b = [x['b_text'] for x in batch]
#         inputs = tokenizer(
#             texts_a,
#             texts_b,
#             return_tensors='pt',
#             truncation=True,
#             padding='max_length',
#             max_length=max_len
#         )
#         inputs = {k: v.to(device) for k, v in inputs.items()}
#         with torch.no_grad():
#             logits = model(**inputs).logits
#             probs = torch.softmax(logits, dim=1)[:, 1].tolist()
#         for item, prob in zip(batch, probs):
#             pair = frozenset((item['idx_a'], item['idx_b']))
#             true = 1 if pair in dup_set else 0
#             pred = 1 if prob >= prob_threshold else 0
#             results.append({
#                 'orig_indexA': item['idx_a'],
#                 'orig_indexB': item['idx_b'],
#                 'duplicate_prob': prob,
#                 'pred_label': pred,
#                 'true_label': true,
#                 'pred_string': 'duplicate' if pred == 1 else 'non-duplicate',
#                 'true_string': 'duplicate' if true == 1 else 'non-duplicate',
#                 'distance_m': item['distance']
#             })

#     # 8. Loop through pairs with progress bar and batch processing
#     n = len(inf)
#     for i in tqdm(range(n), desc="Outer loop"):  # hotel index A
#         a = inf.iloc[i]
#         for j in range(i + 1, n):  # hotel index B
#             b = inf.iloc[j]
#             dist = haversine(a.lat, a.long, b.lat, b.long)
#             if dist > dist_thresh:
#                 continue
#             buffer.append({
#                 'a_text': a.text,
#                 'b_text': b.text,
#                 'idx_a': a.orig_index,
#                 'idx_b': b.orig_index,
#                 'distance': dist
#             })
#             if len(buffer) >= batch_size:
#                 process_batch(buffer)
#                 buffer = []

#     # process remaining
#     if buffer:
#         process_batch(buffer)

#     # 9. Save results with string labels
#     out_path = os.path.join(base_dir, 'data', 'inference_results.csv')
#     df_res = pd.DataFrame(results)
#     df_res.to_csv(out_path, index=False)
#     print(f"Saved inference results with labels to {out_path}")
#new commented code
# haversine vanilla bert
# import os
# import yaml
# import pandas as pd
# import torch
# from transformers import BertForSequenceClassification, BertTokenizerFast
# from math import radians, sin, cos, sqrt, atan2
# from tqdm.auto import tqdm
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# # Haversine distance for filtering
# def haversine(lat1, lon1, lat2, lon2):
#     R = 6371000  # Earth radius in metres
#     phi1, phi2 = radians(lat1), radians(lat2)
#     dphi = radians(lat2 - lat1)
#     dlambda = radians(lon2 - lon1)
#     a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
#     return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# if __name__ == '__main__':
#     # 1. Load config
#     base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#     cfg = yaml.safe_load(open(os.path.join(base_dir, 'configs', 'config.yaml')))
#     inf_cfg = cfg.get('inference', {})

#     # 2. Locate and load the trained model checkpoint
#     model_root = cfg['training']['output_dir']
#     ckpt_dirs = sorted([d for d in os.listdir(model_root) if d.startswith('checkpoint')])
#     print(f"Available checkpoints: {ckpt_dirs}")
#     model_dir = os.path.join(model_root, ckpt_dirs[-1]) if ckpt_dirs else model_root
#     print(f"Loading model from {model_dir}")

#     # 3. Initialize device and load model
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
#     model = BertForSequenceClassification.from_pretrained(model_dir).to(device)
#     model.eval()

#     # 4. Initialize tokenizer
#     tokenizer = BertTokenizerFast.from_pretrained(cfg['model']['name_or_path'])

#     # 5. Read inference inputs (with coords)
#     inf_path = os.path.join(base_dir, 'data', 'addresses_corrected_filled.csv')
#     inf = pd.read_csv(inf_path)

#     # 6. Read ground-truth duplicate labels
#     labels_path = os.path.join(base_dir, 'data', 'inference_labels.csv')
#     df_labels = pd.read_csv(labels_path)
#     dup_set = set(
#         frozenset((row.orig_indexA, row.orig_indexB))
#         for row in df_labels.itertuples()
#     )

#     # 7. Prepare output and batching params
#     results = []
#     prob_threshold = float(inf_cfg.get('duplicate_prob_threshold', 0.7))
#     dist_thresh = float(cfg['data']['distance_threshold_m'])
#     max_len = int(cfg['model']['max_length'])
#     batch_size = int(inf_cfg.get('batch_size', 32))
#     buffer = []

#     def process_batch(batch):
#         texts_a = [x['a_text'] for x in batch]
#         texts_b = [x['b_text'] for x in batch]
#         inputs = tokenizer(
#             texts_a,
#             texts_b,
#             return_tensors='pt',
#             truncation=True,
#             padding='max_length',
#             max_length=max_len
#         )
#         inputs = {k: v.to(device) for k, v in inputs.items()}
#         with torch.no_grad():
#             logits = model(**inputs).logits
#             probs = torch.softmax(logits, dim=1)[:, 1].tolist()
#         for item, prob in zip(batch, probs):
#             pair = frozenset((item['idx_a'], item['idx_b']))
#             true = 1 if pair in dup_set else 0
#             pred = 1 if prob >= prob_threshold else 0
#             results.append({
#                 'orig_indexA': item['idx_a'],
#                 'orig_indexB': item['idx_b'],
#                 'duplicate_prob': prob,
#                 'pred_label': pred,
#                 'true_label': true,
#                 'pred_string': 'duplicate' if pred == 1 else 'non-duplicate',
#                 'true_string': 'duplicate' if true == 1 else 'non-duplicate',
#                 'distance_m': item['distance']
#             })

#     # 8. Loop and batch inference
#     for i in tqdm(range(len(inf)), desc="Outer loop"):
#         a = inf.iloc[i]
#         for j in range(i + 1, len(inf)):
#             b = inf.iloc[j]
#             dist = haversine(a.lat, a.long, b.lat, b.long)
#             if dist > dist_thresh:
#                 continue
#             buffer.append({
#                 'a_text': a.text,
#                 'b_text': b.text,
#                 'idx_a': a.orig_index,
#                 'idx_b': b.orig_index,
#                 'distance': dist
#             })
#             if len(buffer) >= batch_size:
#                 process_batch(buffer)
#                 buffer = []
#     if buffer:
#         process_batch(buffer)

#     # 9. Save and evaluate metrics
#     out_path = os.path.join(base_dir, 'data', 'inference_results.csv')
#     df_res = pd.DataFrame(results)
#     df_res.to_csv(out_path, index=False)
#     print(f"Saved inference results with labels to {out_path}")

#     # 9.1) Dump true positives & false positives
#     tp_path = os.path.join(base_dir, 'data', 'true_positives.csv')
#     fp_path = os.path.join(base_dir, 'data', 'false_positives.csv')
#     df_res[
#         (df_res['pred_label'] == 1) &
#         (df_res['true_label'] == 1)
#     ].to_csv(tp_path, index=False)
#     df_res[
#         (df_res['pred_label'] == 1) &
#         (df_res['true_label'] == 0)
#     ].to_csv(fp_path, index=False)
#     print(f"True positives → {tp_path}")
#     print(f"False positives → {fp_path}")

#     # 9.2) Merge in address strings for both files
#     #    so you can inspect the actual addresses
#     inf_df = inf.rename(columns={'orig_index': 'orig_indexA', 'text': 'addressA'})
#     inf_b = inf.rename(columns={'orig_index': 'orig_indexB', 'text': 'addressB'})

#     tp = pd.read_csv(tp_path)
#     fp = pd.read_csv(fp_path)

#     tp = (
#         tp
#         .merge(inf_df[['orig_indexA', 'addressA']], on='orig_indexA', how='left')
#         .merge(inf_b[['orig_indexB', 'addressB']], on='orig_indexB', how='left')
#     )
#     fp = (
#         fp
#         .merge(inf_df[['orig_indexA', 'addressA']], on='orig_indexA', how='left')
#         .merge(inf_b[['orig_indexB', 'addressB']], on='orig_indexB', how='left')
#     )

#     tp_with_addr = os.path.join(base_dir, 'data', 'true_positives_with_addresses.csv')
#     fp_with_addr = os.path.join(base_dir, 'data', 'false_positives_with_addresses.csv')
#     tp.to_csv(tp_with_addr, index=False)
#     fp.to_csv(fp_with_addr, index=False)
#     print(f"True positives with addresses → {tp_with_addr}")
#     print(f"False positives with addresses → {fp_with_addr}")

#     # 9.3) Compute evaluation metrics
#     y_true = df_res['true_label']
#     y_pred = df_res['pred_label']
#     acc = accuracy_score(y_true, y_pred)
#     prec = precision_score(y_true, y_pred, zero_division=0)
#     rec = recall_score(y_true, y_pred, zero_division=0)
#     f1 = f1_score(y_true, y_pred, zero_division=0)
#     print("\nEvaluation Metrics at threshold {}:".format(prob_threshold))
#     print(f"  Accuracy : {acc:.4f}")
#     print(f"  Precision: {prec:.4f}")
#     print(f"  Recall   : {rec:.4f}")
#     print(f"  F1 Score : {f1:.4f}")


# import os
# import yaml
# import pandas as pd
# import torch
# from transformers import BertForSequenceClassification, BertTokenizerFast
# from tqdm import tqdm
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# if __name__ == '__main__':
#     # 1. Load config
#     base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#     cfg = yaml.safe_load(open(os.path.join(base_dir, 'configs', 'config.yaml')))
#     inf_cfg = cfg.get('inference', {})

#     # 2. Locate and load the trained model checkpoint
#     model_root = cfg['training']['output_dir']
#     ckpt_dirs = sorted([d for d in os.listdir(model_root) if d.startswith('checkpoint')])
#     print(f"Available checkpoints: {ckpt_dirs}")
#     model_dir = os.path.join(model_root, ckpt_dirs[-1]) if ckpt_dirs else model_root
#     print(f"Loading model from {model_dir}")

#     # 3. Initialize device and load model
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
#     model = BertForSequenceClassification.from_pretrained(model_dir).to(device)
#     model.eval()

#     # 4. Initialize tokenizer
#     tokenizer = BertTokenizerFast.from_pretrained(cfg['model']['name_or_path'])

#     # 5. Read inference inputs
#     inf_path = os.path.join(base_dir, 'data', 'inference_input.csv')
#     inf = pd.read_csv(inf_path)

#     # 6. Read ground-truth duplicate labels
#     labels_path = os.path.join(base_dir, 'data', 'inference_labels.csv')
#     df_labels = pd.read_csv(labels_path)
#     dup_set = set(
#         frozenset((row.orig_indexA, row.orig_indexB))
#         for row in df_labels.itertuples()
#     )

#     # 7. Prepare output and batching params
#     results = []
#     prob_threshold = float(inf_cfg.get('duplicate_prob_threshold', 0.7))
#     max_len = int(cfg['model']['max_length'])
#     batch_size = int(inf_cfg.get('batch_size', 32))
#     buffer = []

#     def process_batch(batch):
#         texts_a = [x['a_text'] for x in batch]
#         texts_b = [x['b_text'] for x in batch]
#         inputs = tokenizer(
#             texts_a,
#             texts_b,
#             return_tensors='pt',
#             truncation=True,
#             padding='max_length',
#             max_length=max_len
#         )
#         inputs = {k: v.to(device) for k, v in inputs.items()}
#         with torch.no_grad():
#             logits = model(**inputs).logits
#             probs = torch.softmax(logits, dim=1)[:, 1].tolist()
#         for item, prob in zip(batch, probs):
#             pair = frozenset((item['idx_a'], item['idx_b']))
#             true = 1 if pair in dup_set else 0
#             pred = 1 if prob >= prob_threshold else 0
#             results.append({
#                 'orig_indexA': item['idx_a'],
#                 'orig_indexB': item['idx_b'],
#                 'duplicate_prob': prob,
#                 'pred_label': pred,
#                 'true_label': true,
#                 'pred_string': 'duplicate' if pred == 1 else 'non-duplicate',
#                 'true_string': 'duplicate' if true == 1 else 'non-duplicate'
#             })

#     # 8. Loop through all pairs without geographic filtering
#     n = len(inf)
#     for i in tqdm(range(n), desc="Outer loop"):  # hotel index A
#         a = inf.iloc[i]
#         for j in range(i + 1, n):  # hotel index B
#             b = inf.iloc[j]
#             buffer.append({
#                 'a_text': a.text,
#                 'b_text': b.text,
#                 'idx_a': a.orig_index,
#                 'idx_b': b.orig_index
#             })
#             if len(buffer) >= batch_size:
#                 process_batch(buffer)
#                 buffer = []

#     # process remaining
#     if buffer:
#         process_batch(buffer)

#     # 9. Save and evaluate metrics
#     out_path = os.path.join(base_dir, 'data', 'inference_results.csv')
#     df_res = pd.DataFrame(results)
#     df_res.to_csv(out_path, index=False)
#     print(f"Saved inference results to {out_path}")

#     # Compute evaluation metrics
#     y_true = df_res['true_label']
#     y_pred = df_res['pred_label']
#     acc = accuracy_score(y_true, y_pred)
#     prec = precision_score(y_true, y_pred, zero_division=0)
#     rec = recall_score(y_true, y_pred, zero_division=0)
#     f1 = f1_score(y_true, y_pred, zero_division=0)
#     print(f"\nEvaluation Metrics at threshold {prob_threshold}:")
#     print(f"  Accuracy : {acc:.4f}")
#     print(f"  Precision: {prec:.4f}")
#     print(f"  Recall   : {rec:.4f}")
#     print(f"  F1 Score : {f1:.4f}")
# import os
# import yaml
# import pandas as pd
# import torch
# import argparse
# from transformers import BertForSequenceClassification, BertTokenizerFast
# from tqdm import tqdm
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Evaluate duplicate detection model.")
#     parser.add_argument("--threshold", type=float,
#                         help="Override duplicate probability threshold (default from config)")
#     args = parser.parse_args()

#     # 1. Load config\    
#     base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#     cfg = yaml.safe_load(open(os.path.join(base_dir, 'configs', 'config.yaml')))
#     inf_cfg = cfg.get('inference', {})

#     # Determine threshold: command-line > config > default 0.5
#     prob_threshold = args.threshold if args.threshold is not None \
#         else float(inf_cfg.get('duplicate_prob_threshold', 0.5))
#     print(f"Using duplicate probability threshold: {prob_threshold}")

#     # 2. Locate and load the trained model checkpoint
#     model_root = cfg['training']['output_dir']
#     ckpt_dirs = sorted([d for d in os.listdir(model_root) if d.startswith('checkpoint')])
#     print(f"Available checkpoints: {ckpt_dirs}")
#     model_dir = os.path.join(model_root, ckpt_dirs[-1]) if ckpt_dirs else model_root
#     print(f"Loading model from {model_dir}")

#     # 3. Initialize device and load model
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
#     model = BertForSequenceClassification.from_pretrained(model_dir).to(device)
#     model.eval()

#     # 4. Initialize tokenizer
#     tokenizer = BertTokenizerFast.from_pretrained(cfg['model']['name_or_path'])

#     # 5. Read inference inputs
#     inf_path = os.path.join(base_dir, 'data', 'inference_input.csv')
#     inf = pd.read_csv(inf_path)

#     # 6. Read ground-truth duplicate labels
#     labels_path = os.path.join(base_dir, 'data', 'inference_labels.csv')
#     df_labels = pd.read_csv(labels_path)
#     dup_set = set(
#         frozenset((row.orig_indexA, row.orig_indexB))
#         for row in df_labels.itertuples()
#     )

#     # 7. Prepare batching params and results
#     max_len = int(cfg['model']['max_length'])
#     batch_size = int(inf_cfg.get('batch_size', 32))
#     results = []
#     buffer = []

#     def process_batch(batch):
#         texts_a = [x['a_text'] for x in batch]
#         texts_b = [x['b_text'] for x in batch]
#         inputs = tokenizer(
#             texts_a,
#             texts_b,
#             return_tensors='pt',
#             truncation=True,
#             padding='max_length',
#             max_length=max_len
#         )
#         inputs = {k: v.to(device) for k, v in inputs.items()}
#         with torch.no_grad():
#             logits = model(**inputs).logits
#             probs = torch.softmax(logits, dim=1)[:, 1].tolist()
#         for item, prob in zip(batch, probs):
#             pair = frozenset((item['idx_a'], item['idx_b']))
#             true = 1 if pair in dup_set else 0
#             pred = 1 if prob >= prob_threshold else 0
#             results.append({
#                 'orig_indexA': item['idx_a'],
#                 'orig_indexB': item['idx_b'],
#                 'duplicate_prob': prob,
#                 'pred_label': pred,
#                 'true_label': true,
#                 'pred_string': 'duplicate' if pred == 1 else 'non-duplicate',
#                 'true_string': 'duplicate' if true == 1 else 'non-duplicate'
#             })

#     # 8. Loop through all pairs
#     n = len(inf)
#     for i in tqdm(range(n), desc="Outer loop"):  
#         a = inf.iloc[i]
#         for j in range(i + 1, n):
#             b = inf.iloc[j]
#             buffer.append({
#                 'a_text': a.text,
#                 'b_text': b.text,
#                 'idx_a': a.orig_index,
#                 'idx_b': b.orig_index
#             })
#             if len(buffer) >= batch_size:
#                 process_batch(buffer)
#                 buffer = []
#     if buffer:
#         process_batch(buffer)

#     # 9. Save results
#     out_path = os.path.join(base_dir, 'data', 'inference_results.csv')
#     df_res = pd.DataFrame(results)
#     df_res.to_csv(out_path, index=False)
#     print(f"Saved inference results to {out_path}")

#     # 10. Extract false and true positives with addresses and save
#     # Merge addresses back
#     addr_df = inf[['orig_index', 'text']].rename(columns={'text': 'address'})
#     df_res = (
#         df_res
#         .merge(addr_df, how='left', left_on='orig_indexA', right_on='orig_index')
#         .rename(columns={'address':'addressA'})
#         .drop(columns=['orig_index'])
#         .merge(addr_df, how='left', left_on='orig_indexB', right_on='orig_index')
#         .rename(columns={'address':'addressB'})
#         .drop(columns=['orig_index'])
#     )

#     # False positives: predicted duplicate but actually non-duplicate
#     fp = df_res[(df_res.pred_label == 1) & (df_res.true_label == 0)][
#         ['orig_indexA','addressA','orig_indexB','addressB','duplicate_prob']
#     ]
#     fp.to_csv(os.path.join(base_dir, 'data', 'false_positives.csv'), index=False)
#     print(f"Found {len(fp)} false positives; saved to false_positives.csv")

#     # True positives: predicted duplicate and actually duplicate
#     tp = df_res[(df_res.pred_label == 1) & (df_res.true_label == 1)][
#         ['orig_indexA','addressA','orig_indexB','addressB','duplicate_prob']
#     ]
#     tp.to_csv(os.path.join(base_dir, 'data', 'true_positives.csv'), index=False)
#     print(f"Found {len(tp)} true positives; saved to true_positives.csv")

#     # 11. Compute and print metrics
#     y_true = df_res['true_label']
#     y_pred = df_res['pred_label']
#     acc = accuracy_score(y_true, y_pred)
#     prec = precision_score(y_true, y_pred, zero_division=0)
#     rec = recall_score(y_true, y_pred, zero_division=0)
#     f1 = f1_score(y_true, y_pred, zero_division=0)
#     print(f"\nEvaluation Metrics at threshold {prob_threshold}:")
#     print(f"  Accuracy : {acc:.4f}")
#     print(f"  Precision: {prec:.4f}")
#     print(f"  Recall   : {rec:.4f}")
#     print(f"  F1 Score : {f1:.4f}")

# import os
# import yaml
# import pandas as pd
# import torch
# from transformers import BertTokenizerFast
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from tqdm.auto import tqdm
# from math import radians, sin, cos, sqrt, atan2

# # Import your Siamese model classes
# from train import SiameseBertClassifier, SiameseConfig

# # Haversine distance for filtering
# def haversine(lat1, lon1, lat2, lon2):
#     R = 6371000  # Earth radius in metres
#     phi1, phi2 = radians(lat1), radians(lat2)
#     dphi = radians(lat2 - lat1)
#     dlambda = radians(lon2 - lon1)
#     a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
#     return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# if __name__ == '__main__':
#     # 1. Load config
#     base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#     cfg = yaml.safe_load(open(os.path.join(base_dir, 'configs', 'config.yaml')))
#     inf_cfg = cfg.get('inference', {})

#     # 2. Locate best checkpoint
#     ckpt_root = cfg['training']['output_dir']
#     ckpts = sorted(d for d in os.listdir(ckpt_root) if d.startswith('checkpoint'))
#     best_ckpt = os.path.join(ckpt_root, ckpts[-1] if ckpts else '')
#     print(f"Loading model from {best_ckpt}")

#     # 3. Device & model
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
#     model_cfg = SiameseConfig.from_pretrained(best_ckpt)
#     model = SiameseBertClassifier.from_pretrained(best_ckpt, config=model_cfg).to(device)
#     model.eval()

#     # 4. Tokenizer
#     tokenizer = BertTokenizerFast.from_pretrained(cfg['model']['name_or_path'])

#             # 5. Read addresses
#     addr_path = inf_cfg.get('input_csv', 'data/addresses_corrected_filled.csv')
#     # df_addr = pd.read_csv(os.path.join(base_dir, addr_path))  # columns: orig_index,text,lat,long
#     df_addr = pd.read_csv(os.path.join(base_dir, addr_path)).head(100)

#     # Generate all unique address pairs (i < j)
#     pairs = []
#     for i in range(len(df_addr)):
#         a = df_addr.iloc[i]
#         for j in range(i + 1, len(df_addr)):
#             b = df_addr.iloc[j]
#             pairs.append({
#                 'a_text': a.text,
#                 'b_text': b.text,
#                 'idx_a': a.orig_index,
#                 'idx_b': b.orig_index
#             })
#     print(f"Generated {len(pairs)} total address pairs for inference")

#     # Load evaluation pairs (true duplicates)
#     labels_csv = inf_cfg.get('labels_csv', 'data/inference_labels.csv')
#     df_labels = pd.read_csv(os.path.join(base_dir, labels_csv))  # columns: orig_indexA,orig_indexB
#     dup_set = set(frozenset((r.orig_indexA, r.orig_indexB)) for r in df_labels.itertuples())

#     # 6. Batch inference
#     results = []
#     batch_size = int(inf_cfg.get('batch_size', 32))
#     prob_thresh = float(inf_cfg.get('duplicate_prob_threshold', 0.7))

#     def predict_batch(batch):
#         enc1 = tokenizer([x['a_text'] for x in batch], padding='max_length', truncation=True,
#                          max_length=cfg['model']['max_length'], return_tensors='pt')
#         enc2 = tokenizer([x['b_text'] for x in batch], padding='max_length', truncation=True,
#                          max_length=cfg['model']['max_length'], return_tensors='pt')
#         inputs = {
#             'input_ids1': enc1['input_ids'].to(device),
#             'attention_mask1': enc1['attention_mask'].to(device),
#             'input_ids2': enc2['input_ids'].to(device),
#             'attention_mask2': enc2['attention_mask'].to(device),
#         }
#         with torch.no_grad():
#             logits = model(**inputs).logits
#             return torch.softmax(logits, dim=1)[:,1].cpu().tolist()

#     # iterate in batches
#     for i in tqdm(range(0, len(pairs), batch_size), desc='Inference batches'):
#         batch = pairs[i:i+batch_size]
#         probs = predict_batch(batch)
#         for item, p in zip(batch, probs):
#             results.append({**item, 'duplicate_prob': p, 'pred_label': int(p>=prob_thresh)})

#     # 7. Save inference results
#     out_csv = inf_cfg.get('output_csv', 'data/inference_results.csv')
#     out_path = os.path.join(base_dir, out_csv)
#     pd.DataFrame(results).to_csv(out_path, index=False)
#     print(f"Saved inference results to {out_path}")

#     # 8. Load labels and evaluate
#     labels_csv = inf_cfg.get('labels_csv', 'data/inference_labels.csv')
#     df_labels = pd.read_csv(os.path.join(base_dir, labels_csv))
#     dup_set = set(frozenset((r.orig_indexA, r.orig_indexB)) for r in df_labels.itertuples())

#     y_true = [1 if frozenset((r['idx_a'], r['idx_b'])) in dup_set else 0 for r in results]
#     y_pred = [r['pred_label'] for r in results]

#     acc = accuracy_score(y_true, y_pred)
#     prec = precision_score(y_true, y_pred, zero_division=0)
#     rec = recall_score(y_true, y_pred, zero_division=0)
#     f1 = f1_score(y_true, y_pred, zero_division=0)
#     print("\nEvaluation Metrics:")
#     print(f"  Accuracy : {acc:.4f}")
#     print(f"  Precision: {prec:.4f}")
#     print(f"  Recall   : {rec:.4f}")
#     print(f"  F1 Score : {f1:.4f}")
#     df_labels = pd.read_csv(os.path.join(base_dir, labels_csv))
#     dup_set = set(frozenset((r.orig_indexA, r.orig_indexB)) for r in df_labels.itertuples())

#     # Gather misclassified rows
#     wrong_rows = []
#     for r in results:
#         idx_a, idx_b = r['idx_a'], r['idx_b']
#         true_label = int(frozenset((idx_a, idx_b)) in dup_set)
#         if r['pred_label'] != true_label:
#             wrong_rows.append({
#                 'idx_a'         : idx_a,
#                 'idx_b'         : idx_b,
#                 'a_text'        : r['a_text'],
#                 'b_text'        : r['b_text'],
#                 'duplicate_prob': r['duplicate_prob'],
#                 'pred_label'    : r['pred_label'],
#                 'true_label'    : true_label
#             })

#     # Convert to DataFrame
#     df_wrong = pd.DataFrame(wrong_rows)

#     # 1) Print to console
#     print("\nMisclassified Pairs:")
#     for _, row in df_wrong.iterrows():
#         print(f"- [{row.idx_a}] “{row.a_text}”  ↔  [{row.idx_b}] “{row.b_text}”"
#               f"  (pred={row.pred_label}, true={row.true_label}, p={row.duplicate_prob:.2f})")

#     # 2) Save to CSV
#     mis_csv = inf_cfg.get('misclassified_csv', 'data/misclassified_pairs.csv')
#     mis_path = os.path.join(base_dir, mis_csv)
#     df_wrong.to_csv(mis_path, index=False)
#     print(f"\nSaved misclassified pairs to {mis_path}")

# keeping haversine
# 0.4
# import os
# import yaml
# import pandas as pd
# import torch
# import torch.nn.functional as F
# from transformers import (
#     BertTokenizerFast,
#     BertConfig,
#     BertPreTrainedModel,
#     BertModel
# )
# from math import radians, sin, cos, sqrt, atan2
# from tqdm.auto import tqdm
# from sklearn.metrics import accuracy_score, precision_recall_curve
# from Levenshtein import ratio

# # Haversine distance function
# def haversine(lat1, lon1, lat2, lon2):
#     R = 6371000  # Earth radius in meters
#     phi1, phi2 = radians(lat1), radians(lat2)
#     dphi = radians(lat2 - lat1)
#     dlambda = radians(lon2 - lon1)
#     a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
#     return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# # Simple similarity head matching training
# class BertSimilarityHead(BertPreTrainedModel):
#     config_class = BertConfig

#     def __init__(self, config):
#         super().__init__(config)
#         self.bert = BertModel(config)
#         hidden = config.hidden_size
#         # classifier on [pooled_output; edit_dist]
#         self.classifier = torch.nn.Linear(hidden + 1, config.num_labels)
#         self.init_weights()

#     def forward(self, input_ids, attention_mask=None, token_type_ids=None, edit_dist=None, **kwargs):
#         outputs = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             return_dict=True
#         )
#         pooled = outputs.pooler_output
#         # append edit distance
#         edit = edit_dist.unsqueeze(1).to(pooled.dtype)
#         feats = torch.cat([pooled, edit], dim=1)
#         logits = self.classifier(feats)
#         return logits

# if __name__ == '__main__':
#     base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#     cfg = yaml.safe_load(open(os.path.join(base_dir, 'configs', 'config.yaml')))
#     inf_cfg = cfg.get('inference', {})

#     model_dir = cfg['training']['output_dir']
#     tokenizer_dir = model_dir

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     bert_conf = BertConfig.from_pretrained(
#         model_dir,
#         num_labels=int(cfg['model']['num_labels'])
#     )
#     model = BertSimilarityHead.from_pretrained(model_dir, config=bert_conf).to(device)
#     model.eval()

#     tokenizer = BertTokenizerFast.from_pretrained(tokenizer_dir)

#     inf = pd.read_csv(os.path.join(base_dir, 'data', 'addresses_corrected_filled.csv'))
#     labels = pd.read_csv(os.path.join(base_dir, 'data', 'inference_labels.csv'))
#     dup = set(frozenset((r.orig_indexA, r.orig_indexB)) for r in labels.itertuples())

#     thresh = float(cfg['data']['distance_threshold_m'])
#     max_len = int(cfg['model']['max_length'])
#     batch_size = int(inf_cfg.get('batch_size', 32))

#     probs, trues = [], []
#     buffer = []

#     def process_batch(batch):
#         texts_a = [x['a'] for x in batch]
#         texts_b = [x['b'] for x in batch]
#         ed = [ratio(a, b) for a, b in zip(texts_a, texts_b)]
#         enc = tokenizer(
#             texts_a,
#             texts_b,
#             return_tensors='pt',
#             padding='max_length',
#             truncation=True,
#             max_length=max_len
#         )
#         enc = {k: v.to(device) for k, v in enc.items()}
#         edt = torch.tensor(ed, device=device)
#         with torch.no_grad():
#             logits = model(
#                 input_ids=enc['input_ids'],
#                 attention_mask=enc['attention_mask'],
#                 token_type_ids=enc['token_type_ids'],
#                 edit_dist=edt
#             )
#             probs_batch = F.softmax(logits, dim=1)[:, 1].cpu().tolist()
#         for item, p in zip(batch, probs_batch):
#             pair = frozenset((item['i'], item['j']))
#             probs.append(p)
#             trues.append(1 if pair in dup else 0)

#     for idx in tqdm(range(len(inf)), desc='Comparisons'):
#         a = inf.iloc[idx]
#         for jdx in range(idx + 1, len(inf)):
#             b = inf.iloc[jdx]
#             dist = haversine(a.lat, a.long, b.lat, b.long)
#             if dist > thresh:
#                 continue
#             buffer.append({'a': a.text, 'b': b.text, 'i': a.orig_index, 'j': b.orig_index})
#             if len(buffer) >= batch_size:
#                 process_batch(buffer)
#                 buffer = []
#     if buffer:
#         process_batch(buffer)

#     # Compute precision-recall and best threshold
#     prec, rec, th = precision_recall_curve(trues, probs)
#     f1_scores = 2 * prec * rec / (prec + rec + 1e-8)
#     best_idx = f1_scores.argmax()
#     best_thresh = th[best_idx]
#     best_prec = prec[best_idx]
#     best_rec = rec[best_idx]
#     best_f1 = f1_scores[best_idx]

#     # Print metrics
#     print(f"Best threshold = {best_thresh:.3f}")
#     print(f"Precision @ best F1: {best_prec:.4f}")
#     print(f"Recall    @ best F1: {best_rec:.4f}")
#     print(f"F1 Score  @ best F1: {best_f1:.4f}")
#     acc = accuracy_score(trues, [1 if p >= best_thresh else 0 for p in probs])
#     print(f"Accuracy  = {acc:.4f}")

#     # Save threshold sweep to CSV
#     df_metrics = pd.DataFrame({
#         'threshold': th,
#         'precision': prec[:-1],
#         'recall': rec[:-1],
#         'f1': f1_scores[:-1]
#     })
#     df_metrics.to_csv(
#         os.path.join(base_dir, 'data', 'threshold_sweep.csv'),
#         index=False
#     )


# without haversine
# evaluate.py
# import os
# import yaml
# import pandas as pd
# import torch
# import torch.nn.functional as F
# from transformers import (
#     BertTokenizerFast,
#     BertConfig,
#     BertPreTrainedModel,
#     BertModel
# )
# from tqdm.auto import tqdm
# from sklearn.metrics import accuracy_score, precision_recall_curve
# from Levenshtein import ratio

# # Simple similarity head matching training
# class BertSimilarityHead(BertPreTrainedModel):
#     config_class = BertConfig

#     def __init__(self, config):
#         super().__init__(config)
#         self.bert = BertModel(config)
#         hidden = config.hidden_size
#         # classifier on [pooled_output; edit_dist]
#         self.classifier = torch.nn.Linear(hidden + 1, config.num_labels)
#         self.init_weights()

#     def forward(self, input_ids, attention_mask=None, token_type_ids=None, edit_dist=None, **kwargs):
#         outputs = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             return_dict=True
#         )
#         pooled = outputs.pooler_output
#         # append edit distance
#         edit = edit_dist.unsqueeze(1).to(pooled.dtype)
#         feats = torch.cat([pooled, edit], dim=1)
#         logits = self.classifier(feats)
#         return logits

# if __name__ == '__main__':
#     base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#     cfg = yaml.safe_load(open(os.path.join(base_dir, 'configs', 'config.yaml')))
#     inf_cfg = cfg.get('inference', {})

#     model_dir = cfg['training']['output_dir']
#     tokenizer_dir = model_dir

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     bert_conf = BertConfig.from_pretrained(
#         model_dir,
#         num_labels=int(cfg['model']['num_labels'])
#     )
#     model = BertSimilarityHead.from_pretrained(model_dir, config=bert_conf).to(device)
#     model.eval()

#     tokenizer = BertTokenizerFast.from_pretrained(tokenizer_dir)

#     inf = pd.read_csv(os.path.join(base_dir, 'data', 'addresses_corrected_filled.csv'))
#     labels = pd.read_csv(os.path.join(base_dir, 'data', 'inference_labels.csv'))
#     dup = set(frozenset((r.orig_indexA, r.orig_indexB)) for r in labels.itertuples())

#     max_len = int(cfg['model']['max_length'])
#     batch_size = int(inf_cfg.get('batch_size', 32))

#     probs, trues = [], []
#     buffer = []

#     def process_batch(batch):
#         texts_a = [x['a'] for x in batch]
#         texts_b = [x['b'] for x in batch]
#         ed = [ratio(a, b) for a, b in zip(texts_a, texts_b)]
#         enc = tokenizer(
#             texts_a,
#             texts_b,
#             return_tensors='pt',
#             padding='max_length',
#             truncation=True,
#             max_length=max_len
#         )
#         enc = {k: v.to(device) for k, v in enc.items()}
#         edt = torch.tensor(ed, device=device)
#         with torch.no_grad():
#             logits = model(
#                 input_ids=enc['input_ids'],
#                 attention_mask=enc['attention_mask'],
#                 token_type_ids=enc['token_type_ids'],
#                 edit_dist=edt
#             )
#             probs_batch = F.softmax(logits, dim=1)[:, 1].cpu().tolist()
#         for item, p in zip(batch, probs_batch):
#             pair = frozenset((item['i'], item['j']))
#             probs.append(p)
#             trues.append(1 if pair in dup else 0)

#     for idx in tqdm(range(len(inf)), desc='Comparisons'):
#         a = inf.iloc[idx]
#         for jdx in range(idx + 1, len(inf)):
#             b = inf.iloc[jdx]
#             buffer.append({'a': a.text, 'b': b.text, 'i': a.orig_index, 'j': b.orig_index})
#             if len(buffer) >= batch_size:
#                 process_batch(buffer)
#                 buffer = []
#     if buffer:
#         process_batch(buffer)

#     # Compute precision-recall and best threshold
#     prec, rec, th = precision_recall_curve(trues, probs)
#     f1_scores = 2 * prec * rec / (prec + rec + 1e-8)
#     best_idx = f1_scores.argmax()
#     best_thresh = th[best_idx]
#     best_prec = prec[best_idx]
#     best_rec = rec[best_idx]
#     best_f1 = f1_scores[best_idx]

#     # Print metrics
#     print(f"Best threshold = {best_thresh:.3f}")
#     print(f"Precision @ best F1: {best_prec:.4f}")
#     print(f"Recall    @ best F1: {best_rec:.4f}")
#     print(f"F1 Score  @ best F1: {best_f1:.4f}")
#     acc = accuracy_score(trues, [1 if p >= best_thresh else 0 for p in probs])
#     print(f"Accuracy  = {acc:.4f}")

#     # Save threshold sweep to CSV
#     df_metrics = pd.DataFrame({
#         'threshold': th,
#         'precision': prec[:-1],
#         'recall': rec[:-1],
#         'f1': f1_scores[:-1]
#     })
#     df_metrics.to_csv(
#         os.path.join(base_dir, 'data', 'threshold_sweep.csv'),
#         index=False
#     )
# evaluate.py
#!/usr/bin/env python
# 0.51

# import os
# import yaml
# import pandas as pd
# import torch
# import torch.nn.functional as F
# from transformers import (
#     BertTokenizerFast,
#     BertConfig,
#     BertPreTrainedModel,
#     BertModel
# )
# from math import radians, sin, cos, sqrt, atan2
# from tqdm.auto import tqdm
# from sklearn.metrics import accuracy_score, precision_recall_curve
# from Levenshtein import ratio

# # Haversine distance function
# def haversine(lat1, lon1, lat2, lon2):
#     R = 6371000  # Earth radius in meters
#     phi1, phi2 = radians(lat1), radians(lat2)
#     dphi = radians(lat2 - lat1)
#     dlambda = radians(lon2 - lon1)
#     a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
#     return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# # Similarity model head - MUST match the training architecture
# class BertSimilarityHead(BertPreTrainedModel):
#     config_class = BertConfig

#     def __init__(self, config):
#         super().__init__(config)
#         self.bert = BertModel(config)
#         # Match the exact architecture from train.py
#         self.classifier = torch.nn.Sequential(
#             torch.nn.Linear(config.hidden_size + 1, 256),
#             torch.nn.GELU(),
#             torch.nn.Dropout(0.3),
#             torch.nn.Linear(256, 128),
#             torch.nn.GELU(),
#             torch.nn.Dropout(0.3),
#             torch.nn.Linear(128, config.num_labels),
#         )
#         self.init_weights()

#     def forward(self, input_ids, attention_mask=None, token_type_ids=None, edit_dist=None, **kwargs):
#         outputs = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             return_dict=True
#         )
#         pooled = outputs.pooler_output
#         edit_tensor = edit_dist.unsqueeze(1).to(pooled.dtype)
#         features = torch.cat([pooled, edit_tensor], dim=1)
#         return self.classifier(features)

# if __name__ == '__main__':
#     # Load config
#     base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#     with open(os.path.join(base_dir, 'configs', 'config.yaml')) as f:
#         cfg = yaml.safe_load(f)
#     inf_cfg = cfg.get('inference', {})

#     # Try multiple possible model directories
#     possible_dirs = [
#         os.path.join(base_dir, 'outputs'),  # ../outputs (where model actually is)
#         os.path.join(base_dir, cfg['training']['output_dir']),  # ../scripts/outputs
#         os.path.join(os.path.dirname(__file__), cfg['training']['output_dir']),  # ./scripts/outputs  
#         cfg['training']['output_dir'],  # scripts/outputs (relative)
#         os.path.abspath(cfg['training']['output_dir'])  # Absolute if provided
#     ]
    
#     model_dir = None
#     for dir_path in possible_dirs:
#         abs_path = os.path.abspath(dir_path)
#         print(f"Checking: {abs_path}")
        
#         # Check for pytorch_model.bin or model.safetensors
#         if (os.path.exists(os.path.join(abs_path, 'pytorch_model.bin')) or 
#             os.path.exists(os.path.join(abs_path, 'model.safetensors'))):
#             model_dir = abs_path
#             print(f"✓ Found model in: {model_dir}")
#             break
    
#     if model_dir is None:
#         print("Model not found in any expected location. Checking all possible files:")
#         for dir_path in possible_dirs:
#             abs_path = os.path.abspath(dir_path)
#             if os.path.exists(abs_path):
#                 print(f"\nFiles in {abs_path}:")
#                 try:
#                     files = os.listdir(abs_path)
#                     for f in files:
#                         print(f"  - {f}")
#                 except PermissionError:
#                     print(f"  Permission denied")
#             else:
#                 print(f"\nDirectory doesn't exist: {abs_path}")
        
#         raise FileNotFoundError(
#             f"Model checkpoint not found in any of these locations:\n" + 
#             "\n".join(f"  - {os.path.abspath(d)}" for d in possible_dirs) + 
#             "\n\nMake sure you've run training first: python train.py"
#         )

#     # Device, model, tokenizer
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
#     print(f"Loading model from: {model_dir}")
    
#     try:
#         bert_conf = BertConfig.from_pretrained(model_dir, num_labels=int(cfg['model']['num_labels']))
#         model = BertSimilarityHead.from_pretrained(model_dir, config=bert_conf).to(device)
#         model.eval()
#         tokenizer = BertTokenizerFast.from_pretrained(model_dir)
#         print("✓ Model and tokenizer loaded successfully")
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         raise

#     # Load data
#     data_files = {
#         'addresses': os.path.join(base_dir, 'data', 'addresses_corrected_filled.csv'),
#         'labels': os.path.join(base_dir, 'data', 'inference_labels.csv')
#     }
    
#     for name, path in data_files.items():
#         if not os.path.exists(path):
#             raise FileNotFoundError(f"{name.title()} file not found: {path}")
    
#     inf_df = pd.read_csv(data_files['addresses'])
#     labels_df = pd.read_csv(data_files['labels'])
#     dup_set = set(frozenset((r.orig_indexA, r.orig_indexB)) for r in labels_df.itertuples())
    
#     print(f"Loaded {len(inf_df)} addresses and {len(labels_df)} labeled pairs")

#     # Params
#     thresh = float(cfg['data']['distance_threshold_m'])
#     max_len = int(cfg['model']['max_length'])
#     batch_size = int(inf_cfg.get('batch_size', 32))
    
#     # Fix the infinity issue
#     max_pairs_config = inf_cfg.get('max_pairs', None)
#     if max_pairs_config is None or max_pairs_config == float('inf'):
#         max_pairs = float('inf')  # Keep as float
#     else:
#         max_pairs = int(max_pairs_config)
    
#     print(f"Distance threshold: {thresh}m")
#     print(f"Max sequence length: {max_len}")
#     print(f"Batch size: {batch_size}")
#     print(f"Max pairs to process: {max_pairs}")

#     probs, trues = [], []
#     buffer, pair_count = [], 0

#     def process_batch(batch):
#         texts_a = [x['a'] for x in batch]
#         texts_b = [x['b'] for x in batch]
#         edits = [ratio(a, b) for a, b in zip(texts_a, texts_b)]
#         enc = tokenizer(texts_a, texts_b, return_tensors='pt', padding='max_length', truncation=True, max_length=max_len)
#         enc = {k: v.to(device) for k, v in enc.items()}
#         edit_tensor = torch.tensor(edits, device=device)
#         with torch.no_grad():
#             logits = model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'], token_type_ids=enc['token_type_ids'], edit_dist=edit_tensor)
#             batch_probs = F.softmax(logits, dim=1)[:, 1].cpu().tolist()
#         for item, p in zip(batch, batch_probs):
#             pair = frozenset((item['i'], item['j']))
#             probs.append(p)
#             trues.append(1 if pair in dup_set else 0)

#     # Iterate pairs
#     print("Starting pairwise comparisons...")
#     for idx in tqdm(range(len(inf_df)), desc='Comparisons'):
#         row_a = inf_df.iloc[idx]
#         for jdx in range(idx+1, len(inf_df)):
#             if pair_count >= max_pairs:
#                 break
#             row_b = inf_df.iloc[jdx]
#             dist = haversine(row_a['lat'], row_a['long'], row_b['lat'], row_b['long'])
#             if dist > thresh:
#                 continue
#             buffer.append({'a': row_a['text'], 'b': row_b['text'], 'i': row_a['orig_index'], 'j': row_b['orig_index']})
#             pair_count += 1
#             if len(buffer) >= batch_size:
#                 process_batch(buffer)
#                 buffer.clear()
#         if pair_count >= max_pairs:
#             break

#     if buffer:
#         process_batch(buffer)

#     print(f"\nProcessed {pair_count} pairs total")
#     print(f"Found {sum(trues)} duplicate pairs out of {len(trues)} comparisons")

#     # Metrics
#     prec, rec, thresholds = precision_recall_curve(trues, probs)
#     f1s = 2 * prec * rec / (prec + rec + 1e-8)
#     best_idx = f1s.argmax()
    
#     print(f"\n=== RESULTS ===")
#     print(f"Best threshold = {thresholds[best_idx]:.3f}")
#     print(f"Precision = {prec[best_idx]:.4f}")
#     print(f"Recall = {rec[best_idx]:.4f}")
#     print(f"F1 = {f1s[best_idx]:.4f}")
    
#     acc = accuracy_score(trues, [1 if p>=thresholds[best_idx] else 0 for p in probs])
#     print(f"Accuracy = {acc:.4f}")

#     # Save results
#     output_path = os.path.join(base_dir, 'data', 'threshold_sweep.csv')
#     pd.DataFrame({
#         'threshold': thresholds,
#         'precision': prec[:-1],
#         'recall': rec[:-1],
#         'f1': f1s[:-1]
#     }).to_csv(output_path, index=False)
    
#     print(f"\nResults saved to: {output_path}")


# evaluate_jointbert.py
# Updated evaluation script for JointBERT architecture (three heads)

# evaluate_jointbert.py
# Updated evaluation script for JointBERT architecture (three heads)

# evaluate_jointbert.py
# Updated evaluation script for JointBERT architecture (three heads)

import os
import yaml
import pandas as pd
import torch
import torch.nn.functional as F
from math import radians, sin, cos, sqrt, atan2
from transformers import BertTokenizerFast, BertConfig, BertPreTrainedModel, BertModel
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, precision_recall_curve
from Levenshtein import ratio

# Haversine distance for geospatial filtering
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # radius in meters
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

class BertJointHead(BertPreTrainedModel):
    """
    Joint binary match head + dual entity-ID heads (edit_dist input)
    """
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        hidden = config.hidden_size
        # binary match head takes pooled + edit_dist
        self.match_head = torch.nn.Linear(hidden + 1, config.num_labels)
        # ID heads take only pooled CLS
        self.idA_head  = torch.nn.Linear(hidden, config.num_labels_entity)
        self.idB_head  = torch.nn.Linear(hidden, config.num_labels_entity)
        self.post_init()

    def forward(self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                edit_dist=None,
                **kwargs):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        pooled = out.pooler_output
        ed = edit_dist.unsqueeze(1).to(pooled.dtype)
        feats = torch.cat([pooled, ed], dim=1)
        logits_match = self.match_head(feats)
        logits_idA   = self.idA_head(pooled)
        logits_idB   = self.idB_head(pooled)
        return logits_match, logits_idA, logits_idB

if __name__ == '__main__':
    # Load configuration
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cfg_path = os.path.join(base_dir, 'configs', 'config.yaml')
    cfg = yaml.safe_load(open(cfg_path))
    inf_cfg = cfg.get('inference', {})

    # Locate trained model
    model_dir = os.path.abspath(cfg['training']['output_dir'])
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Determine num_labels_entity from training CSV
    train_csv = os.path.join(base_dir, cfg['data']['input_csv'])
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")
    df_train = pd.read_csv(train_csv)
    num_entities = int(df_train[['clusterA','clusterB']].max().max()) + 1
    print(f"Detected {num_entities} distinct clusters (entity classes)")

    # Load tokenizer and model config
    bert_conf = BertConfig.from_pretrained(model_dir,
                                           num_labels=int(cfg['model']['num_labels']))
    bert_conf.num_labels_entity = num_entities

    # Load model + tokenizer
    model = BertJointHead.from_pretrained(model_dir, config=bert_conf).to(device)
    model.eval()
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    print("Loaded JointBERT model & tokenizer")

    # Load inference data
    addr_path  = os.path.join(base_dir, cfg['data']['test_csv'])
    label_path = os.path.join(base_dir, cfg['data']['inference_labels'])
    inf_df  = pd.read_csv(addr_path)
    labels = pd.read_csv(label_path)
    dup_set = set(frozenset((r.orig_indexA, r.orig_indexB)) for r in labels.itertuples())

    # Parameters
    thresh    = float(cfg['data']['distance_threshold_m'])
    max_len   = int(cfg['model']['max_length'])
    batch_sz  = int(inf_cfg.get('batch_size', 32))
    max_pairs = inf_cfg.get('max_pairs', float('inf'))

    print(f"Distance threshold = {thresh}m, batch size = {batch_sz}")

    probs, trues = [], []
    buffer, count = [], 0

    def process_batch(batch):
        texts_a = [x['a'] for x in batch]
        texts_b = [x['b'] for x in batch]
        edits    = [ratio(a, b) for a, b in zip(texts_a, texts_b)]
        enc = tokenizer(texts_a, texts_b,
                        return_tensors='pt', padding='max_length', truncation=True,
                        max_length=max_len)
        enc = {k:v.to(device) for k,v in enc.items()}
        edit_t = torch.tensor(edits, device=device)
        with torch.no_grad():
            logits_match, _, _ = model(
                input_ids=enc['input_ids'],
                attention_mask=enc['attention_mask'],
                token_type_ids=enc['token_type_ids'],
                edit_dist=edit_t
            )
            scores = F.softmax(logits_match, dim=1)[:,1].cpu().tolist()
        for item, p in zip(batch, scores):
            pair = frozenset((item['i'], item['j']))
            probs.append(p)
            trues.append(1 if pair in dup_set else 0)

    # Generate candidate pairs
    print("Starting pairwise matching...")
    for i, row_a in tqdm(inf_df.iterrows(), total=len(inf_df)):
        for j in range(i+1, len(inf_df)):
            if count >= max_pairs:
                break
            row_b = inf_df.iloc[j]
            dist = haversine(row_a['lat'], row_a['long'], row_b['lat'], row_b['long'])
            if dist > thresh:
                continue
            buffer.append({'a': row_a['text'], 'b': row_b['text'],
                           'i': row_a['orig_index'], 'j': row_b['orig_index']})
            count += 1
            if len(buffer) >= batch_sz:
                process_batch(buffer)
                buffer.clear()
        if count >= max_pairs:
            break

    if buffer:
        process_batch(buffer)

    print(f"Processed {count} pairs, found {sum(trues)} positives")

    # Compute metrics and threshold sweep
    prec, rec, ths = precision_recall_curve(trues, probs)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    best = f1s.argmax()
    best_t = ths[best]

    print("\n=== RESULTS ===")
    print(f"Best threshold: {best_t:.3f}")
    print(f"Precision: {prec[best]:.4f}, Recall: {rec[best]:.4f}, F1: {f1s[best]:.4f}")
    acc = accuracy_score(trues, [1 if p>=best_t else 0 for p in probs])
    print(f"Accuracy: {acc:.4f}")

    # Ensure output directory exists
    out_dir = os.path.join(base_dir, cfg['training']['output_dir'])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'threshold_sweep.csv')
    
    # Save threshold sweep
    out = pd.DataFrame({'threshold': ths, 'precision': prec[:-1], 'recall': rec[:-1], 'f1': f1s[:-1]})
    out.to_csv(out_path, index=False)
    print(f"Threshold sweep saved to: {out_path}")


